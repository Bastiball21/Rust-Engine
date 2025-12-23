use crate::parameters::SearchParameters;
use crate::search;
use crate::state::{GameState, WHITE, BOTH, K, k, N, B, n, b, NO_PIECE};
use crate::tt::TranspositionTable;
use rand::{Rng, SeedableRng, RngCore};
use rand::rngs::StdRng;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

// Tuning Configuration
const GAMES_PER_ITERATION: usize = 1000; // Must be even (pairs)
const DEFAULT_A: f64 = 500.0; // 10 * max_iter (assuming 50) ? Tunable.
const DEFAULT_C: f64 = 0.5;
const DEFAULT_ALPHA: f64 = 0.602;
const DEFAULT_GAMMA: f64 = 0.101;

#[derive(Serialize, Deserialize, Debug)]
struct SpsaState {
    iteration: usize,
    params: SearchParameters,
    // Hyperparameters
    learning_rate_scale: f64, // a
    perturbation_scale: f64,  // c
    decay_offset: f64,        // A
    alpha: f64,
    gamma: f64,
    // RNG State (seed)
    rng_seed: u64,
    // History for logging
    history: Vec<SpsaLogEntry>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct SpsaLogEntry {
    iteration: usize,
    score_diff: f64, // (wins_plus - wins_minus) / games
    gradient_norm: f64,
}

impl SpsaState {
    fn new() -> Self {
        SpsaState {
            iteration: 1,
            params: SearchParameters::default(),
            learning_rate_scale: 0.5,
            perturbation_scale: DEFAULT_C,
            decay_offset: DEFAULT_A,
            alpha: DEFAULT_ALPHA,
            gamma: DEFAULT_GAMMA,
            rng_seed: 42,
            history: Vec::new(),
        }
    }

    fn load(path: &str) -> Self {
        if let Ok(file) = File::open(path) {
            let reader = BufReader::new(file);
            if let Ok(state) = serde_json::from_reader::<BufReader<File>, SpsaState>(reader) {
                println!("Resuming SPSA from iteration {}", state.iteration);
                return state;
            }
        }
        println!("Starting fresh SPSA tuning.");
        Self::new()
    }

    fn save(&self, path: &str) {
        if let Ok(file) = File::create(path) {
            serde_json::to_writer_pretty(file, self).unwrap();
        }
    }
}

pub fn run_tuning() {
    // Attempt to load NNUE for evaluation accuracy, fallback to HCE if missing
    crate::nnue::init_nnue("nn-aether.nnue");

    let mut state = SpsaState::load("spsa_state.json");
    let mut rng = StdRng::seed_from_u64(state.rng_seed);

    println!("--- SPSA Tuner Started ---");
    println!("Params: a={}, c={}, A={}, alpha={}, gamma={}",
             state.learning_rate_scale, state.perturbation_scale, state.decay_offset, state.alpha, state.gamma);

    loop {
        let iter_idx = state.iteration as f64;
        let ak = state.learning_rate_scale / (state.decay_offset + iter_idx).powf(state.alpha);
        let ck = state.perturbation_scale / iter_idx.powf(state.gamma);

        let delta = generate_delta(&mut rng);

        // Perturb
        let mut theta_plus = state.params.clone();
        apply_perturbation(&mut theta_plus, &delta, ck);
        constrain_params(&mut theta_plus);

        let mut theta_minus = state.params.clone();
        apply_perturbation(&mut theta_minus, &delta, -ck);
        constrain_params(&mut theta_minus);

        println!("Iter {}: Running {} games... ck={:.4}, ak={:.4}", state.iteration, GAMES_PER_ITERATION, ck, ak);

        let start = Instant::now();
        // Run Batch: Theta+ vs Theta-
        // We use shared openings and same seeds for game logic if possible (engine is deterministic)
        let score_diff = run_match_batch(&theta_plus, &theta_minus, GAMES_PER_ITERATION, &mut rng);

        let elapsed = start.elapsed().as_secs_f64();
        let games_sec = GAMES_PER_ITERATION as f64 / elapsed;

        // Gradient Step
        let step_size = score_diff / (2.0 * ck);
        apply_gradient(&mut state.params, &delta, step_size * ak);
        constrain_params(&mut state.params);

        // Logging
        let grad_norm = step_size.abs(); // Simplified norm
        println!("Iter {} Done. Score Diff: {:.4}. Grad Norm: {:.4}. Speed: {:.1} gps",
                 state.iteration, score_diff, grad_norm, games_sec);

        state.history.push(SpsaLogEntry {
            iteration: state.iteration,
            score_diff,
            gradient_norm: grad_norm,
        });

        state.iteration += 1;
        state.rng_seed = rng.next_u64(); // Advance seed
        state.save("spsa_state.json");

        // Also save readable params
        state.params.save_to_json("spsa_current_params.json").ok();
    }
}

const PARAM_COUNT: usize = 7;

fn generate_delta(rng: &mut impl Rng) -> Vec<f64> {
    let mut d = Vec::with_capacity(PARAM_COUNT);
    for _ in 0..PARAM_COUNT {
        d.push(if rng.random_bool(0.5) { 1.0 } else { -1.0 });
    }
    d
}

fn apply_perturbation(params: &mut SearchParameters, delta: &[f64], scale: f64) {
    params.lmr_base += delta[0] * scale;
    params.lmr_divisor += delta[1] * scale;

    params.nmp_base = (params.nmp_base as f64 + delta[2] * scale).round() as i32;
    params.nmp_divisor = (params.nmp_divisor as f64 + delta[3] * scale).round() as i32;

    params.rfp_margin = (params.rfp_margin as f64 + delta[4] * scale * 10.0).round() as i32;
    params.razoring_base = (params.razoring_base as f64 + delta[5] * scale * 10.0).round() as i32;
    params.razoring_multiplier = (params.razoring_multiplier as f64 + delta[6] * scale * 10.0).round() as i32;

    params.recalculate_tables();
}

fn apply_gradient(params: &mut SearchParameters, delta: &[f64], step: f64) {
    apply_perturbation(params, delta, step);
}

fn constrain_params(params: &mut SearchParameters) {
    params.lmr_base = params.lmr_base.max(0.0).min(5.0);
    params.lmr_divisor = params.lmr_divisor.max(1.0).min(10.0);
    params.nmp_base = params.nmp_base.max(0).min(10);
    params.nmp_divisor = params.nmp_divisor.max(1).min(10);
    params.rfp_margin = params.rfp_margin.max(0).min(200);
    params.razoring_base = params.razoring_base.max(0).min(1000);
    params.razoring_multiplier = params.razoring_multiplier.max(0).min(500);
    params.recalculate_tables();
}

// Match Execution
fn run_match_batch(p1: &SearchParameters, p2: &SearchParameters, games: usize, rng: &mut StdRng) -> f64 {
    let num_threads = std::thread::available_parallelism().unwrap().get().max(1);
    let pairs = games / 2;

    // Generate openings
    let mut openings = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        openings.push(generate_random_opening(rng));
    }

    let p1_arc = Arc::new(p1.clone());
    let p2_arc = Arc::new(p2.clone());
    let openings_arc = Arc::new(openings);

    let total_score = Arc::new(Mutex::new(0.0));
    let mut handles = vec![];

    let chunk_size = (pairs + num_threads - 1) / num_threads;

    for i in 0..num_threads {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(pairs);
        if start >= end { break; }

        let my_p1 = p1_arc.clone();
        let my_p2 = p2_arc.clone();
        let my_ops = openings_arc.clone();
        let my_score = total_score.clone();

        let builder = thread::Builder::new().stack_size(8 * 1024 * 1024);
        handles.push(builder.spawn(move || {
            let mut local_score = 0.0;
            let mut tt = TranspositionTable::new(1, 1); // 1MB private TT

            for idx in start..end {
                let root = my_ops[idx];

                // Game 1: P1 White, P2 Black
                tt.clear();
                tt.new_search();
                let s1 = play_game(&my_p1, &my_p2, root, &tt);
                local_score += s1;

                // Game 2: P2 White, P1 Black
                tt.clear();
                tt.new_search();
                let s2 = play_game(&my_p2, &my_p1, root, &tt);
                local_score += 1.0 - s2; // P1 score (Black)
            }

            let mut lock = my_score.lock().unwrap();
            *lock += local_score;
        }).unwrap());
    }

    for h in handles {
        h.join().unwrap();
    }

    let final_score = *total_score.lock().unwrap();
    // Normalize to [-1, 1]
    (final_score / (pairs * 2) as f64) * 2.0 - 1.0
}

fn generate_random_opening(rng: &mut impl Rng) -> GameState {
    let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    // 8-ply random walk
    for _ in 0..8 {
         let mut moves = crate::movegen::MoveGenerator::new();
         moves.generate_moves(&state);
         if moves.list.count == 0 { break; }
         let idx = rng.random_range(0..moves.list.count);
         state = state.make_move(moves.list.moves[idx]);
    }
    state
}

fn play_game(
    white: &SearchParameters,
    black: &SearchParameters,
    start_pos: GameState,
    tt: &TranspositionTable
) -> f64 {
    let mut state = start_pos;
    let mut history = vec![state.hash];
    let mut rep = std::collections::HashMap::new();
    rep.insert(state.hash, 1);

    // Increased node count to reduce draw rate and noise
    let limit = search::Limits::FixedNodes(10000);
    let mut sd = search::SearchData::new();
    let stop = Arc::new(AtomicBool::new(false));

    let mut consecutive_draw_score = 0;

    for _ in 0..200 {
        if state.halfmove_clock >= 100 || is_material_draw(&state) { return 0.5; }
        if let Some(&c) = rep.get(&state.hash) { if c >= 3 { return 0.5; } }

        let params = if state.side_to_move == WHITE { white } else { black };

        let (score, best_move) = search::search(
            &state,
            limit,
            tt,
            stop.clone(),
            false,
            history.clone(),
            &mut sd,
            params,
            None,
            None,
        );

        if score.abs() < 10 {
            consecutive_draw_score += 1;
        } else {
            consecutive_draw_score = 0;
        }
        if consecutive_draw_score >= 20 { return 0.5; }

        if let Some(mv) = best_move {
            state = state.make_move(mv);
            history.push(state.hash);
            *rep.entry(state.hash).or_insert(0) += 1;

            if search::is_in_check(&state) && search::is_check(&state, state.side_to_move) {
                 // Mate
            }
        } else {
            return if search::is_in_check(&state) {
                if state.side_to_move == WHITE { 0.0 } else { 1.0 }
            } else {
                0.5
            };
        }
    }
    0.5
}

fn is_material_draw(state: &GameState) -> bool {
    if state.occupancies[BOTH].count_bits() == 2 { return true; }
    if state.occupancies[BOTH].count_bits() == 3 {
        let others = state.occupancies[BOTH] ^ state.bitboards[K] ^ state.bitboards[k];
        if others.count_bits() == 1 {
             let sq = others.get_lsb_index();
             if state.bitboards[N].get_bit(sq as u8) || state.bitboards[B].get_bit(sq as u8) ||
                state.bitboards[n].get_bit(sq as u8) || state.bitboards[b].get_bit(sq as u8) {
                 return true;
             }
        }
    }
    false
}
