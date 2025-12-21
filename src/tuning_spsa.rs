use crate::parameters::SearchParameters;
use crate::search;
use crate::state::{GameState, WHITE};
use crate::tt::TranspositionTable;
use rand::Rng; // Trait must be in scope for generic bounds, even if we use updated methods
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;

// --- Tuning Config ---
const EPOCHS: usize = 100;
const GAMES_PER_EPOCH: usize = 1000;
const SPSA_C: f64 = 0.5; // Perturbation size scaling
const SPSA_A: f64 = 0.5; // Learning rate scaling
const SPSA_ALPHA: f64 = 0.602;
const SPSA_GAMMA: f64 = 0.101;

pub fn run_tuning() {
    println!("--- SPSA Tuner ---");
    println!("Running {} epochs of {} games.", EPOCHS, GAMES_PER_EPOCH);

    let mut params = SearchParameters::default();

    if let Ok(loaded) = SearchParameters::load_from_json("spsa_params.json") {
        println!("Loaded parameters from spsa_params.json");
        params = loaded;
    } else {
        println!("Starting from default parameters.");
    }

    // rand 0.9.x: use rand::rng()
    let mut rng = rand::rng();

    for k in 1..=EPOCHS {
        // 1. Calculate ak and ck
        let ak = SPSA_A / (k as f64 + 50.0).powf(SPSA_ALPHA);
        let ck = SPSA_C / (k as f64).powf(SPSA_GAMMA);

        // 2. Generate Delta (Bernoulli +/- 1)
        let delta = generate_delta(&mut rng);

        // 3. Create Perturbed Parameters
        let mut theta_plus = params.clone();
        apply_perturbation(&mut theta_plus, &delta, ck);

        let mut theta_minus = params.clone();
        apply_perturbation(&mut theta_minus, &delta, -ck);

        // 4. Run Matches (A vs B)
        let score_diff = run_match_batch(&theta_plus, &theta_minus, GAMES_PER_EPOCH);

        // Gradient Estimate: g = (y+ - y-) / (2 * ck) * delta
        let gradient_step = score_diff / (2.0 * ck);

        println!("Epoch {}: Score Diff = {:.4}, ck = {:.4}, ak = {:.4}", k, score_diff, ck, ak);

        apply_gradient(&mut params, &delta, gradient_step * ak);

        params.save_to_json("spsa_params.json").unwrap();
        println!("Parameters saved.");
    }
}

// Map parameters to a vector for perturbation
// 0: lmr_base
// 1: lmr_divisor
// 2: nmp_base
// 3: nmp_divisor
// 4: rfp_margin
// 5: razoring_base
// 6: razoring_multiplier
const PARAM_COUNT: usize = 7;

// Updated for rand 0.9.x
fn generate_delta(rng: &mut impl rand::Rng) -> Vec<f64> {
    let mut d = Vec::with_capacity(PARAM_COUNT);
    for _ in 0..PARAM_COUNT {
        if rng.random_bool(0.5) {
            d.push(1.0);
        } else {
            d.push(-1.0);
        }
    }
    d
}

fn apply_perturbation(params: &mut SearchParameters, delta: &[f64], scale: f64) {
    params.lmr_base += delta[0] * scale;
    params.lmr_divisor += delta[1] * scale;

    params.nmp_base = (params.nmp_base as f64 + delta[2] * scale).round() as i32;
    params.nmp_divisor = (params.nmp_divisor as f64 + delta[3] * scale).round() as i32;

    // Scale integer margins more heavily to ensure movement
    params.rfp_margin = (params.rfp_margin as f64 + delta[4] * scale * 10.0).round() as i32;
    params.razoring_base = (params.razoring_base as f64 + delta[5] * scale * 10.0).round() as i32;
    params.razoring_multiplier = (params.razoring_multiplier as f64 + delta[6] * scale * 10.0).round() as i32;

    params.recalculate_tables();
}

fn apply_gradient(params: &mut SearchParameters, delta: &[f64], step: f64) {
    params.lmr_base += delta[0] * step;
    params.lmr_divisor += delta[1] * step;

    params.nmp_base = (params.nmp_base as f64 + delta[2] * step).round() as i32;
    params.nmp_divisor = (params.nmp_divisor as f64 + delta[3] * step).round() as i32;

    params.rfp_margin = (params.rfp_margin as f64 + delta[4] * step * 10.0).round() as i32;
    params.razoring_base = (params.razoring_base as f64 + delta[5] * step * 10.0).round() as i32;
    params.razoring_multiplier = (params.razoring_multiplier as f64 + delta[6] * step * 10.0).round() as i32;

    params.recalculate_tables();

    println!("Updated Params: {:?}", params);
}

// Runs a batch of games and returns (Wins - Losses) / Total (from perspective of P1)
fn run_match_batch(p1: &SearchParameters, p2: &SearchParameters, games: usize) -> f64 {
    let num_threads = std::thread::available_parallelism().unwrap().get().max(1);
    let games_per_thread = games / num_threads;

    let p1_arc = Arc::new(p1.clone());
    let p2_arc = Arc::new(p2.clone());

    let (tx, rx) = std::sync::mpsc::channel();

    let mut handles = vec![];

    for _ in 0..num_threads {
        let tx = tx.clone();
        let p1 = p1_arc.clone();
        let p2 = p2_arc.clone();

        let builder = thread::Builder::new().stack_size(32 * 1024 * 1024);
        handles.push(builder.spawn(move || {
            let mut wins = 0.0;
            let my_games = games_per_thread;
            for i in 0..my_games {
                // Alternate colors
                let (white, black) = if i % 2 == 0 {
                    (p1.clone(), p2.clone())
                } else {
                    (p2.clone(), p1.clone())
                };

                let result = play_game(&white, &black);

                if i % 2 == 0 {
                    // P1 is white
                    wins += result; // 1.0 if white wins
                } else {
                    // P1 is black
                    wins += 1.0 - result; // 1.0 if black wins (result is 0.0)
                }
            }
            tx.send(wins).unwrap();
        }).unwrap());
    }

    let mut total_wins = 0.0;
    for _ in 0..num_threads {
        total_wins += rx.recv().unwrap();
    }

    for h in handles {
        h.join().unwrap();
    }

    // Normalized score [-1, 1]
    let p1_score = total_wins / (games_per_thread * num_threads) as f64;
    p1_score * 2.0 - 1.0
}

// Plays one game. Returns 1.0 if White wins, 0.0 if Black wins, 0.5 draw.
fn play_game(white_params: &Arc<SearchParameters>, black_params: &Arc<SearchParameters>) -> f64 {
    let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // rand 0.9.x
    let mut rng = rand::rng();
    let mut history = vec![state.hash];
    let mut rep_history = std::collections::HashMap::new();
    rep_history.insert(state.hash, 1);

    // 8 ply random opening
    for _ in 0..8 {
         let mut moves = crate::movegen::MoveGenerator::new();
         moves.generate_moves(&state);
         if moves.list.count == 0 { return 0.5; }

         let idx = rng.random_range(0..moves.list.count);
         let m = moves.list.moves[idx];
         state = state.make_move(m);
         history.push(state.hash);
         *rep_history.entry(state.hash).or_insert(0) += 1;
    }

    let mut tt = TranspositionTable::new(1);

    // Fast tuning match parameters (e.g. 1000 nodes)
    let limit = search::Limits::FixedNodes(1000);

    let mut sd_white = search::SearchData::new();
    let mut sd_black = search::SearchData::new();

    let stop = Arc::new(AtomicBool::new(false));

    for _ in 0..200 { // Max 200 moves
        if state.halfmove_clock >= 100 { return 0.5; }
        if let Some(&c) = rep_history.get(&state.hash) {
            if c >= 3 { return 0.5; }
        }

        let params = if state.side_to_move == WHITE { white_params } else { black_params };
        let sd = if state.side_to_move == WHITE { &mut sd_white } else { &mut sd_black };

        let (_, best_move) = search::search(
            &state,
            limit,
            &tt,
            stop.clone(),
            false, // not main thread
            history.clone(),
            sd,
            params
        );

        if let Some(mv) = best_move {
            state = state.make_move(mv);
            history.push(state.hash);
            *rep_history.entry(state.hash).or_insert(0) += 1;

            // Adjudication could be added here
        } else {
            // No move -> Mate or Stalemate
            if search::is_in_check(&state) {
                return if state.side_to_move == WHITE { 0.0 } else { 1.0 }; // Loss for current side
            } else {
                return 0.5;
            }
        }
    }

    0.5
}
