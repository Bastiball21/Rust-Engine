use crate::parameters::SearchParameters;
use crate::search;
use crate::state::{GameState, WHITE, BLACK, P, N, B, R, Q, K, p, n, b, r, q, k, BOTH};
use crate::tt::TranspositionTable;
use rand::Rng; // Trait must be in scope for generic bounds
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

    let mut rng = rand::rng();

    for epoch in 1..=EPOCHS {
        // 1. Calculate ak and ck
        let ak = SPSA_A / (epoch as f64 + 50.0).powf(SPSA_ALPHA);
        let ck = SPSA_C / (epoch as f64).powf(SPSA_GAMMA);

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

        println!("Epoch {}: Score Diff = {:.4}, ck = {:.4}, ak = {:.4}", epoch, score_diff, ck, ak);

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

    // Ensure even number of games for pairs
    let num_pairs = games / 2;
    let pairs_per_thread = num_pairs / num_threads;
    let remainder = num_pairs % num_threads;

    let p1_arc = Arc::new(p1.clone());
    let p2_arc = Arc::new(p2.clone());

    // Generate Openings
    let mut openings = Vec::with_capacity(num_pairs);
    let mut rng = rand::rng();
    let start_state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    for _ in 0..num_pairs {
        let mut state = start_state.clone();
        let mut hist = std::collections::HashMap::new();
        hist.insert(state.hash, 1);

        let mut moves = crate::movegen::MoveGenerator::new();
        let mut valid = true;

        for _ in 0..8 {
            moves.generate_moves(&state);
            if moves.list.count == 0 {
                valid = false;
                break;
            }
            let idx = rng.random_range(0..moves.list.count);
            let m = moves.list.moves[idx];
            state = state.make_move(m); // Legacy make_move is fine for setup

            // Check for early repetition or material draw in opening (unlikely but safe)
            if *hist.entry(state.hash).or_insert(0) >= 3 {
                 valid = false;
                 break;
            }
        }

        if valid {
            openings.push(state);
        } else {
            // Retry or just push startpos (fallback)
            openings.push(start_state);
        }
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let mut handles = vec![];

    // Create TTs (1MB each)
    let mut tts = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        tts.push(TranspositionTable::new(1));
    }

    let mut start_idx = 0;

    for i in 0..num_threads {
        let tx = tx.clone();
        let p1 = p1_arc.clone();
        let p2 = p2_arc.clone();

        // Distribute openings
        let count = pairs_per_thread + if i < remainder { 1 } else { 0 };
        let thread_openings = openings[start_idx..start_idx + count].to_vec();
        start_idx += count;

        let mut tt = tts.pop().unwrap(); // Move TT into thread

        let builder = thread::Builder::new().stack_size(32 * 1024 * 1024);
        handles.push(builder.spawn(move || {
            let mut wins = 0.0;

            for opening in thread_openings {
                // Game A: P1 White, P2 Black
                let score_a = play_game(&p1, &p2, &opening, &mut tt);

                // Game B: P2 White, P1 Black
                let score_b = play_game(&p2, &p1, &opening, &mut tt);

                // P1 Score = Score A (as White) + (1.0 - Score B (as Black))
                wins += score_a + (1.0 - score_b);
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

    // Total Games = num_pairs * 2
    // Wins is sum of P1 points.
    let total_games = (num_pairs * 2) as f64;
    let p1_score = total_wins / total_games;

    // Normalized score [-1, 1]
    p1_score * 2.0 - 1.0
}

fn is_material_draw(state: &GameState) -> bool {
    let occ = state.occupancies[BOTH];
    let count = occ.count_bits();

    if count > 3 { return false; }

    // K vs K
    if count == 2 { return true; }

    // KB vs K or KN vs K
    // Remove kings
    let kings = state.bitboards[K] | state.bitboards[k];
    let others = occ.0 & !kings.0;

    if others == 0 { return true; } // Just kings

    let knights = state.bitboards[N] | state.bitboards[n];
    let bishops = state.bitboards[B] | state.bitboards[b];

    // Check if remaining piece is Knight
    if (others & knights.0) != 0 && (others & !knights.0) == 0 {
        return true;
    }

    // Check if remaining piece is Bishop
    if (others & bishops.0) != 0 && (others & !bishops.0) == 0 {
        return true;
    }

    false
}

// Plays one game. Returns 1.0 if White wins, 0.0 if Black wins, 0.5 draw.
fn play_game(
    white_params: &Arc<SearchParameters>,
    black_params: &Arc<SearchParameters>,
    start_state: &GameState,
    tt: &mut TranspositionTable
) -> f64 {
    tt.clear(); // Clear TT before game start to ensure stability

    let mut state = *start_state;

    let mut history = vec![state.hash];
    let mut rep_history = std::collections::HashMap::new();
    rep_history.insert(state.hash, 1);

    // Deep tuning depth
    let limit = search::Limits::FixedNodes(5000);

    let mut sd_white = search::SearchData::new();
    let mut sd_black = search::SearchData::new();

    let stop = Arc::new(AtomicBool::new(false));

    let mut white_wins = 0;
    let mut black_wins = 0;

    for _ in 0..200 { // Max 200 moves
        // Material Draw Check
        if is_material_draw(&state) {
            return 0.5;
        }

        if state.halfmove_clock >= 100 { return 0.5; }
        if let Some(&c) = rep_history.get(&state.hash) {
            if c >= 3 { return 0.5; }
        }

        let params = if state.side_to_move == WHITE { white_params } else { black_params };
        let sd = if state.side_to_move == WHITE { &mut sd_white } else { &mut sd_black };

        let (score, best_move) = search::search(
            &state,
            limit,
            tt, // Pass as immutable ref (Search uses atomics)
            stop.clone(),
            false, // not main thread
            history.clone(),
            sd,
            params
        );

        if let Some(mv) = best_move {
            // Score Adjudication (Absolute White Perspective)
            let white_score = if state.side_to_move == WHITE { score } else { -score };

            if white_score > 400 {
                white_wins += 1;
                black_wins = 0;
            } else if white_score < -400 {
                black_wins += 1;
                white_wins = 0;
            } else {
                white_wins = 0;
                black_wins = 0;
            }

            if white_wins >= 5 { return 1.0; } // White Wins
            if black_wins >= 5 { return 0.0; } // Black Wins (White Loses)

            state = state.make_move(mv);
            history.push(state.hash);
            *rep_history.entry(state.hash).or_insert(0) += 1;
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
