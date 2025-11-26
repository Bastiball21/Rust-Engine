use crate::state::{GameState, WHITE};
use crate::eval::{self, PARAMS, EvalParams};
use crate::search;
use crate::tt::TranspositionTable;
use crate::time::{TimeManager, TimeControl};
use std::sync::{Arc, atomic::AtomicBool};
use std::fs::File;
use std::io::{self, BufRead};

pub fn run_tuning() {
    println!("Loading dataset from 'tuning.epd'...");
    let positions = load_epd("tuning.epd");
    if positions.is_empty() {
        println!("No positions found. Make sure 'tuning.epd' exists.");
        return;
    }
    println!("Loaded {} positions.", positions.len());

    // Initial error
    let initial_error = compute_total_error(&positions);
    println!("Initial Error: {:.6}", initial_error);

    // Simple Local Search Tuning
    // We tweak each parameter by a small amount (+1/-1) and see if error drops.
    let mut best_error = initial_error;
    let mut improved = true;
    let mut iteration = 0;

    while improved {
        improved = false;
        iteration += 1;
        println!("Iteration {}...", iteration);

        unsafe {
            // Tune Material
            for i in 0..6 {
                if try_improve_param(&mut PARAMS.material[i], &mut best_error, &positions) { improved = true; }
            }
            // Tune Mobility
            for i in 0..4 {
                if try_improve_param(&mut PARAMS.mobility_bonus[i], &mut best_error, &positions) { improved = true; }
            }
            // Tune Pawn PSQT (Center squares only for speed test)
            for i in 24..40 {
                if try_improve_param(&mut PARAMS.pawn_table[i], &mut best_error, &positions) { improved = true; }
            }
        }

        if improved {
            println!("New Best Error: {:.6}", best_error);
            unsafe { print_params(&PARAMS); }
        }
    }
    println!("Tuning Complete.");
}

struct TunerEntry {
    state: GameState,
    result: f64, // 1.0 (Win), 0.5 (Draw), 0.0 (Loss)
}

fn load_epd(path: &str) -> Vec<TunerEntry> {
    let mut entries = Vec::new();
    if let Ok(file) = File::open(path) {
        for line in io::BufReader::new(file).lines() {
            if let Ok(l) = line {
                let parts: Vec<&str> = l.split_whitespace().collect();
                // EPD format: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 c0 "1.0";
                // Simplified parse:
                let fen = format!("{} {} {} {} {} {}", parts[0], parts[1], parts[2], parts[3], parts[4], parts[5]);
                let result_str = parts.last().unwrap().trim_matches(|c| c == '"' || c == ';');
                let result = match result_str {
                    "1-0" | "1.0" => 1.0,
                    "0-1" | "0.0" => 0.0,
                    "1/2-1/2" | "0.5" => 0.5,
                    _ => 0.5, 
                };
                entries.push(TunerEntry { state: GameState::parse_fen(&fen), result });
            }
        }
    }
    entries
}

fn sigmoid(score: f64) -> f64 {
    1.0 / (1.0 + 10.0f64.powf(-score / 400.0))
}

fn compute_total_error(positions: &[TunerEntry]) -> f64 {
    let mut total_error = 0.0;
    for entry in positions {
        // Use Quiescence Search for static eval to account for tactical noise
        let score = quiescence_eval(&entry.state);
        let sigmoid_score = sigmoid(score as f64);
        let diff = entry.result - sigmoid_score;
        total_error += diff * diff;
    }
    total_error / positions.len() as f64
}

// Helper to run just Q-Search for scoring
fn quiescence_eval(state: &GameState) -> i32 {
    // Create dummy structures
    let mut tt = TranspositionTable::new(1);
    let tm = TimeManager::new(TimeControl::Infinite, state.side_to_move);
    let stop = Arc::new(AtomicBool::new(false));
    let mut info = search::SearchInfo::new(tm, stop);
    
    // Use evaluate() directly for speed in this simplified tuner, 
    // OR call q-search if exposed. For Texel, Eval() is usually enough if dataset is "quiet" positions.
    // Let's use the raw eval function we just made tunable.
    eval::evaluate(state)
}

fn try_improve_param(param: &mut i32, best_error: &mut f64, positions: &[TunerEntry]) -> bool {
    let original = *param;
    let mut improved = false;

    // Try +1
    *param = original + 1;
    let err_plus = compute_total_error(positions);
    if err_plus < *best_error {
        *best_error = err_plus;
        return true;
    }

    // Try -1
    *param = original - 1;
    let err_minus = compute_total_error(positions);
    if err_minus < *best_error {
        *best_error = err_minus;
        return true;
    }

    // Reset if no improvement
    *param = original;
    false
}

fn print_params(p: &EvalParams) {
    println!("Current Material: {:?}", p.material);
    // Print others as needed
}