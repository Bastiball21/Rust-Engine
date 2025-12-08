// src/datagen.rs
use crate::state::{GameState, WHITE};
use crate::search;
use crate::tt::TranspositionTable;
use crate::time::{TimeManager, TimeControl};
use std::sync::{Arc, atomic::AtomicBool};
use std::fs::OpenOptions;
use std::io::Write;

pub fn run_datagen(games: usize) {
    println!("Starting Datagen for {} games at Depth 6...", games);
    let mut file = OpenOptions::new().create(true).append(true).open("aether_data.txt").unwrap();

    // 64MB TT is good for Depth 6
    let mut tt = TranspositionTable::new(64);
    let stop = Arc::new(AtomicBool::new(false));

    for i in 0..games {
        let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let mut fens = Vec::new();
        let result_val;

        loop {
            let mut moves = crate::movegen::MoveGenerator::new();
            moves.generate_moves(&state);

            // 1. Check Game Over
            if moves.list.count == 0 {
                if crate::search::is_in_check(&state) {
                    result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                } else {
                    result_val = 0.5;
                }
                break;
            }
            if state.halfmove_clock >= 100 {
                result_val = 0.5;
                break;
            }

            // 2. Set Depth (Random opening vs Real search)
            // FIX: Changed from 4 to 6 for better quality
            let depth = if state.fullmove_number < 8 { 1 } else { 6 };

            let tm = TimeManager::new(TimeControl::MoveTime(50), state.side_to_move, 0);

            // Pass true for main_thread to see output if needed, but for now we keep false to reduce spam.
            // If it hangs, we can change to true.
            search::search(&state, tm, &tt, stop.clone(), depth, false, vec![]);

            // 3. Get Score & Best Move
            let (tt_score, _, _, best_move_opt) = tt.probe_data(state.hash).unwrap_or((0, 0, 0, None));
            let best_move = best_move_opt.unwrap_or(moves.list.moves[0]);

            // --- ADJUDICATION ---
            if tt_score.abs() > 20000 {
                if tt_score > 0 {
                    result_val = if state.side_to_move == WHITE { 1.0 } else { 0.0 };
                } else {
                    result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                }
                break;
            }

            // Save Data: FEN | SCORE | RESULT
            let white_relative_score = if state.side_to_move == WHITE { tt_score } else { -tt_score };

            if state.fullmove_number > 8 {
                fens.push(format!("{} | {} | ", state.to_fen(), white_relative_score));
            }

            state = state.make_move(best_move);
        }

        // Write to file
        for line in fens {
            writeln!(file, "{}{}", line, result_val).unwrap();
        }

        // Print progress
        println!("Generated game {} / {} (Result: {})", i+1, games, result_val);

        tt.clear();
    }
}