// src/datagen.rs
use crate::search;
use crate::state::{GameState, WHITE};
use crate::time::{TimeControl, TimeManager};
use crate::tt::TranspositionTable;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::sync::{atomic::AtomicBool, Arc};
use crate::bullet_helper::convert_to_bullet;
use bulletformat::BulletFormat;

pub fn run_datagen(games: usize) {
    println!("Starting Datagen for {} games at Depth 6...", games);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("aether_data.bin")
        .unwrap();

    let mut writer = BufWriter::new(file);

    // 64MB TT is good for Depth 6
    let mut tt = TranspositionTable::new(64);
    let stop = Arc::new(AtomicBool::new(false));

    for i in 0..games {
        let mut state =
            GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

        let mut positions: Vec<(GameState, i16)> = Vec::new();
        let result_val;

        loop {
            let mut moves = crate::movegen::MoveGenerator::new();
            moves.generate_moves(&state);

            // 1. Check Game Over
            if moves.list.count == 0 {
                if crate::search::is_in_check(&state) {
                    result_val = if state.side_to_move == WHITE {
                        0.0
                    } else {
                        1.0
                    };
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
            let depth = if state.fullmove_number < 8 { 1 } else { 6 };

            let tm = TimeManager::new(TimeControl::MoveTime(50), state.side_to_move, 0);
            search::search(&state, tm, &tt, stop.clone(), depth, false, vec![]);

            // 3. Get Score & Best Move
            let (tt_score, _, _, best_move_opt) =
                tt.probe_data(state.hash).unwrap_or((0, 0, 0, None));
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
            let white_relative_score = if state.side_to_move == WHITE {
                tt_score
            } else {
                -tt_score
            };

            if state.fullmove_number > 8 {
                positions.push((state.clone(), white_relative_score as i16));
            }

            state = state.make_move(best_move);
        }

        // Buffer for this game
        let mut game_data = Vec::with_capacity(positions.len());
        for (pos_state, score) in positions {
            let board = convert_to_bullet(&pos_state, score, result_val);
            game_data.push(board);
        }

        // Write batch
        bulletformat::ChessBoard::write_to_bin(&mut writer, &game_data).unwrap();

        // Print progress
        println!(
            "Generated game {} / {} (Result: {})",
            i + 1,
            games,
            result_val
        );

        tt.clear();
    }
}
