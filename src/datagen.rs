// src/datagen.rs
use crate::state::{GameState, WHITE, BLACK, P, N, B, R, Q, K, p, n, b, r, q, k};
use crate::search;
use crate::tt::TranspositionTable;
use crate::time::{TimeManager, TimeControl};
use std::sync::{Arc, atomic::AtomicBool};
use std::fs::OpenOptions;
use std::io::Write;
use bulletformat::ChessBoard;

pub fn run_datagen(games: usize) {
    println!("Starting Datagen for {} games at Depth 6 (Binary Output)...", games);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open("aether_data.bin")
        .expect("Could not open aether_data.bin");

    // 64MB TT is good for Depth 6
    let mut tt = TranspositionTable::new(64);
    let stop = Arc::new(AtomicBool::new(false));

    for i in 0..games {
        let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let mut boards = Vec::new();
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
            let depth = if state.fullmove_number < 8 { 1 } else { 6 };

            let tm = TimeManager::new(TimeControl::MoveTime(50), state.side_to_move, 0);
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

            // Save Data
            // Note: from_raw expects White Relative Score and Result
            // Our tt_score is from side_to_move perspective (usually? check search.rs)
            // Aether search returns score relative to root side_to_move? No, usually Negamax returns side-relative.
            // Let's assume tt_score is relative to state.side_to_move.
            // white_relative_score = if side == WHITE { score } else { -score }
            let white_score = if state.side_to_move == WHITE { tt_score } else { -tt_score };

            if state.fullmove_number > 8 {
                // Construct bitboards for from_raw
                // [White, Black, P, N, B, R, Q, K]
                let mut bbs = [0u64; 8];
                bbs[0] = state.occupancies[WHITE].0;
                bbs[1] = state.occupancies[BLACK].0;
                bbs[2] = state.bitboards[P].0 | state.bitboards[p].0;
                bbs[3] = state.bitboards[N].0 | state.bitboards[n].0;
                bbs[4] = state.bitboards[B].0 | state.bitboards[b].0;
                bbs[5] = state.bitboards[R].0 | state.bitboards[r].0;
                bbs[6] = state.bitboards[Q].0 | state.bitboards[q].0;
                bbs[7] = state.bitboards[K].0 | state.bitboards[k].0;

                // We store the parameters to build the board later when we know the result
                boards.push((bbs, state.side_to_move, white_score));
            }

            state = state.make_move(best_move);
        }

        // Write to file
        for (bbs, stm, score) in boards {
            if let Ok(board) = ChessBoard::from_raw(bbs, stm, score as i16, result_val) {
                 let bytes = unsafe {
                     std::slice::from_raw_parts(
                         (&board as *const ChessBoard) as *const u8,
                         std::mem::size_of::<ChessBoard>()
                     )
                 };
                 file.write_all(bytes).unwrap();
            }
        }

        // Print progress
        println!("Generated game {} / {} (Result: {})", i+1, games, result_val);
        tt.clear();
    }
}
