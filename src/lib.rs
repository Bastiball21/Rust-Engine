#![allow(unused_imports, unused_variables, dead_code)]
pub mod bitboard;
pub mod book;
pub mod bullet_helper;
pub mod chess960; // Helper
pub mod datagen;
pub mod endgame;
pub mod eval;
pub mod logging;
pub mod movegen;
pub mod nnue;
pub mod nnue_scratch;
pub mod pawn;
pub mod perft;
pub mod search;
pub mod state;
pub mod tactical_test;
pub mod tests;
#[cfg(test)]
pub mod tests_chess960;
pub mod threat;
pub mod time;
pub mod tt;
pub mod tuning;
pub mod uci;
pub mod zobrist; // Tests
pub mod syzygy; // Added
pub mod parameters;
pub mod tuning_spsa;
pub mod history; // Added

use std::env;
use std::thread;

pub fn run_cli() {
    logging::init_logging();

    // 1. Initialize Global Tables
    zobrist::init_zobrist();
    bitboard::init_magic_tables();
    movegen::init_move_tables();
    eval::init_eval();
    threat::init_threat();

    // 2. Load NNUE (Implicitly loaded in uci_loop or init_nnue)

    // 3. Check for arguments (CLI Mode)
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        if args[1] == "tune" {
            tuning::run_tuning();
            return;
        }
        if args[1] == "tune-search" {
            tuning_spsa::run_tuning();
            return;
        }
        if args[1] == "test" {
            tests::run_mate_suite();
            return;
        }
        if args[1] == "perft" {
            perft::run_perft_suite();
            return;
        }
        if args[1] == "eval" {
            let state = state::GameState::parse_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            );
            println!("--- Debugging Eval ---");
            // threat::analyze is no longer needed for eval::evaluate, but can be useful for debug
            let threat = threat::analyze(&state);
            println!("Threat Info: {:?}", threat);

            // Create temporary accumulator for single eval
            // Create temporary accumulator for single eval
            let mut accumulators = [nnue::Accumulator::default(); 2];
            state.refresh_accumulator(&mut accumulators);
            let mut scratch = nnue_scratch::NNUEScratch::default();

            let score = eval::evaluate(&state, Some(&mut accumulators), Some(&mut scratch), -32000, 32000);
            println!("Final Score (CP): {}", score);
            if score.abs() > 1000 {
                println!("WARNING: Score is massive! This causes the 'Mate in 1' bug.");
            } else {
                println!("SUCCESS: Score is normal.");
            }
            return;
        }

        if args[1] == "convert" {
            if args.len() < 4 {
                println!("Usage: cargo run --release -- convert <pgn_file> <output_bin>");
                return;
            }
            let pgn_path = args[2].clone();
            let output_path = args[3].clone();
            datagen::convert_pgn(&pgn_path, &output_path);
            return;
        }

        if args[1] == "bench" {
             // Basic benchmark: Fixed depth search from startpos
             let state = state::GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
             let tt = std::sync::Arc::new(tt::TranspositionTable::new(16, 1)); // 16MB
             let mut data = search::SearchData::new();
             let params = parameters::SearchParameters::default();
             let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
             let mut stack = [search::StackEntry::default(); search::STACK_SIZE];

             println!("Running Benchmark: Startpos Depth 14");
             search::search(
                 &state,
                 search::Limits::FixedDepth(14),
                 &tt,
                 stop,
                 true,
                 &[],
                 &mut data,
                 &mut stack,
                 &params,
                 search::SearchMode::Play,
                 None,
                 Some(0)
             );
             return;
        }

        if args[1] == "datagen" {
            // Usage: "Aether datagen <games> <threads> <depth> <filename> <seed> [book_path] [book_ply]"
            // Index: 0      1        2        3          4        5          6       7           8

            let games = if args.len() > 2 {
                args[2].parse().unwrap_or(100)
            } else {
                100
            };

            let threads = if args.len() > 3 {
                args[3].parse().unwrap_or(1)
            } else {
                1
            };

            let _depth = if args.len() > 4 {
                args[4].parse().unwrap_or(8)
            } else {
                8
            };

            let filename = if args.len() > 5 {
                args[5].clone()
            } else {
                "aether_data.bin".to_string()
            };

            let seed = if args.len() > 6 {
                args[6].parse().unwrap_or_else(|_| {
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64
                })
            } else {
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64
            };

            let book_path = if args.len() > 7 {
                Some(args[7].clone())
            } else {
                None
            };

            let book_ply = if args.len() > 8 {
                args[8].parse().unwrap_or(0)
            } else {
                0
            };

            let config = datagen::DatagenConfig {
                num_games: games,
                num_threads: threads,
                filename,
                seed,
                book_path,
                book_ply,
            };

            // Ensure NNUE is loaded (File or Embedded)
            crate::nnue::ensure_nnue_loaded();
            datagen::run_datagen(config);
            return;
        }
    }

    // 4. Normal Mode: Launch UCI
    let builder = thread::Builder::new()
        .name("uci_thread".into())
        .stack_size(32 * 1024 * 1024);

    let handler = builder
        .spawn(|| {
            uci::uci_loop();
        })
        .unwrap();

    handler.join().unwrap();
}
