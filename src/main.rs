// src/main.rs
#![allow(unused_imports, unused_variables, dead_code)]
mod bitboard;
mod book;
mod bullet_helper;
mod chess960; // Helper
mod datagen;
mod endgame;
mod eval;
mod logging;
mod movegen;
mod nnue;
mod pawn;
mod perft;
mod search;
mod state;
mod tactical_test;
mod tests;
#[cfg(test)]
mod tests_chess960;
mod threat;
mod time;
mod tt;
mod tuning;
mod uci;
mod zobrist; // Tests
mod syzygy; // Added
mod parameters;
mod tuning_spsa;

use std::env;
use std::thread;

fn main() {
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
            let score = eval::evaluate(&state, -32000, 32000);
            println!("Final Score (CP): {}", score);
            if score.abs() > 1000 {
                println!("WARNING: Score is massive! This causes the 'Mate in 1' bug.");
            } else {
                println!("SUCCESS: Score is normal.");
            }
            return;
        }

        if args[1] == "datagen" {
            // Usage: "Aether datagen <games> <threads> <depth> <filename> [book_path] [book_ply]"
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

            let depth = if args.len() > 4 {
                args[4].parse().unwrap_or(6)
            } else {
                6
            };

            let filename = if args.len() > 5 {
                args[5].clone()
            } else {
                "aether_data.bin".to_string()
            };

            // Optional Book
            let book_path = if args.len() > 6 {
                let s = args[6].clone();
                if s == "none" || s == "-" {
                    None
                } else {
                    Some(s)
                }
            } else {
                None
            };

            let book_ply = if args.len() > 7 {
                args[7].parse().unwrap_or(16)
            } else {
                16
            };

            let seed = if args.len() > 8 {
                args[8].parse().unwrap_or_else(|_| {
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

            let config = datagen::DatagenConfig {
                num_games: games,
                num_threads: threads,
                depth,
                filename,
                book_path,
                book_ply,
                seed,
            };

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
