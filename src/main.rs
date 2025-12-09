// src/main.rs
#![allow(unused_imports, unused_variables, dead_code)]
mod bitboard;
mod state;
mod movegen;
mod search;
mod eval;
mod zobrist;
mod tt;
mod uci;
mod time;
mod tuning;
mod pawn;
mod endgame;
mod perft;
mod tests;
mod nnue;
mod datagen;

use std::thread;
use std::env;

fn main() {
    // 1. Initialize Global Tables
    zobrist::init_zobrist();
    bitboard::init_magic_tables();
    movegen::init_move_tables();
    eval::init_eval();

    // 2. Load NNUE
    nnue::init_nnue("nn-aether.nnue");

    // 3. Check for arguments (CLI Mode)
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 {
        if args[1] == "tune" {
            tuning::run_tuning();
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
            let state = state::GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            println!("--- Debugging Eval ---");
            let score = eval::evaluate(&state);
            println!("Final Score (CP): {}", score);
            if score.abs() > 1000 {
                println!("WARNING: Score is massive! This causes the 'Mate in 1' bug.");
            } else {
                println!("SUCCESS: Score is normal.");
            }
            return;
        }
        // --- FIX: Moved inside the main function logic ---
        if args[1] == "datagen" {
            // Usage: "Aether datagen 1000"
            let count = if args.len() > 2 { args[2].parse().unwrap_or(100) } else { 100 };
            datagen::run_datagen(count);
            return;
        }
    }

    // 4. Normal Mode: Launch UCI
    let builder = thread::Builder::new()
        .name("uci_thread".into())
        .stack_size(32 * 1024 * 1024); 

    let handler = builder.spawn(|| {
        uci::uci_loop();
    }).unwrap();

    handler.join().unwrap();
}
