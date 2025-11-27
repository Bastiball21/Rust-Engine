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
mod tuning; // Register the tuning module

use std::thread;
use std::env;

fn main() {
    // 1. Check for "tune" argument
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "tune" {
        tuning::run_tuning();
        return; // Exit after tuning
    }

    // 2. Normal Mode: Launch UCI
    let builder = thread::Builder::new()
        .name("uci_thread".into())
        .stack_size(32 * 1024 * 1024); 

    let handler = builder.spawn(|| {
        uci::uci_loop();
    }).unwrap();

    handler.join().unwrap();
}