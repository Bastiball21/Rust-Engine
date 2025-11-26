#![allow(unused_imports, unused_variables, dead_code)]
mod bitboard;
mod state;
mod movegen;
mod search;
mod eval;
mod zobrist;
mod tt;
mod uci;
mod time; // Added this

use std::thread;

fn main() {
    // Standard Stack Size Increase
    let builder = thread::Builder::new()
        .name("uci_thread".into())
        .stack_size(32 * 1024 * 1024); 

    let handler = builder.spawn(|| {
        uci::uci_loop();
    }).unwrap();

    handler.join().unwrap();
}
