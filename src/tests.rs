use crate::state::GameState;
use crate::search;
use crate::tt::TranspositionTable;
use crate::time::{TimeManager, TimeControl};
use std::sync::{Arc, atomic::AtomicBool};

pub fn run_mate_suite() {
    println!("--- Aether Mate Verification Suite ---");

    let positions = [
        // 1. Scholar's Mate (Mate in 1)
        ("Mate in 1", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 1),

        // 2. Lasker vs Thomas (Mate in 7)
        // Position after 11. Qxh7+ Kxh7. White to move and mate.
        ("Lasker-Thomas (M7)", "r3r3/pbppqppk/1pn1pb2/8/3PN3/2PB1N2/PP3PPP/R3K2R w KQ - 0 1", 7),

        // 3. Quiet Mate in 3 (Tests Q-Search pruning)
        ("Quiet Mate in 3", "r5rk/5p1p/5R2/4B3/8/8/7P/7K w - - 0 1", 3),

        // 4. Evasion Test (Should be Draw/0.00, NOT Mate)
        ("Evasion Logic", "8/8/8/8/8/3k4/3p4/3K4 w - - 2 2", 0)
    ];

    let mut tt = TranspositionTable::new(16);
    let stop = Arc::new(AtomicBool::new(false));

    for (name, fen, expected_ply) in positions.iter() {
        print!("Testing: {:<20} | ", name);

        let state = GameState::parse_fen(fen);

        // FIX: Use Infinite time. This prevents the engine from stopping early (e.g. at Depth 12)
        // when it thinks it's losing material, forcing it to find the deep mate at Depth 14+.
        let tm = TimeManager::new(TimeControl::Infinite, state.side_to_move, 0);

        // FIX: Search depth calculation.
        // Lasker (M7) requires ~14 ply. Old depth was too shallow.
        // We now use (Mate * 2) + 2 safety buffer.
        let search_depth = if *expected_ply == 0 { 6 } else { (expected_ply * 2 + 2) as u8 };

        search::search(&state, tm, &tt, stop.clone(), search_depth, true, vec![]);

        println!("(Expected: mate {})", expected_ply);
        tt.clear();
    }
}