use crate::movegen::{is_square_attacked, MoveGenerator};
use crate::state::{k, GameState, K, WHITE};
use std::time::Instant;

pub fn run_perft_suite() {
    println!("--- Aether Perft Suite ---");

    let positions = [
        (
            "Start Position",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            [1, 20, 400, 8902, 197281, 4865609], // Depths 0-5
        ),
        (
            "Position 2 (Kiwipete)",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            [1, 48, 2039, 97862, 4085603, 193690690],
        ),
        (
            "Position 3",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            [1, 14, 191, 2812, 43238, 674624],
        ),
        (
            "Position 4 (Passer/En Passant)",
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            [1, 6, 264, 9467, 422333, 15833292],
        ),
        (
            "Position 5 (Mirrored Mate)",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            [1, 44, 1486, 62379, 2103487, 89941194],
        ),
        (
            "Castling Check (Custom)",
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            [1, 26, 0, 0, 0, 0], // Depth 1: 26 moves (5K+14R+7R = 26)
        ),
    ];

    let mut total_nodes = 0;
    let mut total_time = 0;

    for (name, fen, expected) in positions.iter() {
        println!("\nTesting: {}", name);
        let mut state = GameState::parse_fen(fen);

        // Run Depth 5 (or 4 if 5 is too slow for quick tests)
        let depth = 4.min(expected.len() - 1); // Limit to depth 4 for speed in suite
        if expected.len() <= depth {
            continue;
        }

        let start = Instant::now();
        let nodes = perft(&mut state, depth as u8);
        let elapsed = start.elapsed().as_millis();

        total_nodes += nodes;
        total_time += elapsed;

        println!("Depth {}: Nodes: {} Time: {}ms", depth, nodes, elapsed);

        if nodes == expected[depth] {
            println!("RESULT: PASS");
        } else {
            println!("RESULT: FAIL (Expected {})", expected[depth]);
            // If failed, print divide for debug
            perft_divide(&mut state, depth as u8);
            // return; // Continue to test others
        }
    }

    println!("\n--- SUITE COMPLETE ---");
    println!("Total Nodes: {}", total_nodes);
    println!("Total Time:  {}ms", total_time);
    if total_time > 0 {
        println!(
            "NPS:         {}",
            (total_nodes as u128 * 1000) / total_time
        );
    }
}

// Recursive perft function
pub fn perft(state: &GameState, depth: u8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut nodes = 0;
    let mut generator = MoveGenerator::new();
    generator.generate_moves(state);

    for i in 0..generator.list.count {
        let mv = generator.list.moves[i];

        // Use your engine's existing "make-then-check" logic
        let next_state = state.make_move(mv);

        // Verify Legality
        let our_king = if state.side_to_move == WHITE { K } else { k };
        let king_sq = next_state.bitboards[our_king].get_lsb_index() as u8;

        if !is_square_attacked(&next_state, king_sq, next_state.side_to_move) {
            nodes += perft(&next_state, depth - 1);
        }
    }
    nodes
}

// Debugging tool: Prints move counts for the first ply
pub fn perft_divide(state: &GameState, depth: u8) {
    println!("--- Perft Divide Depth {} ---", depth);
    let mut generator = MoveGenerator::new();
    generator.generate_moves(state);

    let mut total = 0;

    for i in 0..generator.list.count {
        let mv = generator.list.moves[i];
        let next_state = state.make_move(mv);

        let our_king = if state.side_to_move == WHITE { K } else { k };
        let king_sq = next_state.bitboards[our_king].get_lsb_index() as u8;

        if !is_square_attacked(&next_state, king_sq, next_state.side_to_move) {
            let count = perft(&next_state, depth - 1);
            println!(
                "{}{}: {}",
                crate::search::square_to_coord(mv.source()),
                crate::search::square_to_coord(mv.target()),
                count
            );
            total += count;
        }
    }
    println!("Total: {}", total);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::GameState;

    #[test]
    fn test_perft_start_pos() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();
        let state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(perft(&state, 1), 20);
        assert_eq!(perft(&state, 2), 400);
        assert_eq!(perft(&state, 3), 8902);
        // assert_eq!(perft(&state, 4), 197281); // Takes ~0.5s
    }

    #[test]
    fn test_perft_kiwipete() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();
        let state = GameState::parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        assert_eq!(perft(&state, 1), 48);
        assert_eq!(perft(&state, 2), 2039);
        assert_eq!(perft(&state, 3), 97862);
    }

    #[test]
    fn test_perft_castling() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();
        let state = GameState::parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_eq!(perft(&state, 1), 26);
    }

    #[test]
    fn test_make_unmake_symmetry() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();
        let state = GameState::parse_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");

        let mut generator = MoveGenerator::new();
        generator.generate_moves(&state);

        let original_hash = state.hash;

        // We verify that unmake restores the state perfectly
        for i in 0..generator.list.count {
            let mv = generator.list.moves[i];

            let mut test_state = state; // Copy
            // In tests we can pass None for accumulator
            let unmake_info = test_state.make_move_inplace(mv, &mut None);
            test_state.unmake_move(mv, unmake_info, &mut None);

            assert_eq!(test_state.hash, original_hash, "Hash mismatch after unmake move {:?}", mv);
            assert_eq!(test_state.bitboards[WHITE].0, state.bitboards[WHITE].0);
            assert_eq!(test_state.en_passant, state.en_passant);
            assert_eq!(test_state.castling_rights, state.castling_rights);
        }
    }
}
