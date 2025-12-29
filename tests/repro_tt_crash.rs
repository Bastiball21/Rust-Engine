
#[cfg(test)]
mod tests {
    use aether::state::{GameState, Move};
    use aether::tt::TranspositionTable;
    use aether::movegen;
    use aether::zobrist;
    use aether::bitboard;

    #[test]
    fn test_tt_crash_repro() {
        // Initialize globals required for State/MoveGen
        zobrist::init_zobrist();
        bitboard::init_magic_tables();
        // movegen tables are initialized via bitboard::init_magic_tables in this engine

        // FEN from the crash log
        // r1bq1b1r/2npk1p1/3p2p1/p4P2/4n3/P3P2p/7P/PRB1KBNR w K - 0 22
        let fen = "r1bq1b1r/2npk1p1/3p2p1/p4P2/4n3/P3P2p/7P/PRB1KBNR w K - 0 22";
        let state = GameState::parse_fen(fen);

        // Crash Move: Move { source: 57, target: 0, promotion: None, is_capture: true }
        // Source 57 is b8. Target 0 is a1.
        // In this FEN, b8 (57) is EMPTY.
        let mv = Move::new(57, 0, None, true);

        println!("Testing Move: {:?}", mv);
        println!("FEN: {}", fen);

        // 1. Verify state setup
        assert_eq!(state.board[57], 12, "Square 57 should be NO_PIECE (12)");

        // 2. Check movegen::is_move_pseudo_legal
        // This is what tt.is_pseudo_legal calls.
        let is_pseudo = movegen::is_move_pseudo_legal(&state, mv);
        println!("is_move_pseudo_legal: {}", is_pseudo);
        assert!(!is_pseudo, "is_move_pseudo_legal should return FALSE for move from empty square");

        // 3. Check TT wrapper
        let tt = TranspositionTable::new(1, 1);
        let tt_valid = tt.is_pseudo_legal(&state, mv);
        assert!(!tt_valid, "tt.is_pseudo_legal should return FALSE");

        // 4. Check state.is_move_consistent
        // This is the secondary guard in datagen.
        let is_consistent = state.is_move_consistent(mv);
        println!("is_move_consistent: {}", is_consistent);
        assert!(!is_consistent, "state.is_move_consistent should return FALSE");

        // 5. Attempt make_move_inplace (Should panic or be protected by assertions)
        // We use std::panic::catch_unwind to verify it panics as expected if we bypass checks.

        let result = std::panic::catch_unwind(|| {
            let mut s = state.clone();
            let _ = s.make_move(mv);
        });

        assert!(result.is_err(), "make_move SHOULD panic for this illegal move");
    }
}
