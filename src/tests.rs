#[cfg(test)]
pub mod tests {
    use crate::movegen::{self, MoveGenerator};
    use crate::state::{GameState, Move, B, K, N, P, Q, R, NO_PIECE};

    // Helper to find a move in the list
    fn find_move(state: &GameState, from: u8, to: u8, promotion: Option<usize>) -> Option<Move> {
        let mut generator = MoveGenerator::new();
        generator.generate_moves(state);
        for i in 0..generator.list.count {
            let m = generator.list.moves[i];
            if m.source() == from && m.target() == to && m.promotion() == promotion {
                return Some(m);
            }
        }
        None
    }

    #[test]
    fn test_move_compression() {
        // Test packing and unpacking of moves
        let source = 10;
        let target = 20;
        let promo = Some(Q);
        let capture = true;

        let mv = Move::new(source, target, promo, capture);

        assert_eq!(mv.source(), source);
        assert_eq!(mv.target(), target);
        assert_eq!(mv.promotion(), promo);
        assert_eq!(mv.is_capture(), capture);

        let null_mv = Move::default();
        assert!(null_mv.is_null());
        assert_eq!(null_mv.source(), 0);
        assert_eq!(null_mv.target(), 0);
    }

    #[test]
    fn test_simple_position() {
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        let state = GameState::parse_fen("8/8/8/8/8/8/4P3/4K3 w - - 0 1");
        let mut generator = MoveGenerator::new();
        generator.generate_moves(&state);

        // e2e3, e2e4, e1d1, e1f1, e1e2? No e1e2 (self capture)
        // e1f2, e1d2
        // e2 is 12. e3=20, e4=28.
        // e1 is 4. d1=3, f1=5, d2=11, e2=12 (blocked), f2=13.

        let mut found_e2e4 = false;
        for i in 0..generator.list.count {
            let m = generator.list.moves[i];
            if m.source() == 12 && m.target() == 28 {
                found_e2e4 = true;
            }
        }
        assert!(found_e2e4);
    }

    #[test]
    fn test_castling_chess960_off() {
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        // Standard position
        let mut state = GameState::parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        // e1 (4) -> g1 (6) ?? No, internal is e1 -> h1 (7)

        let mut generator = MoveGenerator::new();
        generator.generate_moves(&state);

        let mut has_kingside = false;
        let mut has_queenside = false;

        for i in 0..generator.list.count {
            let m = generator.list.moves[i];
            if m.source() == 4 {
                if m.target() == 7 { has_kingside = true; }
                if m.target() == 0 { has_queenside = true; }
            }
        }

        assert!(has_kingside, "White O-O missing");
        assert!(has_queenside, "White O-O-O missing");
    }

    #[test]
    fn test_capture_desync_recovery() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        // Position: White Rook at A1, Black Pawn at A2. Capture A1-A2.
        // FEN: 8/8/8/8/8/8/p7/R3K3 w Q - 0 1
        let mut state = GameState::parse_fen("8/8/8/8/8/8/p7/R3K3 w Q - 0 1");

        // Squares: A1=0, A2=8.
        // Confirm setup
        assert!(state.bitboards[R].get_bit(0));
        assert!(state.bitboards[crate::state::p].get_bit(8));
        assert_eq!(state.board[0], R as u8);
        assert_eq!(state.board[8], crate::state::p as u8);

        // Construct the capture move: A1xA2
        let mv = Move::new(0, 8, None, true);

        // Manually desynchronize board array at target
        state.board[8] = NO_PIECE as u8;

        // Perform the move. This should trigger the recovery logic (and warning).
        // It should NOT panic.
        let info = state.make_move_inplace(mv);

        // Verify recovery was successful
        // The captured piece should be correctly identified as 'p'
        assert_eq!(info.captured, crate::state::p as u8);

        // Verify the board state after move
        // A2 should now contain White Rook
        assert_eq!(state.board[8], R as u8);
        assert!(state.bitboards[R].get_bit(8));
        // A1 should be empty
        assert_eq!(state.board[0], NO_PIECE as u8);
        assert!(!state.bitboards[R].get_bit(0));
        // Black Pawn should be gone from bitboards
        assert!(!state.bitboards[crate::state::p].get_bit(8));
    }
}

pub fn run_mate_suite() {
    // Placeholder for legacy test logic or mate suite.
    println!("Mate suite disabled.");
}
