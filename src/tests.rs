use crate::movegen;
use crate::state::{GameState, BOTH, NO_PIECE, P, R, WHITE};

pub fn run_mate_suite() {
    println!("Running mate suite...");
    // Implementation of mate suite runner
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use crate::state::{GameState, BOTH, NO_PIECE, P, R, WHITE};

    #[test]
    fn test_is_consistent_valid() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        let state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        assert!(state.is_consistent(), "Start position should be consistent");
    }

    #[test]
    fn test_is_consistent_desync_occupancies() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        // Corrupt Occupancy (remove e2 pawn bit from BOTH)
        let e2 = 12; // rank 1, file 4 = 1*8 + 4 = 12
        state.occupancies[BOTH].pop_bit(e2);

        // Bitboard[P] still has bit at e2, but Occupancies[BOTH] does not.
        assert!(
            !state.is_consistent(),
            "State should be inconsistent when Occupancy lacks a bit present in Bitboards"
        );
    }

    #[test]
    fn test_is_consistent_desync_board_vs_bitboard() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        // Corrupt Bitboard (remove e2 pawn bit from P)
        let e2 = 12;
        state.bitboards[P].pop_bit(e2);

        // Board[e2] is P (0), but Bitboard[P] is missing the bit.
        assert!(
            !state.is_consistent(),
            "State should be inconsistent when Board has piece but Bitboard is missing bit"
        );
    }

    #[test]
    fn test_is_consistent_ghost_piece() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        // This simulates the "Capture on Empty" panic scenario
        // Bitboard says piece exists, Board says NO_PIECE
        let mut state = GameState::parse_fen("8/8/8/8/8/8/8/8 w - - 0 1");

        // Add a "Ghost" Rook at a1
        state.bitboards[R].set_bit(0);
        state.occupancies[WHITE].set_bit(0);
        state.occupancies[BOTH].set_bit(0);

        // Ensure Board says NO_PIECE
        state.board[0] = NO_PIECE as u8;

        assert!(
            !state.is_consistent(),
            "State should be inconsistent when Bitboard has piece but Board is empty"
        );
    }
}
