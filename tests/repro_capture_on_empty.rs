#[cfg(test)]
mod tests {
    use aether::state::{GameState, Move};

    #[test]
    fn repro_capture_on_empty() {
        // Init global tables (needed for bitboard operations if used inside GameState)
        aether::bitboard::init_magic_tables();
        aether::zobrist::init_zobrist();

        // FEN from crash
        let fen = "1nb1k1nr/7P/7P/6p1/8/5N2/2pK1P2/qqB2B1R b - - 0 35";
        let mut board = GameState::parse_fen(fen);

        // Construct the move exactly as in logs
        // Move { source: 52, target: 60, promotion: None, is_capture: true }
        // 52 = e7, 60 = e8
        let mv = Move::new(52, 60, None, true);

        // Verify pre-conditions
        println!("FEN: {}", board.to_fen());
        println!("Move: {:?}", mv);
        println!("Source (52) Piece: {}", board.board[52]);
        println!("Target (60) Piece: {}", board.board[60]);

        // Attempt to apply move and expect panic or error
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
             board.make_move_inplace(mv, &mut None);
        }));

        assert!(result.is_err(), "Expected make_move to panic due to invalid capture on empty square");
    }
}
