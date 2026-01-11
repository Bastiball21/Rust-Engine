#[cfg(test)]
mod tests {
    use aether::state::{GameState, Move};
    use aether::zobrist;
    use aether::bitboard;
    use aether::movegen;

    #[test]
    fn test_tt_collision_protection() {
        // Initialize globals
        zobrist::init_zobrist();
        bitboard::init_magic_tables();
        movegen::init_move_tables();

        // 1. Setup a state
        let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        state.compute_hash();

        // 2. Create a move that is Pseudo-Legal but Inconsistent (Capture on Empty Target)
        // Source: e2 (12) -> Target e4 (28).
        // e2 is Pawn. e4 is Empty.
        // Valid pawn push? Yes.
        // But if we mark it as CAPTURE:
        let bad_mv = Move::new(12, 28, None, true); // Captured flag set!

        let is_pseudo = movegen::is_move_pseudo_legal(&state, bad_mv);
        // We expect pseudo_legal to be true if it only checks geometry?
        // Or false if it checks target occupancy for capture?
        // movegen usually allows "Capture" if target is occupied OR EP.
        // Here target e4 is empty. EP is not e4.
        // So pseudo_legal SHOULD return FALSE.

        let is_consistent = state.is_move_consistent(bad_mv);
        // Consistent should return FALSE because target is empty for capture.

        println!("Bad Move 1 (Capture on Empty): {:?}, Pseudo: {}, Consistent: {}", bad_mv, is_pseudo, is_consistent);

        assert!(!is_pseudo || !is_consistent, "Capture on empty should be rejected");

        // 3. Create a move where Source is Empty (Desync simulation)
        // Move from a3 (16) -> a4 (24).
        // a3 is Empty.
        let bad_mv_2 = Move::new(16, 24, None, false);

        let is_pseudo_2 = movegen::is_move_pseudo_legal(&state, bad_mv_2);
        // movegen checks if source piece exists. If empty, should be false.

        let is_consistent_2 = state.is_move_consistent(bad_mv_2);
        // Consistent checks if board[source] == NO_PIECE. Should be false.

        println!("Bad Move 2 (Source Empty): {:?}, Pseudo: {}, Consistent: {}", bad_mv_2, is_pseudo_2, is_consistent_2);

        assert!(!is_pseudo_2 || !is_consistent_2, "Move from empty source should be rejected");

        // 4. Case: Source Occupied (BB) but Empty (Mailbox). Desync.
        // We cannot simulate this easily on `state` because we can't hack internals easily from integration test
        // without `unsafe` or if fields are public. `board` is public.

        // Hacking state:
        let mut desync_state = state.clone();
        // Clear e2 in mailbox, keep in bitboard
        desync_state.board[12] = 12; // NO_PIECE

        let mv_desync = Move::new(12, 28, None, false);

        let is_pseudo_3 = movegen::is_move_pseudo_legal(&desync_state, mv_desync);
        // movegen uses BITBOARDS for piece existence?
        // If it uses bitboards, it sees piece.
        // If it uses mailbox, it sees empty.

        let is_consistent_3 = desync_state.is_move_consistent(mv_desync);
        // Consistent uses MAILBOX. So it sees empty. Returns FALSE.

        println!("Bad Move 3 (Desync Source): {:?}, Pseudo: {}, Consistent: {}", mv_desync, is_pseudo_3, is_consistent_3);

        // If pseudo is TRUE (bitboard) and consistent is FALSE (mailbox).
        // Then `is_pseudo && is_consistent` is FALSE. Safe.
        // If we relied ONLY on `is_pseudo`, we would crash!

        if is_pseudo_3 && !is_consistent_3 {
            println!("VERIFIED: is_pseudo_legal returns true on desync, but is_move_consistent saves us!");
        } else {
             println!("Note: is_pseudo_legal returned {} on desync.", is_pseudo_3);
        }

        assert!(!(is_pseudo_3 && is_consistent_3), "Should fail combined check");
    }
}
