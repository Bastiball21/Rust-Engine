#[cfg(test)]
mod tests {
    use crate::movegen::{self, MoveGenerator};
    use crate::state::{GameState, Move, B, K, N, P, Q, R};

    #[test]
    fn test_chess960_castling_move_generation() {
        crate::zobrist::init_zobrist(); // Needed for hash computation in parse_fen
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        // Position: R K R (Chess960 style) on first rank
        // a1 b1 c1 d1 e1 f1 g1 h1
        // B  B  R  K  R  .  .  Q
        // c1 is R(2), d1 is K(3), e1 is R(4). f1(5), g1(6) empty.
        // FEN: bbrkr2q/pppppppp/8/8/8/8/PPPPPPPP/BBRKR2Q w KQkq - 0 1

        let state = GameState::parse_fen("bbrkr2q/pppppppp/8/8/8/8/PPPPPPPP/BBRKR2Q w KQkq - 0 1");

        let mut generator = MoveGenerator::new();
        generator.generate_moves(&state);

        // Internal Move: King(3) -> Rook(2) (Queenside)
        // Internal Move: King(3) -> Rook(4) (Kingside)

        let mut found_qs = false;
        let mut found_ks = false;

        for i in 0..generator.list.count {
            let mv = generator.list.moves[i];
            if mv.source() == 3 {
                if mv.target() == 2 { found_qs = true; }
                if mv.target() == 4 { found_ks = true; }
            }
        }

        assert!(found_qs, "960 Queenside Castle missing (K@d1 -> R@c1)");
        assert!(found_ks, "960 Kingside Castle missing (K@d1 -> R@e1)");
    }

    #[test]
    fn test_chess960_castling_execution() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();

        for i in 0..10 {
            let state = crate::chess960::generate_chess960_position(i);

            let wk_sq = state.bitboards[K].get_lsb_index();
            let wk_file = (wk_sq % 8) as u8;

            let ks_rook_file = state.castling_rook_files[crate::state::WHITE][0];
            let qs_rook_file = state.castling_rook_files[crate::state::WHITE][1];

            // King-side move: King captures Rook
            let ks_target = ks_rook_file;
            let ks_move = Move::new(wk_file, ks_target, None, true);

            let mut state_ks = state.clone();
            let mut acc = None;
            state_ks.make_move_inplace(ks_move, &mut acc);

            // Verify positions
            // KS Castling -> King to G1 (6), Rook to F1 (5)
            assert_eq!(
                state_ks.board[6] as usize,
                K,
                "King should be on G1 after KS castling. Index: {}",
                i
            );
            assert_eq!(
                state_ks.board[5] as usize,
                R,
                "Rook should be on F1 after KS castling. Index: {}",
                i
            );

            // Queen-side move
            let qs_target = qs_rook_file;
            let qs_move = Move::new(wk_file, qs_target, None, true);

            let mut state_qs = state.clone();
            state_qs.make_move_inplace(qs_move, &mut acc);

            // QS Castling -> King to C1 (2), Rook to D1 (3)
            assert_eq!(
                state_qs.board[2] as usize,
                K,
                "King should be on C1 after QS castling. Index: {}",
                i
            );
            assert_eq!(
                state_qs.board[3] as usize,
                R,
                "Rook should be on D1 after QS castling. Index: {}",
                i
            );
        }
    }
}
