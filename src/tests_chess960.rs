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
}
