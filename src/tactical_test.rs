// src/tactical_test.rs
#[cfg(test)]
mod tactical_tests {
    use crate::state::GameState;
    use crate::threat::{self, ThreatInfo};
    use crate::state::{WHITE, BLACK};
    use crate::state::N;

    #[test]
    fn test_dominance_bonus() {
        // Initialize tables (OnceLock handles multiple calls safely)
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables(); // Required for get_knight_attacks
        crate::threat::init_threat();

        // Position where White Knight can jump to e5 (dominant square)
        // e5 is controlled by d4 pawn.
        let fen = "rnbq1rk1/pp2bppp/5n2/2pp4/3P4/2N2N2/PPP1BPPP/R1BQ1RK1 w - - 0 1";
        let state = GameState::parse_fen(fen);

        let e5 = 36; // rank 4 (0-7), file 4 (e). 4*8+4 = 36.
        let knight_idx = N; // White Knight

        let score = threat::is_dominant_square(&state, e5 as u8, knight_idx, WHITE);

        // Should be positive (Rank 5, supported by d4 pawn)
        assert!(score > 0, "Knight on e5 should get dominance bonus. Score: {}", score);
    }

    #[test]
    fn test_threat_creation_delta() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables(); // Required for get_knight_attacks
        crate::threat::init_threat();
        crate::eval::init_eval(); // Required for PawnTable in evaluate_hce

        // White Bishop on c2, Queen on d1. Black King on h8, pawn h7.
        // Move: Qd1-d3 (threatens Qh7#). Quiet move.

        let fen = "r4r1k/pp4pp/2p5/8/8/8/PPB2PPP/R2Q2K1 w - - 0 1";

        let state = GameState::parse_fen(fen);

        // Move Qd1-d3 (d1=3, d3=19)
        let mv = crate::state::Move {
            source: 3,
            target: 19,
            promotion: None,
            is_capture: false,
        };

        let current_threat = threat::analyze(&state);
        let impact = threat::analyze_quiet_move_impact(&state, mv, &current_threat);

        println!("Threat Score: {}", impact.threat_score);
        assert!(impact.threat_score > 0, "Quiet move creating mate threat should have positive threat score");
        assert!(impact.is_tactical, "Should be classified as tactical quiet move");
    }
}
