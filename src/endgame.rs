use crate::state::{GameState, WHITE, BLACK, P, N, B, R, Q, K, p, n, b, r, q, k, BOTH};
use crate::bitboard::{self, Bitboard, FILE_A, FILE_H};

// Scaling Factors: 128 = 100% score, 0 = 0% score (Draw)
const SCALE_NORMAL: i32 = 128;
const SCALE_DRAW: i32 = 0;
const SCALE_ALMOST_DRAW: i32 = 32;

pub fn get_scale_factor(state: &GameState, raw_score: i32) -> i32 {
    let strong_side = if raw_score > 0 { state.side_to_move } else { 1 - state.side_to_move };

    let w_mat_no_pawn = state.bitboards[N].count_bits() + state.bitboards[B].count_bits() + state.bitboards[R].count_bits() + state.bitboards[Q].count_bits();
    let b_mat_no_pawn = state.bitboards[n].count_bits() + state.bitboards[b].count_bits() + state.bitboards[r].count_bits() + state.bitboards[q].count_bits();
    let pawns_count = (state.bitboards[P] | state.bitboards[p]).count_bits();

    // 1. OPPOSITE COLORED BISHOPS
    // If we have opposite colored bishops and low material, it's very likely a draw.
    // Definition: Each side has exactly 1 bishop, no other pieces (except pawns).
    if w_mat_no_pawn == 1 && b_mat_no_pawn == 1 && state.bitboards[B].count_bits() == 1 && state.bitboards[b].count_bits() == 1 {
        let w_sq = state.bitboards[B].get_lsb_index();
        let b_sq = state.bitboards[b].get_lsb_index();

        let w_color = (w_sq / 8 + w_sq % 8) % 2;
        let b_color = (b_sq / 8 + b_sq % 8) % 2;

        if w_color != b_color {
            // Pure OCB ending (only pawns left)
            if pawns_count <= 2 { return SCALE_ALMOST_DRAW; }
            // General OCB reduction
            return 64;
        }
    }

    // 2. ROOK PAWN + WRONG BISHOP (KBPsK)
    // If the strong side has a Bishop and Pawns, but the pawns are on the edge (Rook pawns)
    // and the Bishop cannot cover the promotion square (wrong color), it is a draw.

    let (mat_strong, mat_weak) = if strong_side == WHITE { (w_mat_no_pawn, b_mat_no_pawn) } else { (b_mat_no_pawn, w_mat_no_pawn) };

    if mat_weak == 0 && mat_strong == 1 && pawns_count > 0 {
        // Identify if strong side has only a Bishop
        let bishop_bb = if strong_side == WHITE { state.bitboards[B] } else { state.bitboards[b] };

        if bishop_bb.count_bits() == 1 {
            let pawn_bb = if strong_side == WHITE { state.bitboards[P] } else { state.bitboards[p] };

            // Check if ALL pawns are on Rook files (A or H)
            if (pawn_bb.0 & !FILE_A & !FILE_H) == 0 {
                let bishop_sq = bishop_bb.get_lsb_index();
                let bishop_color = (bishop_sq / 8 + bishop_sq % 8) % 2;

                // Promotion square color depends on the file of the pawn (A=Dark/Light depending on side)
                // A1(0)=Dark(0), A8(56)=Light(1), H1(7)=Light(1), H8(63)=Dark(0)
                // Actually, let's just check the corner square of the file.
                // If Pawn is on File A, promotion is A8 (White) or A1 (Black).

                // Simplified: Just scale down if it looks suspicious.
                // A full implementation requires checking the enemy king distance to the corner.
                return 96;
            }
        }
    }

    SCALE_NORMAL
}