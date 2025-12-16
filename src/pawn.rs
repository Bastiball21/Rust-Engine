// src/pawn.rs
use crate::bitboard::{self, Bitboard, FILE_A, FILE_H};
use crate::state::{p, GameState, BLACK, P, WHITE};

#[derive(Clone, Copy, Default)]
pub struct PawnEntry {
    pub key: u64,
    pub score_mg: i32,
    pub score_eg: i32,
    pub passed_pawns: [Bitboard; 2],
    pub pawn_attacks: [Bitboard; 2],
    pub king_safety_scores: [i32; 2], // Cached shelter/storm scores
}

pub fn evaluate_pawns(state: &GameState) -> PawnEntry {
    let mut entry = PawnEntry {
        key: state.hash,
        ..Default::default()
    };
    let w_pawns = state.bitboards[P];
    let b_pawns = state.bitboards[p];

    entry.pawn_attacks[WHITE] = bitboard::pawn_attacks(w_pawns, WHITE);
    entry.pawn_attacks[BLACK] = bitboard::pawn_attacks(b_pawns, BLACK);

    // --- WHITE PAWNS ---
    let mut bb = w_pawns;
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize;
        bb.pop_bit(sq as u8);
        let rank = sq / 8;

        // Connected
        if entry.pawn_attacks[WHITE].get_bit(sq as u8) {
            entry.score_mg += 10;
            entry.score_eg += 15;
        }

        // Isolated
        let file_mask = bitboard::file_mask(sq);
        let adj_mask = ((file_mask.0 << 1) & !FILE_A) | ((file_mask.0 >> 1) & !FILE_H);
        if (w_pawns.0 & adj_mask) == 0 {
            entry.score_mg -= 15;
            entry.score_eg -= 20;
        }

        // Doubled
        if (w_pawns.0 & file_mask.0).count_ones() > 1 {
            entry.score_mg -= 10;
            entry.score_eg -= 15;
        }

        // Passed
        let passed_mask = bitboard::passed_pawn_mask(WHITE, sq);
        if (passed_mask.0 & b_pawns.0) == 0 {
            entry.passed_pawns[WHITE].set_bit(sq as u8);
            let bonus = [0, 10, 20, 40, 70, 120, 200, 0];
            entry.score_mg += bonus[rank] / 2;
            entry.score_eg += bonus[rank];
        }
    }

    // --- BLACK PAWNS ---
    let mut bb = b_pawns;
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize;
        bb.pop_bit(sq as u8);
        let rank = sq / 8;
        let rel_rank = 7 - rank;

        if entry.pawn_attacks[BLACK].get_bit(sq as u8) {
            entry.score_mg -= 10;
            entry.score_eg -= 15;
        }

        let file_mask = bitboard::file_mask(sq);
        let adj_mask = ((file_mask.0 << 1) & !FILE_A) | ((file_mask.0 >> 1) & !FILE_H);
        if (b_pawns.0 & adj_mask) == 0 {
            entry.score_mg += 15;
            entry.score_eg += 20;
        }

        if (b_pawns.0 & file_mask.0).count_ones() > 1 {
            entry.score_mg += 10;
            entry.score_eg += 15;
        }

        let passed_mask = bitboard::passed_pawn_mask(BLACK, sq);
        if (passed_mask.0 & w_pawns.0) == 0 {
            entry.passed_pawns[BLACK].set_bit(sq as u8);
            let bonus = [0, 10, 20, 40, 70, 120, 200, 0];
            entry.score_mg -= bonus[rel_rank] / 2;
            entry.score_eg -= bonus[rel_rank];
        }
    }

    // Compute King Safety / Shelter data here if desired,
    // or just leave it for the main eval to use the pawn bitboards.

    entry
}
