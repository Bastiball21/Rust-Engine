// src/eval.rs
use crate::bitboard::{self, Bitboard};
use crate::state::{b, k, n, p, q, r, GameState, B, BLACK, BOTH, K, N, P, Q, R, WHITE};
#[cfg(feature = "tuning")]
use std::sync::atomic::{AtomicI32, Ordering};

// --- CONDITIONAL TYPE ALIASING ---
#[cfg(feature = "tuning")]
type EvalValue = AtomicI32;
#[cfg(not(feature = "tuning"))]
type EvalValue = i32;

// --- ACCESS HELPER ---
#[cfg(feature = "tuning")]
trait Access {
    fn val(&self) -> i32;
}

#[cfg(feature = "tuning")]
impl Access for AtomicI32 {
    fn val(&self) -> i32 {
        self.load(Ordering::Relaxed)
    }
}

#[cfg(not(feature = "tuning"))]
trait Access {
    fn val(&self) -> i32;
}

#[cfg(not(feature = "tuning"))]
impl Access for i32 {
    #[inline(always)]
    fn val(&self) -> i32 {
        *self
    }
}

// --- MACROS ---
#[cfg(feature = "tuning")]
macro_rules! a { ($($x:expr),*) => { [ $(AtomicI32::new($x)),* ] } }

#[cfg(not(feature = "tuning"))]
macro_rules! a { ($($x:expr),*) => { [ $($x),* ] } }

// --- CONFIGURATION ---
// Material
#[rustfmt::skip] pub static MG_VALS: [EvalValue; 6] = a![ 82, 337, 365, 477, 1025, 0 ];
#[rustfmt::skip] pub static EG_VALS: [EvalValue; 6] = a![ 94, 281, 297, 512,  936, 0 ];

// Phase Weights
pub const PHASE_WEIGHTS: [i32; 6] = [0, 1, 1, 2, 4, 0];
// pub const TOTAL_PHASE: i32 = 24;

// King Safety Weights
pub const KING_TROPISM_PENALTY: [i32; 8] = [10, 8, 5, 2, 0, 0, 0, 0];
pub const SHIELD_MISSING_PENALTY: i32 = -20;
pub const SHIELD_OPEN_FILE_PENALTY: i32 = -30;
// pub const SAFE_CHECK_BONUS: i32 = 15;

const LAZY_EVAL_MARGIN: i32 = 600;

// Mobility Weights [Piece][SquareCount] -> (Offset, Weight)
const MOBILITY_BONUS: [(i32, i32); 4] = [
    (0, 4), // Knight
    (1, 4), // Bishop
    (2, 4), // Rook
    (4, 4), // Queen
];

// --- PIECE-SQUARE TABLES (PeSTO) ---
#[rustfmt::skip] pub static MG_PAWN_TABLE: [EvalValue; 64] = a![ 0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20, -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0 ];
#[rustfmt::skip] pub static EG_PAWN_TABLE: [EvalValue; 64] = a![ 0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0 ];

#[rustfmt::skip] pub static MG_KNIGHT_TABLE: [EvalValue; 64] = a![ -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84, 129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23 ];
#[rustfmt::skip] pub static EG_KNIGHT_TABLE: [EvalValue; 64] = a![ -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64 ];

#[rustfmt::skip] pub static MG_BISHOP_TABLE: [EvalValue; 64] = a![ -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50, 37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18, 10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21 ];
#[rustfmt::skip] pub static EG_BISHOP_TABLE: [EvalValue; 64] = a![ -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 5, 10, 10, -6, -7, -11, -17, -9, -4, -9, -4, -6, -17, -21, -17, -21, -8, -4, -6, -6, -8, -20 ];

#[rustfmt::skip] pub static MG_ROOK_TABLE: [EvalValue; 64] = a![ 32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16, -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26 ];
#[rustfmt::skip] pub static EG_ROOK_TABLE: [EvalValue; 64] = a![ 13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20 ];

#[rustfmt::skip] pub static MG_QUEEN_TABLE: [EvalValue; 64] = a![ -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50 ];
#[rustfmt::skip] pub static EG_QUEEN_TABLE: [EvalValue; 64] = a![ -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9, 3, 22, 24, 45, 43, 40, 36, 14, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5, -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41 ];

#[rustfmt::skip] pub static MG_KING_TABLE: [EvalValue; 64] = a![ -65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14 ];
#[rustfmt::skip] pub static EG_KING_TABLE: [EvalValue; 64] = a![ -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43 ];

// --- STRUCTS (REQUIRED for Tuning) ---
pub struct Trace {
    pub terms: Vec<(usize, i8)>,
    pub fixed_mg: i32,
    pub fixed_eg: i32,
}

impl Trace {
    pub fn new() -> Self {
        Self {
            terms: Vec::with_capacity(64),
            fixed_mg: 0,
            fixed_eg: 0,
        }
    }

    #[inline(always)]
    pub fn add(&mut self, index: usize, count: i8) {
        self.terms.push((index, count));
    }
}

// --- INIT ---

pub fn init_eval() {}

// --- MAIN EVAL ---
pub fn evaluate(state: &GameState, accumulators: &Option<&[crate::nnue::Accumulator; 2]>, alpha: i32, beta: i32) -> i32 {
    let lazy_score = evaluate_lazy(state);
    let margin = LAZY_EVAL_MARGIN;

    if lazy_score + margin <= alpha {
        return alpha;
    }
    if lazy_score - margin >= beta {
        return beta;
    }

    if let Some(acc) = accumulators {
        if crate::nnue::NETWORK.get().is_some() {
            let score = crate::nnue::evaluate(
                &acc[state.side_to_move],
                &acc[1 - state.side_to_move],
            );
            return score;
        }
    }

    // Fallback to HCE
    let score = evaluate_hce(state, alpha, beta);
    if state.side_to_move == BLACK {
        -score
    } else {
        score
    }
}

// Helper for Lazy Eval (Pass 1 of HCE)
fn evaluate_lazy(state: &GameState) -> i32 {
    let mut mg = 0;
    let mut eg = 0;
    let mut phase = 0;

    let pawn_entry = crate::pawn::evaluate_pawns(state);
    mg += pawn_entry.score_mg;
    eg += pawn_entry.score_eg;

    for side in [WHITE, BLACK] {
        let us_sign = if side == WHITE { 1 } else { -1 };

        for piece_type in 0..6 {
            let piece_idx = if side == WHITE { piece_type } else { piece_type + 6 };
            let mut bb = state.bitboards[piece_idx];

            let count = bb.count_bits() as i32;
            phase += count * PHASE_WEIGHTS[piece_type];

            let base_mg = MG_VALS[piece_type].val();
            let base_eg = EG_VALS[piece_type].val();

            while bb.0 != 0 {
                let sq = bb.get_lsb_index() as usize;
                bb.pop_bit(sq as u8);
                mg += (base_mg + get_pst(piece_type, sq, side, true)) * us_sign;
                eg += (base_eg + get_pst(piece_type, sq, side, false)) * us_sign;
            }
        }
    }

    let phase_clamped = phase.clamp(0, 24);
    let mut score = (mg * phase_clamped + eg * (24 - phase_clamped)) / 24;
    let scale = crate::endgame::get_scale_factor(state, score);
    score = (score * scale) / 128;

    if state.side_to_move == BLACK { -score } else { score }
}

pub fn evaluate_hce(state: &GameState, alpha: i32, beta: i32) -> i32 {
    let mut mg = 0;
    let mut eg = 0;
    let mut phase = 0;

    let pawn_entry = crate::pawn::evaluate_pawns(state);
    mg += pawn_entry.score_mg;
    eg += pawn_entry.score_eg;

    for side in [WHITE, BLACK] {
        let us_sign = if side == WHITE { 1 } else { -1 };

        for piece_type in 0..6 {
            let piece_idx = if side == WHITE {
                piece_type
            } else {
                piece_type + 6
            };
            let mut bb = state.bitboards[piece_idx];

            let count = bb.count_bits() as i32;
            phase += count * PHASE_WEIGHTS[piece_type];

            let base_mg = MG_VALS[piece_type].val();
            let base_eg = EG_VALS[piece_type].val();

            while bb.0 != 0 {
                let sq = bb.get_lsb_index() as usize;
                bb.pop_bit(sq as u8);

                mg += (base_mg + get_pst(piece_type, sq, side, true)) * us_sign;
                eg += (base_eg + get_pst(piece_type, sq, side, false)) * us_sign;
            }
        }
    }

    let phase_clamped = phase.clamp(0, 24);
    let mut score = (mg * phase_clamped + eg * (24 - phase_clamped)) / 24;
    let scale = crate::endgame::get_scale_factor(state, score);
    score = (score * scale) / 128;

    let score_perspective = if state.side_to_move == BLACK { -score } else { score };

    if score_perspective + LAZY_EVAL_MARGIN <= alpha {
        return if state.side_to_move == BLACK { -alpha } else { alpha };
    }
    if score_perspective - LAZY_EVAL_MARGIN >= beta {
        return if state.side_to_move == BLACK { -beta } else { beta };
    }

    // PASS 2: Expensive
    let mut king_rings = [Bitboard(0); 2];
    let mut king_sqs = [0usize; 2];
    for side in [WHITE, BLACK] {
        let k_bb = state.bitboards[if side == WHITE { K } else { k }];
        let k_sq = if k_bb.0 != 0 {
            k_bb.get_lsb_index() as usize
        } else {
            0
        };
        king_sqs[side] = k_sq;
        king_rings[side] = crate::movegen::get_king_attacks(k_sq as u8);
    }

    let mut attacks_by_side = [Bitboard(0); 2];
    let mut king_attack_weight = [0; 2];
    let mut king_attack_count = [0; 2];
    let mut ring_attack_counts = [[0u8; 64]; 2];

    let mut coordination_score = [0; 2];
    let occ = state.occupancies[BOTH];

    for side in [WHITE, BLACK] {
        let us = side;
        let them = 1 - side;
        let us_sign = if us == WHITE { 1 } else { -1 };

        let king_ring_them = king_rings[them];

        let us_occupancies = state.occupancies[us];
        let my_rooks = state.bitboards[if us == WHITE { R } else { r }];
        let my_bishops = state.bitboards[if us == WHITE { B } else { b }];
        let my_queens = state.bitboards[if us == WHITE { Q } else { q }];

        for piece_type in 0..6 {
            let piece_idx = if us == WHITE {
                piece_type
            } else {
                piece_type + 6
            };
            let mut bb = state.bitboards[piece_idx];

            while bb.0 != 0 {
                let sq = bb.get_lsb_index() as usize;
                bb.pop_bit(sq as u8);

                let attacks = match piece_type {
                    N => crate::movegen::get_knight_attacks(sq as u8),
                    B => bitboard::get_bishop_attacks(sq as u8, occ),
                    R => bitboard::get_rook_attacks(sq as u8, occ),
                    Q => bitboard::get_queen_attacks(sq as u8, occ),
                    K => crate::movegen::get_king_attacks(sq as u8),
                    P => bitboard::pawn_attacks(Bitboard(1 << sq), us),
                    _ => Bitboard(0),
                };

                attacks_by_side[us] = attacks_by_side[us] | attacks;

                if piece_type != P && piece_type != K {
                    let mob_cnt = (attacks & !us_occupancies).count_bits() as i32;
                    let mob_idx = match piece_type {
                        N => 0,
                        B => 1,
                        R => 2,
                        Q => 3,
                        _ => 0,
                    };
                    let (offset, weight) = MOBILITY_BONUS[mob_idx];
                    let s = (mob_cnt - offset) * weight;
                    mg += s * us_sign;
                    eg += s * us_sign;
                }

                if piece_type == N || piece_type == B || piece_type == R {
                    let s = crate::threat::is_dominant_square(state, sq as u8, piece_type, us);
                    mg += s * us_sign;
                    eg += s * us_sign;
                }

                if piece_type != K && piece_type != P {
                    let k_file = king_sqs[them] % 8;
                    let k_rank = king_sqs[them] / 8;
                    let d_file = (k_file as i32 - (sq % 8) as i32).abs();
                    let d_rank = (k_rank as i32 - (sq / 8) as i32).abs();
                    let dist = d_file.max(d_rank) as usize;
                    let pen = KING_TROPISM_PENALTY[dist];
                    mg += pen * us_sign;
                    eg += (pen / 2) * us_sign;
                }

                if piece_type != K {
                    let att_on_ring = attacks & king_ring_them;
                    if att_on_ring.0 != 0 {
                        let weight = match piece_type {
                            P => 10,
                            N => 25,
                            B => 25,
                            R => 50,
                            Q => 75,
                            _ => 0,
                        };
                        king_attack_weight[them] += weight;
                        king_attack_count[them] += 1;

                        let mut iter = att_on_ring;
                        while iter.0 != 0 {
                            let s = iter.get_lsb_index();
                            iter.pop_bit(s as u8);
                            ring_attack_counts[them][s as usize] += 1;
                        }
                    }
                }

                if piece_type == R || piece_type == B || piece_type == Q {
                    if piece_type == R {
                        if (attacks & (my_rooks | my_queens)).0 != 0 {
                            coordination_score[us] += 10;
                        }
                    } else if piece_type == B {
                        if (attacks & (my_bishops | my_queens)).0 != 0 {
                            coordination_score[us] += 10;
                        }
                    } else if piece_type == Q {
                        if (bitboard::get_rook_attacks(sq as u8, occ) & my_rooks).0 != 0 {
                            coordination_score[us] += 10;
                        }
                        if (bitboard::get_bishop_attacks(sq as u8, occ) & my_bishops).0 != 0 {
                            coordination_score[us] += 10;
                        }
                    }
                }
            }
        }
    }

    for side in [WHITE, BLACK] {
        let us_sign = if side == WHITE { 1 } else { -1 };
        let k_sq = king_sqs[side];

        let mut shield_pen = 0;
        let k_rank = k_sq / 8;
        if (side == WHITE && k_rank < 3) || (side == BLACK && k_rank > 4) {
            let my_pawns = state.bitboards[if side == WHITE { P } else { p }];
            let enemy_pawns = state.bitboards[if side == WHITE { p } else { P }];
            let k_file = k_sq % 8;
            for f_offset in -1..=1 {
                let f = k_file as i32 + f_offset;
                if f >= 0 && f <= 7 {
                    let mask = bitboard::file_mask(f as usize);
                    if (my_pawns.0 & mask.0).count_ones() == 0 {
                        shield_pen += SHIELD_MISSING_PENALTY;
                        if (enemy_pawns.0 & mask.0).count_ones() == 0 {
                            shield_pen += SHIELD_OPEN_FILE_PENALTY;
                        }
                    }
                }
            }
        }
        mg += shield_pen * us_sign;

        if pawn_entry.pawn_attacks[1 - side].get_bit(k_sq as u8) {
            mg -= 50 * us_sign;
        }

        let mut danger = king_attack_weight[side];
        if king_attack_count[side] >= 2 {
            danger += king_attack_count[side] * 10;
        }

        let ring = king_rings[side];
        let undefended = ring & !attacks_by_side[side];
        let attacked = ring & attacks_by_side[1 - side];
        let danger_zone = undefended & attacked;
        danger += (danger_zone.count_bits() as i32) * 10;

        let mut cluster_pen = 0;
        let mut r_iter = ring;
        while r_iter.0 != 0 {
            let s = r_iter.get_lsb_index();
            r_iter.pop_bit(s as u8);
            let c = ring_attack_counts[side][s as usize];
            if c >= 2 {
                cluster_pen += (c as i32 - 1) * 20;
            }
        }

        if danger > 50 {
            mg -= danger * us_sign;
        }
        mg -= cluster_pen * us_sign;
        eg -= (cluster_pen / 2) * us_sign;
    }

    mg += coordination_score[WHITE] - coordination_score[BLACK];
    eg += (coordination_score[WHITE] - coordination_score[BLACK]) / 2;

    let us = state.side_to_move;
    let them = 1 - us;
    let us_sign = if us == WHITE { 1 } else { -1 };

    let mut hanging_val = 0;
    let us_occ = state.occupancies[us];
    let attacked_by_them = us_occ & attacks_by_side[them];
    let mut iter = attacked_by_them;
    while iter.0 != 0 {
        let sq = iter.get_lsb_index() as u8;
        iter.pop_bit(sq);

        if sq == king_sqs[us] as u8 {
            continue;
        }

        let defended = attacks_by_side[us].get_bit(sq);
        let val = get_piece_value(state, sq);

        if !defended {
            hanging_val += val;
        } else {
            let pawn_attacks = bitboard::pawn_attacks(Bitboard(1 << sq), us);
            if (pawn_attacks & state.bitboards[if them == WHITE { P } else { p }]).0 != 0 {
                if val > 100 {
                    hanging_val += val - 100;
                }
            }
        }
    }

    mg -= hanging_val * us_sign;
    eg -= (hanging_val / 2) * us_sign;

    let phase_clamped = phase.clamp(0, 24);
    let mut score = (mg * phase_clamped + eg * (24 - phase_clamped)) / 24;
    let scale = crate::endgame::get_scale_factor(state, score);
    score = (score * scale) / 128;

    score
}

fn evaluate_fixed(state: &GameState) -> (i32, i32) {
    let mut mg = 0;
    let mut eg = 0;

    for piece in 0..6 {
        let mut bb = state.bitboards[piece];
        while bb.0 != 0 {
            let sq = bb.get_lsb_index() as usize;
            bb.pop_bit(sq as u8);
            mg += MG_VALS[piece].val() + get_pst(piece, sq, WHITE, true);
            eg += EG_VALS[piece].val() + get_pst(piece, sq, WHITE, false);
        }
        let mut bb = state.bitboards[piece + 6];
        while bb.0 != 0 {
            let sq = bb.get_lsb_index() as usize;
            bb.pop_bit(sq as u8);
            mg -= MG_VALS[piece].val() + get_pst(piece, sq, BLACK, true);
            eg -= EG_VALS[piece].val() + get_pst(piece, sq, BLACK, false);
        }
    }
    (mg, eg)
}

fn get_pst(piece: usize, sq: usize, side: usize, is_mg: bool) -> i32 {
    let index = if side == WHITE { sq ^ 56 } else { sq };
    match piece {
        P => {
            if is_mg {
                MG_PAWN_TABLE[index].val()
            } else {
                EG_PAWN_TABLE[index].val()
            }
        }
        N => {
            if is_mg {
                MG_KNIGHT_TABLE[index].val()
            } else {
                EG_KNIGHT_TABLE[index].val()
            }
        }
        B => {
            if is_mg {
                MG_BISHOP_TABLE[index].val()
            } else {
                EG_BISHOP_TABLE[index].val()
            }
        }
        R => {
            if is_mg {
                MG_ROOK_TABLE[index].val()
            } else {
                EG_ROOK_TABLE[index].val()
            }
        }
        Q => {
            if is_mg {
                MG_QUEEN_TABLE[index].val()
            } else {
                EG_QUEEN_TABLE[index].val()
            }
        }
        K => {
            if is_mg {
                MG_KING_TABLE[index].val()
            } else {
                EG_KING_TABLE[index].val()
            }
        }
        _ => 0,
    }
}

fn get_piece_value(state: &GameState, sq: u8) -> i32 {
    for piece in 0..12 {
        if state.bitboards[piece].get_bit(sq) {
            return match piece % 6 {
                0 => 100,   // Pawn
                1 => 320,   // Knight
                2 => 330,   // Bishop
                3 => 500,   // Rook
                4 => 900,   // Queen
                5 => 20000, // King
                _ => 0,
            };
        }
    }
    0
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum_256_epi32(v: __m256i) -> i32 {
    let v128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let v64 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, 0b00_00_11_10));
    let v32 = _mm_add_epi32(v64, _mm_shuffle_epi32(v64, 0b00_00_00_01));
    _mm_cvtsi128_si32(v32)
}

// --- TRACE (Required for Tuning) ---
pub fn trace_evaluate(state: &GameState, trace: &mut Trace) -> i32 {
    let (fix_mg, fix_eg) = evaluate_fixed(state);
    trace.fixed_mg = fix_mg;
    trace.fixed_eg = fix_eg;
    for piece in 0..6 {
        let w_count = state.bitboards[piece].count_bits() as i8;
        let b_count = state.bitboards[piece + 6].count_bits() as i8;
        let net = w_count - b_count;
        if net != 0 {
            trace.add(piece, net);
            trace.add(piece + 6, net);
        }
    }
    // Tuning: just return optimized eval, assuming trace is only for fixed params
    evaluate_hce(state, -32000, 32000)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{GameState, BLACK};

    #[test]
    fn test_king_hanging_bug() {
        // Initialize globals
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();
        crate::eval::init_eval();
        crate::threat::init_threat();

        // Position: White King at e1, Black Rook at e2. King is in check.
        // No other pieces. King is attacked by Rook.
        // King square e1 is NOT defended by any white piece (no other pieces).
        // Bug: King is marked as "hanging", penalty 20,000 applied.
        let fen = "8/8/8/8/8/8/4r3/4K3 w - - 0 1";
        let state = GameState::parse_fen(fen);

        // Pass None for accumulator (no NNUE in HCE test)
        let score = evaluate(&state, &None, -32000, 32000);
        println!("Score: {}", score);

        // Without fix, score should be roughly:
        // Material: 0 vs 500 (Rook) -> -500
        // Hanging King: -20,000
        // Total: ~ -20,500
        //
        // With fix, score should be ~ -500 (plus/minus PST/positional).

        assert!(
            score > -10000,
            "Score {} indicates King is treated as hanging (approx -20000)",
            score
        );
    }

    #[test]
    fn test_hce_evaluation_perspective() {
        // Initialize globals
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();
        crate::movegen::init_move_tables();
        crate::eval::init_eval();
        crate::threat::init_threat();

        // 1. Black Winning Position (Black King safe, White King in corner, Black Queen attacking)
        // FEN: 7k/8/8/8/8/8/8/K6q b - - 0 1
        // Black to move.
        let fen = "7k/8/8/8/8/8/8/K6q b - - 0 1";
        let state = GameState::parse_fen(fen);

        assert_eq!(state.side_to_move, BLACK);

        // Ensure NNUE is NOT used
        assert!(crate::nnue::NETWORK.get().is_none());

        let score = evaluate(&state, &None, -32000, 32000);

        println!("Score for Black (Winning): {}", score);

        // Score should be positive because it is relative to side to move (Black).
        // Since Black is winning, score > 0.
        // If bug exists, it returns absolute score (negative), so score < 0.
        assert!(
            score > 0,
            "Score should be positive for winning side (Black), got {}",
            score
        );
    }
}
