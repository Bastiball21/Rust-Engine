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

// --- TUNED PARAMETERS ---
#[rustfmt::skip] pub static MG_VALS: [EvalValue; 6] = a![ 5, 131, 139, 247, 990, 0 ];
#[rustfmt::skip] pub static EG_VALS: [EvalValue; 6] = a![ -32, 2, -7, -32, 884, 0 ];

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
#[rustfmt::skip] pub static MG_PAWN_TABLE: [EvalValue; 64] = a![ 0, 0, 0, 0, 0, 0, 0, 0, 87, 123, 53, 87, 62, 121, 28, -15, -15, -6, 10, 10, 52, 41, 14, -27, -25, -27, -10, -33, -37, 5, -22, -27, -24, -16, 4, -16, -4, 0, -2, -8, -23, 12, -1, 44, 36, 5, 31, -1, -4, 30, 12, 34, 28, 31, 47, 15, 0, 0, 0, 0, 0, 0, 0, 0 ];
#[rustfmt::skip] pub static EG_PAWN_TABLE: [EvalValue; 64] = a![ 0, 0, 0, 0, 0, 0, 0, 0, 139, 140, 135, 114, 130, 114, 147, 164, 56, 61, 53, 35, 36, 21, 55, 60, 19, 5, 4, -10, -10, 10, -3, 19, 25, 10, 14, 24, 19, 11, 21, 21, 14, 20, 12, 31, 28, 20, 21, 20, 20, 16, 2, 18, 16, 2, 2, 14, 0, 0, 0, 0, 0, 0, 0, 0 ];
#[rustfmt::skip] pub static MG_KNIGHT_TABLE: [EvalValue; 64] = a![ -168, -89, -34, -49, 59, -97, -14, -108, -71, -42, 67, 32, 20, 57, 6, -17, -47, 55, 27, 52, 75, 122, 69, 43, -8, 3, 8, 28, 4, 59, 3, 19, -7, 1, 1, 0, 9, 1, 19, -7, -18, -2, -12, -1, 12, -15, 24, -11, -28, -52, -9, 0, 3, 12, -13, -19, -103, 13, -52, -28, -11, -22, 5, -22 ];
#[rustfmt::skip] pub static EG_KNIGHT_TABLE: [EvalValue; 64] = a![ -57, -38, -15, -29, -33, -28, -60, -99, -24, -11, -29, -9, -12, -29, -24, -50, -26, -25, -4, -7, -12, -20, -23, -41, -14, -4, 1, -2, 0, -3, 0, -18, -18, -10, -1, 7, -1, 5, 0, -17, -22, -6, -10, 1, 3, -13, -18, -21, -41, -21, -8, -7, -4, -21, -22, -41, -27, -45, -18, -13, -19, -11, -42, -61 ];
#[rustfmt::skip] pub static MG_BISHOP_TABLE: [EvalValue; 64] = a![ -32, 2, -82, -37, -24, -43, 6, -10, -24, 11, -19, -15, 28, 55, 13, -45, -18, 33, 35, 33, 30, 46, 33, -6, -3, 4, 13, 37, 22, 29, 6, -2, -6, 9, -1, 12, 19, -7, 6, 2, -2, 10, 4, -16, 0, 11, 7, 5, 3, -9, 7, 20, -3, 15, -7, 0, -30, -1, 39, -17, -8, 33, -39, -22 ];
#[rustfmt::skip] pub static EG_BISHOP_TABLE: [EvalValue; 64] = a![ -16, -21, -11, -10, -5, -10, -16, -25, -8, -8, 1, -13, -5, -17, -4, -12, 0, -15, -11, -8, -10, 0, -4, 1, -6, -2, 0, -8, -3, -1, 1, 0, -11, -3, -2, -2, -8, -1, -5, -10, -14, -9, -6, -11, -1, -20, -11, -14, -16, -7, -9, -4, -11, -11, -22, -21, -12, -18, 1, -2, -2, -3, -9, -20 ];
#[rustfmt::skip] pub static MG_ROOK_TABLE: [EvalValue; 64] = a![ 25, 36, 26, 44, 58, 6, 29, 39, 18, 20, 47, 52, 71, 61, 20, 40, -9, 13, 21, 28, 11, 41, 57, 15, -23, -11, 1, 20, 19, 32, -9, -20, -37, -26, -16, -4, 6, -7, 4, -27, -44, -26, -18, -17, -2, 0, -6, -35, -44, -18, -22, -14, -2, 9, -4, -71, -1, -12, -14, -11, -12, -3, -25, 0 ];
#[rustfmt::skip] pub static EG_ROOK_TABLE: [EvalValue; 64] = a![ -6, -5, 2, -1, 0, 3, 2, -3, -11, -12, -14, -12, -22, -13, -5, -6, -4, -6, -7, -12, -8, -12, -12, -6, 2, -1, -2, -9, -8, -4, -4, 1, 0, 0, 0, -4, -8, -9, -11, -11, -6, -3, -11, -3, -12, -16, -11, -16, -9, -13, -7, -5, -12, -13, -12, -3, -13, -2, -7, -20, -20, -10, 5, -13 ];
#[rustfmt::skip] pub static MG_QUEEN_TABLE: [EvalValue; 64] = a![ -30, -2, 26, 10, 56, 42, 41, 41, -21, -39, -5, -1, -17, 53, 24, 52, -12, -17, 3, 5, 24, 52, 41, 54, -25, -25, -18, -20, -4, 13, -3, 1, -9, -28, -10, -10, -7, -5, 0, -5, -11, 2, -13, -3, -7, 0, 10, 6, -30, -6, 7, 1, 0, 13, 0, 2, 2, -9, -2, 29, -8, -17, -28, -48 ];
#[rustfmt::skip] pub static EG_QUEEN_TABLE: [EvalValue; 64] = a![ -9, 19, 18, 24, 24, 16, 8, 17, -15, 19, 31, 38, 56, 22, 27, 0, -17, 6, 6, 46, 42, 32, 15, 7, 4, 22, 21, 41, 37, 37, 34, 13, -17, 26, 16, 41, 27, 32, 38, 22, -14, -26, 12, 3, 5, 15, 9, 5, -19, -23, -30, -15, -16, -24, -34, -31, -30, -24, -19, -37, -2, -27, -18, -40 ];
#[rustfmt::skip] pub static MG_KING_TABLE: [EvalValue; 64] = a![ -64, 22, 15, -15, -56, -34, 1, 12, 28, -1, -20, -8, -8, -4, -38, -29, -9, 23, 0, -17, -21, 4, 19, -22, -16, -21, -13, -28, -32, -28, -17, -36, -47, 0, -27, -41, -49, -48, -35, -51, -12, -11, -21, -46, -46, -32, -18, -24, 3, 8, -4, -55, -33, -19, -2, 0, -10, 29, 11, -34, 26, -7, -9, 29 ];
#[rustfmt::skip] pub static EG_KING_TABLE: [EvalValue; 64] = a![ -73, -36, -19, -20, -12, 12, 2, -18, -12, 15, 10, 11, 12, 33, 18, 7, 9, 13, 16, 8, 13, 36, 31, 11, -6, 17, 15, 16, 14, 17, 12, 2, -12, 0, 18, 15, 8, 2, -2, -11, -11, 9, 15, 23, 9, 7, -3, -4, -20, 0, 13, 19, 10, 4, -7, -8, -45, -25, -8, 20, 2, 6, -4, -7 ];

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
    // 1. Calculate Fixed Score (Evaluation terms NOT being tuned, e.g. Mobility, Kingsafety if not in params)
    // We assume 'evaluate_fixed' returns the FULL score currently, which includes PSTs.
    // Since we want to TUNE PSTs, we must subtract their current values from the fixed score
    // OR (better) just calculate the truly fixed parts.

    // For simplicity with your current setup:
    // We will initialize fixed score to 0 and ONLY add the terms you are NOT tuning.
    // (Assuming you are only tuning Material and PSTs for now)

    trace.fixed_mg = 0;
    trace.fixed_eg = 0;

    // 2. Loop through all pieces to add Material and PST terms to the trace
    for piece_type in 0..6 {
        for side in [crate::state::WHITE, crate::state::BLACK] {
            let piece_idx = if side == crate::state::WHITE {
                piece_type
            } else {
                piece_type + 6
            };
            let mut bb = state.bitboards[piece_idx];

            // Sign: +1 for White, -1 for Black (relative to the trace perspective)
            // The tuner expects counts. White = +1 count, Black = -1 count.
            let sign = if side == crate::state::WHITE { 1 } else { -1 };

            while bb.0 != 0 {
                let sq = bb.get_lsb_index() as usize;
                bb.pop_bit(sq as u8);

                // --- A. Material Term ---
                // Material MG index: 0..5
                // Material EG index: 6..11
                if side == crate::state::WHITE {
                    trace.add(piece_type, 1); // MG Material White
                    trace.add(piece_type + 6, 1); // EG Material White
                } else {
                    trace.add(piece_type, -1); // MG Material Black
                    trace.add(piece_type + 6, -1); // EG Material Black
                }

                // --- B. PST Term ---
                // Tuning params structure:
                // Indices 0-11: Material
                // Indices 12+: PSTs.
                // Each piece type has 128 params (64 MG + 64 EG).
                // Offset = 12 + (piece_type * 128)

                let pst_base = 12 + (piece_type * 128);

                // For PSTs, the table is always from White's perspective.
                // If Black, we must flip the square (sq ^ 56).
                let table_sq = if side == crate::state::WHITE {
                    sq ^ 56
                } else {
                    sq
                };

                let mg_idx = pst_base + table_sq;
                let eg_idx = pst_base + 64 + table_sq;

                trace.add(mg_idx, sign);
                trace.add(eg_idx, sign);
            }
        }
    }

    // Return value doesn't matter much for trace generation, but we return HCE for consistency
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
