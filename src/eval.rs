// src/eval.rs
use crate::state::{GameState, WHITE, BLACK, BOTH, P, N, B, R, Q, K, p, n, b, r, q, k};
use crate::bitboard::{self, Bitboard, FILE_A, FILE_H};
use crate::pawn::PawnTable;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, Ordering};
use crate::nnue::{self, NNUE};

// --- MACRO FOR ATOMIC ARRAYS ---
macro_rules! a { ($($x:expr),*) => { [ $(AtomicI32::new($x)),* ] } }

// --- CONFIGURATION (Safe Atomics for Tuning) ---
// Material
#[rustfmt::skip] pub static MG_VALS: [AtomicI32; 6] = a![ 82, 337, 365, 477, 1025, 0 ];
#[rustfmt::skip] pub static EG_VALS: [AtomicI32; 6] = a![ 94, 281, 297, 512,  936, 0 ];

// Phase Weights
pub const PHASE_WEIGHTS: [i32; 6] = [0, 1, 1, 2, 4, 0];
pub const TOTAL_PHASE: i32 = 24;

// King Safety Weights
pub const KING_TROPISM_PENALTY: [i32; 8] = [ 10, 8, 5, 2, 0, 0, 0, 0 ];
pub const SHIELD_MISSING_PENALTY: i32 = -20;
pub const SHIELD_OPEN_FILE_PENALTY: i32 = -30;
pub const SAFE_CHECK_BONUS: i32 = 15;

// Mobility Weights [Piece][SquareCount] -> (Offset, Weight)
const MOBILITY_BONUS: [(i32, i32); 4] = [
    (0, 4), // Knight
    (1, 4), // Bishop
    (2, 4), // Rook
    (4, 4), // Queen
];

// --- ATOMIC PIECE-SQUARE TABLES (PeSTO) ---
#[rustfmt::skip] pub static MG_PAWN_TABLE: [AtomicI32; 64] = a![ 0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20, -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0 ];
#[rustfmt::skip] pub static EG_PAWN_TABLE: [AtomicI32; 64] = a![ 0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0 ];

#[rustfmt::skip] pub static MG_KNIGHT_TABLE: [AtomicI32; 64] = a![ -167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84, 129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23 ];
#[rustfmt::skip] pub static EG_KNIGHT_TABLE: [AtomicI32; 64] = a![ -58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64 ];

#[rustfmt::skip] pub static MG_BISHOP_TABLE: [AtomicI32; 64] = a![ -29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50, 37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18, 10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21 ];
#[rustfmt::skip] pub static EG_BISHOP_TABLE: [AtomicI32; 64] = a![ -14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 5, 10, 10, -6, -7, -11, -17, -9, -4, -9, -4, -6, -17, -21, -17, -21, -8, -4, -6, -6, -8, -20 ];

#[rustfmt::skip] pub static MG_ROOK_TABLE: [AtomicI32; 64] = a![ 32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16, -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26 ];
#[rustfmt::skip] pub static EG_ROOK_TABLE: [AtomicI32; 64] = a![ 13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20 ];

#[rustfmt::skip] pub static MG_QUEEN_TABLE: [AtomicI32; 64] = a![ -28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50 ];
#[rustfmt::skip] pub static EG_QUEEN_TABLE: [AtomicI32; 64] = a![ -9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9, 3, 22, 24, 45, 43, 40, 36, 14, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5, -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41 ];

#[rustfmt::skip] pub static MG_KING_TABLE: [AtomicI32; 64] = a![ -65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14 ];
#[rustfmt::skip] pub static EG_KING_TABLE: [AtomicI32; 64] = a![ -74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43 ];

// --- STRUCTS (REQUIRED for Tuning) ---
pub struct Trace {
    pub terms: Vec<(usize, i8)>,
    pub fixed_mg: i32,
    pub fixed_eg: i32,
}

impl Trace {
    pub fn new() -> Self {
        Self { terms: Vec::with_capacity(64), fixed_mg: 0, fixed_eg: 0 }
    }

    #[inline(always)]
    pub fn add(&mut self, index: usize, count: i8) {
        self.terms.push((index, count));
    }
}

// --- INIT ---
static PAWN_TABLE: OnceLock<PawnTable> = OnceLock::new();

pub fn init_eval() {
    PAWN_TABLE.get_or_init(|| PawnTable::new());
}

// --- MAIN EVAL ---
pub fn evaluate(state: &GameState) -> i32 {
    // 1. NNUE (If available)
    if let Some(net) = NNUE.get() {
        // Check if AVX2 is available via crate feature or architecture
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Ensure accumulators are fresh
            let mut acc_us = state.accumulator[state.side_to_move];
            let mut acc_them = state.accumulator[1 - state.side_to_move];

            if state.dirty {
                acc_us.refresh_with_weights(state, state.side_to_move, net);
                acc_them.refresh_with_weights(state, 1 - state.side_to_move, net);
            }

            return unsafe {
                 evaluate_nnue_avx2_inline(&acc_us, &acc_them, net)
            };
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // Fallback (Original Slow Code)
            let mut acc_us = state.accumulator[state.side_to_move];
            let mut acc_them = state.accumulator[1 - state.side_to_move];

            if state.dirty {
                acc_us.refresh_with_weights(state, state.side_to_move, net);
                acc_them.refresh_with_weights(state, 1 - state.side_to_move, net);
            }

            let mut input = [0i8; 512];
            for i in 0..256 {
                input[i] = acc_us.v[i].clamp(0, 127) as i8;
                input[256 + i] = acc_them.v[i].clamp(0, 127) as i8;
            }

            let mut layer1_out = [0i8; 32];
            for i in 0..32 {
                let mut sum = net.layer1_biases[i];
                for j in 0..512 {
                    sum += (input[j] as i32) * (net.layer1_weights[i * 512 + j] as i32);
                }
                layer1_out[i] = (sum >> 6).clamp(0, 127) as i8;
            }

            let mut layer2_out = [0i8; 32];
            for i in 0..32 {
                let mut sum = net.layer2_biases[i];
                for j in 0..32 {
                    sum += (layer1_out[j] as i32) * (net.layer2_weights[i * 32 + j] as i32);
                }
                layer2_out[i] = (sum >> 6).clamp(0, 127) as i8;
            }

            let mut sum = net.output_bias;
            for j in 0..32 {
                sum += (layer2_out[j] as i32) * (net.output_weights[j] as i32);
            }

            return (sum / 16) + 20;
        }
    }
    // 2. HCE Fallback
    evaluate_hce(state)
}

// --- HCE LOGIC ---
pub fn evaluate_hce(state: &GameState) -> i32 {
    let mut mg = 0;
    let mut eg = 0;
    let mut phase = 0;

    // A. Fixed Material + PST
    let (fix_mg, fix_eg) = evaluate_fixed(state);
    mg += fix_mg;
    eg += fix_eg;

    // B. Phase Calculation
    for i in 0..6 {
        let w_cnt = state.bitboards[i].count_bits() as i32;
        let b_cnt = state.bitboards[i+6].count_bits() as i32;
        phase += (w_cnt + b_cnt) * PHASE_WEIGHTS[i];
    }

    // C. Pawn Hash Integration
    let pawn_entry = PAWN_TABLE.get().unwrap().probe(state);
    mg += pawn_entry.score_mg;
    eg += pawn_entry.score_eg;

    // D. Mobility & King Safety
    let (w_mob_mg, w_mob_eg) = evaluate_mobility(state, WHITE);
    let (b_mob_mg, b_mob_eg) = evaluate_mobility(state, BLACK);
    mg += w_mob_mg - b_mob_mg;
    eg += w_mob_eg - b_mob_eg;

    let (w_safe_mg, w_safe_eg) = evaluate_king(state, WHITE, &pawn_entry);
    let (b_safe_mg, b_safe_eg) = evaluate_king(state, BLACK, &pawn_entry);
    mg += w_safe_mg - b_safe_mg;
    eg += w_safe_eg - b_safe_eg;

    // E. Scaling
    let phase = phase.clamp(0, 24);
    let mut score = (mg * phase + eg * (24 - phase)) / 24;
    
    let scale = crate::endgame::get_scale_factor(state, score);
    score = (score * scale) / 128;

    if state.side_to_move == WHITE { score } else { -score }
}

fn evaluate_fixed(state: &GameState) -> (i32, i32) {
    let mut mg = 0;
    let mut eg = 0;

    for piece in 0..6 {
        let mut bb = state.bitboards[piece];
        while bb.0 != 0 {
            let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
            mg += MG_VALS[piece].load(Ordering::Relaxed) + get_pst(piece, sq, WHITE, true);
            eg += EG_VALS[piece].load(Ordering::Relaxed) + get_pst(piece, sq, WHITE, false);
        }
        let mut bb = state.bitboards[piece+6];
        while bb.0 != 0 {
            let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
            mg -= MG_VALS[piece].load(Ordering::Relaxed) + get_pst(piece, sq, BLACK, true);
            eg -= EG_VALS[piece].load(Ordering::Relaxed) + get_pst(piece, sq, BLACK, false);
        }
    }
    (mg, eg)
}

fn get_pst(piece: usize, sq: usize, side: usize, is_mg: bool) -> i32 {
    let index = if side == WHITE { sq ^ 56 } else { sq };
    match piece {
        P => if is_mg { MG_PAWN_TABLE[index].load(Ordering::Relaxed) } else { EG_PAWN_TABLE[index].load(Ordering::Relaxed) },
        N => if is_mg { MG_KNIGHT_TABLE[index].load(Ordering::Relaxed) } else { EG_KNIGHT_TABLE[index].load(Ordering::Relaxed) },
        B => if is_mg { MG_BISHOP_TABLE[index].load(Ordering::Relaxed) } else { EG_BISHOP_TABLE[index].load(Ordering::Relaxed) },
        R => if is_mg { MG_ROOK_TABLE[index].load(Ordering::Relaxed) } else { EG_ROOK_TABLE[index].load(Ordering::Relaxed) },
        Q => if is_mg { MG_QUEEN_TABLE[index].load(Ordering::Relaxed) } else { EG_QUEEN_TABLE[index].load(Ordering::Relaxed) },
        K => if is_mg { MG_KING_TABLE[index].load(Ordering::Relaxed) } else { EG_KING_TABLE[index].load(Ordering::Relaxed) },
        _ => 0
    }
}

// --- NEW FEATURES ---

fn evaluate_mobility(state: &GameState, side: usize) -> (i32, i32) {
    let mut mg = 0;
    let mut eg = 0;
    let us_bb = state.occupancies[side];
    let both_bb = state.occupancies[BOTH];

    // Knight, Bishop, Rook, Queen
    let pieces = [
        (if side == WHITE { N } else { n }, 0),
        (if side == WHITE { B } else { b }, 1),
        (if side == WHITE { R } else { r }, 2),
        (if side == WHITE { Q } else { q }, 3),
    ];

    for &(piece, idx) in &pieces {
        let mut bb = state.bitboards[piece];
        while bb.0 != 0 {
            let sq = bb.get_lsb_index() as u8; bb.pop_bit(sq);
            let attacks = match idx {
                0 => crate::movegen::get_knight_attacks(sq),
                1 => bitboard::get_bishop_attacks(sq, both_bb),
                2 => bitboard::get_rook_attacks(sq, both_bb),
                3 => bitboard::get_queen_attacks(sq, both_bb),
                _ => Bitboard(0)
            };
            let mob = (attacks & !us_bb).count_bits() as i32;
            let (offset, weight) = MOBILITY_BONUS[idx];
            let score = (mob - offset) * weight;
            mg += score; eg += score;
        }
    }
    (mg, eg)
}

fn evaluate_king(state: &GameState, side: usize, pawn_entry: &crate::pawn::PawnEntry) -> (i32, i32) {
    let mut mg = 0;
    let mut eg = 0;
    let king_pc = if side == WHITE { K } else { k };
    let king_sq = state.bitboards[king_pc].get_lsb_index() as usize;
    let king_file = king_sq % 8;
    let king_rank = king_sq / 8;

    let my_pawns = if side == WHITE { state.bitboards[P] } else { state.bitboards[p] };
    let enemy_pawns = if side == WHITE { state.bitboards[p] } else { state.bitboards[P] };

    // 1. SHIELD
    if (side == WHITE && king_rank < 3) || (side == BLACK && king_rank > 4) {
        for f_offset in -1..=1 {
            let f = king_file as i32 + f_offset;
            if f >= 0 && f <= 7 {
                let file_mask = bitboard::file_mask(f as usize);
                if (my_pawns.0 & file_mask.0).count_ones() == 0 {
                    mg += SHIELD_MISSING_PENALTY;
                    if (enemy_pawns.0 & file_mask.0).count_ones() == 0 { mg += SHIELD_OPEN_FILE_PENALTY; }
                }
            }
        }
    }

    // 2. STORM
    if pawn_entry.pawn_attacks[1 - side].get_bit(king_sq as u8) { mg -= 50; }

    // 3. TROPISM
    let enemy_start = if side == WHITE { n } else { N };
    let enemy_end = if side == WHITE { q } else { Q };
    for pc in enemy_start..=enemy_end {
        let mut bb = state.bitboards[pc];
        while bb.0 != 0 {
            let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
            let dist_file = (king_file as i32 - (sq % 8) as i32).abs();
            let dist_rank = (king_rank as i32 - (sq / 8) as i32).abs();
            let dist = dist_file.max(dist_rank) as usize;
            mg -= KING_TROPISM_PENALTY[dist];
            eg -= KING_TROPISM_PENALTY[dist] / 2;
        }
    }

    (mg, eg)
}

// --- NNUE ---
pub fn evaluate_nnue(state: &GameState) -> i32 {
    if let Some(net) = NNUE.get() {
        // Check if AVX2 is available via crate feature or architecture
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            // Ensure accumulators are fresh
            let mut acc_us = state.accumulator[state.side_to_move];
            let mut acc_them = state.accumulator[1 - state.side_to_move];

            if state.dirty {
                // SAFE: We do NOT hold the lock inside refresh_with_weights (it just uses 'net')
                // 'acc_us' is local copy (Accumulator is Copy)
                acc_us.refresh_with_weights(state, state.side_to_move, net);
                acc_them.refresh_with_weights(state, 1 - state.side_to_move, net);
            }

            return unsafe {
                 evaluate_nnue_avx2_inline(&acc_us, &acc_them, net)
            };
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            // Fallback (Original Slow Code)
            let mut acc_us = state.accumulator[state.side_to_move];
            let mut acc_them = state.accumulator[1 - state.side_to_move];

            if state.dirty {
                acc_us.refresh_with_weights(state, state.side_to_move, net);
                acc_them.refresh_with_weights(state, 1 - state.side_to_move, net);
            }

            let mut input = [0i8; 512];
            for i in 0..256 {
                input[i] = acc_us.v[i].clamp(0, 127) as i8;
                input[256 + i] = acc_them.v[i].clamp(0, 127) as i8;
            }

            let mut layer1_out = [0i8; 32];
            for i in 0..32 {
                let mut sum = net.layer1_biases[i];
                for j in 0..512 {
                    sum += (input[j] as i32) * (net.layer1_weights[i * 512 + j] as i32);
                }
                layer1_out[i] = (sum >> 6).clamp(0, 127) as i8;
            }

            let mut layer2_out = [0i8; 32];
            for i in 0..32 {
                let mut sum = net.layer2_biases[i];
                for j in 0..32 {
                    sum += (layer1_out[j] as i32) * (net.layer2_weights[i * 32 + j] as i32);
                }
                layer2_out[i] = (sum >> 6).clamp(0, 127) as i8;
            }

            let mut sum = net.output_bias;
            for j in 0..32 {
                sum += (layer2_out[j] as i32) * (net.output_weights[j] as i32);
            }

            return (sum / 16) + 20;
        }
    }
    // 2. HCE Fallback
    evaluate_hce(state)
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn evaluate_nnue_avx2_inline(acc_us: &crate::nnue::Accumulator, acc_them: &crate::nnue::Accumulator, net: &crate::nnue::NnueWeights) -> i32 {
    let mut input = [0i8; 512];
    let input_ptr = input.as_mut_ptr();

    // Clamp and Pack
    for i in 0..256 {
        *input_ptr.add(i) = acc_us.v[i].clamp(0, 127) as i8;
        *input_ptr.add(256+i) = acc_them.v[i].clamp(0, 127) as i8;
    }

    // --- LAYER 1 ---
    let mut layer1_out = [0i8; 32];

    for i in 0..32 {
        let mut sum_vec = _mm256_setzero_si256();
        let row_offset = i * 512;
        let weights_ptr = net.layer1_weights.as_ptr().add(row_offset);

        for idx in (0..512).step_by(32) {
            let inp = _mm256_loadu_si256(input_ptr.add(idx) as *const __m256i);
            let w = _mm256_loadu_si256(weights_ptr.add(idx) as *const __m256i);

            let product = _mm256_maddubs_epi16(inp, w);
            let ones = _mm256_set1_epi16(1);
            let sum_i32 = _mm256_madd_epi16(product, ones);
            sum_vec = _mm256_add_epi32(sum_vec, sum_i32);
        }

        let total = hsum_256_epi32(sum_vec) + net.layer1_biases[i];
        layer1_out[i] = (total >> 6).clamp(0, 127) as i8;
    }

    // --- LAYER 2 ---
    let mut layer2_out = [0i8; 32];
    let l1_vec = _mm256_loadu_si256(layer1_out.as_ptr() as *const __m256i);

    for i in 0..32 {
        let row_offset = i * 32;
        let w = _mm256_loadu_si256(net.layer2_weights.as_ptr().add(row_offset) as *const __m256i);

        let product = _mm256_maddubs_epi16(l1_vec, w);
        let ones = _mm256_set1_epi16(1);
        let sum_i32 = _mm256_madd_epi16(product, ones);

        let total = hsum_256_epi32(sum_i32) + net.layer2_biases[i];
        layer2_out[i] = (total >> 6).clamp(0, 127) as i8;
    }

    // --- OUTPUT ---
    let l2_vec = _mm256_loadu_si256(layer2_out.as_ptr() as *const __m256i);
    let out_w = _mm256_loadu_si256(net.output_weights.as_ptr() as *const __m256i);

    let product = _mm256_maddubs_epi16(l2_vec, out_w);
    let ones = _mm256_set1_epi16(1);
    let sum_i32 = _mm256_madd_epi16(product, ones);

    let final_sum = hsum_256_epi32(sum_i32) + net.output_bias;

    (final_sum / 16) + 20
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
    evaluate_hce(state)
}
