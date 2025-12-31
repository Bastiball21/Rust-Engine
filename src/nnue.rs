// src/nnue.rs
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::sync::OnceLock;
use crate::nnue_scratch::NNUEScratch;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

static EMBEDDED_NET: &[u8] = include_bytes!("../nn-new.nnue");

// Architecture Constants
#[cfg(not(feature = "nnue_512_64"))]
pub const L1_SIZE: usize = 256;
#[cfg(not(feature = "nnue_512_64"))]
pub const L2_SIZE: usize = 32;
#[cfg(not(feature = "nnue_512_64"))]
pub const L3_SIZE: usize = 32;
#[cfg(not(feature = "nnue_512_64"))]
pub const L4_SIZE: usize = 32;

#[cfg(feature = "nnue_512_64")]
pub const L1_SIZE: usize = 512;
#[cfg(feature = "nnue_512_64")]
pub const L2_SIZE: usize = 64;
// Unused in nnue_512_64 path (kept non-zero for scratch struct layout).
#[cfg(feature = "nnue_512_64")]
pub const L3_SIZE: usize = 1;
// Unused in nnue_512_64 path (kept non-zero for scratch struct layout).
#[cfg(feature = "nnue_512_64")]
pub const L4_SIZE: usize = 1;
pub const OUTPUT_SIZE: usize = 1;

pub const INPUT_SIZE: usize = 768;
pub const NUM_BUCKETS: usize = 32; // Matches bullet_lib ChessBuckets (Standard Mirrored)

#[cfg(not(feature = "nnue_512_64"))]
pub const NETWORK_MAGIC: u32 = 0xAE74E201;
#[cfg(feature = "nnue_512_64")]
pub const NETWORK_MAGIC: u32 = 0xAE74E202;

// Quantization Constants
const QA: i32 = 255;
const QB: i32 = 64;
const OUTPUT_SHIFT: i32 = 6; // QB = 64 -> Shift 6
const Q_ACTIVATION: i32 = 127; // Max activation for ClippedReLU (scaled)
const SCALE: i32 = 400; // Eval scale from trainer
const ACC_MAGIC: u16 = 0x1234;

// SAFE GLOBAL NETWORK
pub static NETWORK: OnceLock<Network> = OnceLock::new();

// SCReLU Lookup Table: (x^2) / 255 for x in [0, 255]
// Ensures bit-identical evaluation across Scalar and SIMD paths.
const SCRELU: [i16; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        // Match the integer approximation used in AVX2
        // y = sq + 128; y = y + (y >> 8); res = y >> 8;
        let sq = i as i32 * i as i32;
        let y = sq + 128;
        let y2 = y + (y >> 8);
        table[i] = (y2 >> 8) as i16;
        i += 1;
    }
    table
};

// --- AVX2 DETECTION ---
static USE_AVX2: OnceLock<bool> = OnceLock::new();

#[inline(always)]
fn use_avx2() -> bool {
    *USE_AVX2.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        { is_x86_feature_detected!("avx2") }
        #[cfg(not(target_arch = "x86_64"))]
        { false }
    })
}

#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
pub struct Accumulator {
    pub v: [i16; L1_SIZE],
    pub magic: u16, // Safety check for initialization
}

impl Accumulator {
    pub fn default() -> Self {
        // Initialize with biases if network is loaded, else 0
        let mut acc = Accumulator {
            v: [0; L1_SIZE],
            magic: 0,
        };
        if let Some(net) = NETWORK.get() {
            acc.v.copy_from_slice(&net.l0_biases);
            acc.magic = ACC_MAGIC;
        }
        acc
    }

    #[inline(always)]
    pub fn refresh(
        &mut self,
        bitboards: &[crate::bitboard::Bitboard; 12],
        perspective: usize,
        king_sq: usize,
    ) {
        if let Some(net) = NETWORK.get() {
            // Start with biases
            self.v.copy_from_slice(&net.l0_biases);
            self.magic = ACC_MAGIC;

            let king_bucket = get_king_bucket(perspective, king_sq);

            // Add all features
            // Iterate over all pieces
            for &color in &[crate::state::WHITE, crate::state::BLACK] {
                let start_pc = if color == crate::state::WHITE { 0 } else { 6 };
                let end_pc = start_pc + 6;

                for piece in start_pc..end_pc {
                    let mut bb = bitboards[piece];
                    while bb.0 != 0 {
                        let sq = bb.pop_lsb() as usize;

                        let idx = make_index(perspective, piece, sq, king_bucket);
                        self.add_feature(idx, net);
                    }
                }
            }
        } else {
            self.magic = 0;
        }
    }

    #[inline(always)]
    pub fn update(
        &mut self,
        added: &[(usize, usize)],
        removed: &[(usize, usize)],
        perspective: usize,
        king_sq: usize,
    ) {
        if let Some(net) = NETWORK.get() {
            let king_bucket = get_king_bucket(perspective, king_sq);

            for &(piece, sq) in removed {
                let idx = make_index(perspective, piece, sq, king_bucket);
                self.sub_feature(idx, net);
            }
            for &(piece, sq) in added {
                let idx = make_index(perspective, piece, sq, king_bucket);
                self.add_feature(idx, net);
            }
        }
    }

    // Helper to add a feature's weights
    #[inline(always)]
    fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        let offset = feature_idx * L1_SIZE;
        let weights = &net.l0_weights[offset..offset + L1_SIZE];

        #[cfg(target_arch = "x86_64")]
        if use_avx2() {
            unsafe {
                 let mut i = 0;
                 while i < L1_SIZE {
                     let v_ptr = self.v.as_mut_ptr().add(i);
                     let w_ptr = weights.as_ptr().add(i);

                     let v_vec = _mm256_load_si256(v_ptr as *const __m256i); // Aligned load for accumulator
                     let w_vec = _mm256_loadu_si256(w_ptr as *const __m256i); // Unaligned load for weights

                     let res = _mm256_add_epi16(v_vec, w_vec);
                     _mm256_store_si256(v_ptr as *mut __m256i, res); // Aligned store for accumulator

                     i += 16;
                 }
            }
            return;
        }

        for i in 0..L1_SIZE {
            self.v[i] = self.v[i].wrapping_add(weights[i]);
        }
    }

    #[inline(always)]
    fn sub_feature(&mut self, feature_idx: usize, net: &Network) {
        let offset = feature_idx * L1_SIZE;
        let weights = &net.l0_weights[offset..offset + L1_SIZE];

        #[cfg(target_arch = "x86_64")]
        if use_avx2() {
            unsafe {
                 let mut i = 0;
                 while i < L1_SIZE {
                     let v_ptr = self.v.as_mut_ptr().add(i);
                     let w_ptr = weights.as_ptr().add(i);

                     let v_vec = _mm256_load_si256(v_ptr as *const __m256i); // Aligned load
                     let w_vec = _mm256_loadu_si256(w_ptr as *const __m256i); // Unaligned load for weights

                     let res = _mm256_sub_epi16(v_vec, w_vec);
                     _mm256_store_si256(v_ptr as *mut __m256i, res); // Aligned store

                     i += 16;
                 }
            }
            return;
        }

        for i in 0..L1_SIZE {
            self.v[i] = self.v[i].wrapping_sub(weights[i]);
        }
    }
}

// --------------------------------------------------------
// Feature Indexer (Chess768 + King Buckets)
// --------------------------------------------------------

pub fn get_king_bucket(perspective: usize, king_sq: usize) -> usize {
    // Relative square based on perspective
    let rel_sq = if perspective == crate::state::WHITE {
        king_sq
    } else {
        king_sq ^ 56
    };

    // Standard Mirrored (32 Buckets)
    // Rank 0..7
    // File 0..3 (A-D). If E-H, mirror to D-A.
    let rank = rel_sq / 8;
    let file = rel_sq % 8;
    let file_folded = if file > 3 { 7 - file } else { file };

    // Index: Rank * 4 + File
    // 0..31
    rank * 4 + file_folded
}

pub fn make_index(perspective: usize, piece: usize, sq: usize, king_bucket: usize) -> usize {
    // piece: 0..11 (P, N, B, R, Q, K, p, n, b, r, q, k)
    // sq: 0..63
    // Perspective: WHITE=0, BLACK=1

    let piece_color = if piece < 6 {
        crate::state::WHITE
    } else {
        crate::state::BLACK
    };
    let piece_type = piece % 6; // 0..5 (P..K)

    // Relative Square
    let orient_sq = if perspective == crate::state::WHITE {
        sq
    } else {
        sq ^ 56
    };

    // Feature Offset
    // If piece_color == perspective -> Friendly (0..383)
    // If piece_color != perspective -> Enemy (384..767)
    let context_offset = if piece_color == perspective { 0 } else { 384 };

    // Index = (Bucket * 768) + Context + PieceType * 64 + Square
    (king_bucket * 768) + context_offset + piece_type * 64 + orient_sq
}

// --------------------------------------------------------
// SIMD Helpers
// --------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_calc_avx2(val: __m256i) -> __m256i {
    // Input: i16 values, clamped 0..255.
    // Goal: Output (val * val) / 255 using integer math.
    // Algorithm: y = x*x + 128; y = y + (y >> 8); res = y >> 8;

    let zero = _mm256_setzero_si256();
    let c128 = _mm256_set1_epi32(128);

    // Unpack to i32 (0..255 -> 0..255)
    let lo_lane = _mm256_castsi256_si128(val);
    let lo_epi32 = _mm256_cvtepu16_epi32(lo_lane);

    let hi_lane = _mm256_extracti128_si256::<1>(val);
    let hi_epi32 = _mm256_cvtepu16_epi32(hi_lane);

    // Square (x * x)
    let lo_sq = _mm256_mullo_epi32(lo_epi32, lo_epi32);
    let hi_sq = _mm256_mullo_epi32(hi_epi32, hi_epi32);

    // y = sq + 128
    let lo_y = _mm256_add_epi32(lo_sq, c128);
    let hi_y = _mm256_add_epi32(hi_sq, c128);

    // y = y + (y >> 8)
    let lo_y_shr = _mm256_srli_epi32(lo_y, 8);
    let hi_y_shr = _mm256_srli_epi32(hi_y, 8);
    let lo_y2 = _mm256_add_epi32(lo_y, lo_y_shr);
    let hi_y2 = _mm256_add_epi32(hi_y, hi_y_shr);

    // res = y >> 8
    let lo_res = _mm256_srli_epi32(lo_y2, 8);
    let hi_res = _mm256_srli_epi32(hi_y2, 8);

    // Pack back to i16
    let packed = _mm256_packus_epi32(lo_res, hi_res);
    _mm256_permute4x64_epi64::<0b11_01_10_00>(packed)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn layer_affine_avx2(input: &[i16], weights: &[i16], biases: &[i16], output: &mut [i16], in_size: usize, out_size: usize) {
    // Quantization shift
    // output = (dot + bias) >> OUTPUT_SHIFT

    for i in 0..out_size {
        let mut sum_vec = _mm256_setzero_si256();
        let w_row_offset = i * in_size;

        for j in (0..in_size).step_by(16) {
            let in_ptr = input.as_ptr().add(j);
            let w_ptr = weights.as_ptr().add(w_row_offset + j);

            let v_in = _mm256_loadu_si256(in_ptr as *const __m256i);
            let v_w = _mm256_loadu_si256(w_ptr as *const __m256i);

            let prod = _mm256_madd_epi16(v_in, v_w);
            sum_vec = _mm256_add_epi32(sum_vec, prod);
        }

        let v_lo = _mm256_castsi256_si128(sum_vec);
        let v_hi = _mm256_extracti128_si256::<1>(sum_vec);
        let v_sum = _mm_add_epi32(v_lo, v_hi);
        let v_sum2 = _mm_hadd_epi32(v_sum, v_sum);
        let v_sum3 = _mm_hadd_epi32(v_sum2, v_sum2);
        let sum_part = _mm_cvtsi128_si32(v_sum3);

        let total_sum = sum_part + biases[i] as i32;
        // Shift right
        output[i] = (total_sum >> OUTPUT_SHIFT) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn clamp_activations_avx2(buffer: &mut [i16]) {
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi16(Q_ACTIVATION as i16);

    // We assume buffer length is a multiple of 16 for AVX2 efficiency in previous layers.
    // L2 (32), L3 (32), L4 (32) are all multiples of 16.
    for i in (0..buffer.len()).step_by(16) {
         let ptr = buffer.as_mut_ptr().add(i);
         let v = _mm256_loadu_si256(ptr as *const __m256i);
         let res = _mm256_min_epi16(_mm256_max_epi16(v, zero), max);
         _mm256_storeu_si256(ptr as *mut __m256i, res);
    }
}

// --------------------------------------------------------
// Evaluation Logic (Duplicated for Dispatch)
// --------------------------------------------------------

pub fn evaluate(stm_acc: &mut Accumulator, ntm_acc: &mut Accumulator, scratch: &mut NNUEScratch, state: &crate::state::GameState) -> i32 {
    let net = match NETWORK.get() {
        Some(n) => n,
        None => return 0,
    };

    if cfg!(debug_assertions) {
        if stm_acc.magic != ACC_MAGIC || ntm_acc.magic != ACC_MAGIC {
            panic!("CRITICAL ERROR: NNUE is loaded but Accumulators are invalid...");
        }
    } else {
         if stm_acc.magic != ACC_MAGIC || ntm_acc.magic != ACC_MAGIC {
             use crate::state::{K, k};
             stm_acc.refresh(&state.bitboards, state.side_to_move, state.bitboards[if state.side_to_move == crate::state::WHITE { K } else { k }].get_lsb_index() as usize);
             ntm_acc.refresh(&state.bitboards, 1 - state.side_to_move, state.bitboards[if (1 - state.side_to_move) == crate::state::WHITE { K } else { k }].get_lsb_index() as usize);
         }
    }

    // Dispatch
    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        return unsafe { evaluate_avx2(stm_acc, ntm_acc, net, scratch) };
    }

    evaluate_scalar(stm_acc, ntm_acc, net, scratch)
}

#[cfg(debug_assertions)]
pub fn verify_accumulator(state: &crate::state::GameState, acc: &Accumulator, perspective: usize) {
    if NETWORK.get().is_none() {
        return;
    }

    let mut fresh = Accumulator::default();
    use crate::state::{K, k};
    let king_sq = if perspective == crate::state::WHITE {
        state.bitboards[K].get_lsb_index() as usize
    } else {
        state.bitboards[k].get_lsb_index() as usize
    };
    fresh.refresh(&state.bitboards, perspective, king_sq);

    for i in 0..L1_SIZE {
        if fresh.v[i] != acc.v[i] {
            eprintln!("ACCUMULATOR MISMATCH Perspective: {}", if perspective == crate::state::WHITE { "White" } else { "Black" });
            eprintln!("Index: {}, Fresh: {}, Incremental: {}", i, fresh.v[i], acc.v[i]);
            eprintln!("FEN: {}", state.to_fen());
            panic!("Accumulator Verification Failed");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn evaluate_avx2(stm_acc: &Accumulator, ntm_acc: &Accumulator, net: &Network, scratch: &mut NNUEScratch) -> i32 {
    // L1 Input generation (AVX2)
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);

    // Dual-perspective funnel: SCReLU(STM) - SCReLU(NTM) (must match trainer)
    for i in (0..L1_SIZE).step_by(16) {
        let stm_ptr = stm_acc.v.as_ptr().add(i);
        let ntm_ptr = ntm_acc.v.as_ptr().add(i);

        let stm_val = _mm256_load_si256(stm_ptr as *const __m256i); // Aligned load
        let ntm_val = _mm256_load_si256(ntm_ptr as *const __m256i); // Aligned load

        let stm_clamped = _mm256_min_epi16(_mm256_max_epi16(stm_val, zero), qa);
        let ntm_clamped = _mm256_min_epi16(_mm256_max_epi16(ntm_val, zero), qa);

        let stm_act = screlu_calc_avx2(stm_clamped);
        let ntm_act = screlu_calc_avx2(ntm_clamped);

        let res = _mm256_sub_epi16(stm_act, ntm_act);
        _mm256_storeu_si256(scratch.hidden_l1.as_mut_ptr().add(i) as *mut __m256i, res);
    }


    // Layer 1: 256 -> 32
    // Note: We use only the first 256 elements of scratch.hidden_l1
    layer_affine_avx2(&scratch.hidden_l1, &net.l1_weights, &net.l1_biases, &mut scratch.l2_out, L1_SIZE, L2_SIZE);
    clamp_activations_avx2(&mut scratch.l2_out);

#[cfg(not(feature = "nnue_512_64"))]
{
    // Layer 2: 32 -> 32
    layer_affine_avx2(&scratch.l2_out, &net.l2_weights, &net.l2_biases, &mut scratch.l3_out, L2_SIZE, L3_SIZE);
    clamp_activations_avx2(&mut scratch.l3_out);

    // Layer 3: 32 -> 32
    layer_affine_avx2(&scratch.l3_out, &net.l3_weights, &net.l3_biases, &mut scratch.l4_out, L3_SIZE, L4_SIZE);
    clamp_activations_avx2(&mut scratch.l4_out);

    // Output Layer: 32 -> 1
    layer_affine_avx2(&scratch.l4_out, &net.l4_weights, &net.l4_biases, &mut scratch.final_out, L4_SIZE, 1);
}

#[cfg(feature = "nnue_512_64")]
{
    // Output Layer: 64 -> 1
    layer_affine_avx2(&scratch.l2_out, &net.l2_weights, &net.l2_biases, &mut scratch.final_out, L2_SIZE, 1);
}

    let output = scratch.final_out[0] as i32;
    (output * SCALE) / (QB * Q_ACTIVATION)
}

fn evaluate_scalar(stm_acc: &Accumulator, ntm_acc: &Accumulator, net: &Network, scratch: &mut NNUEScratch) -> i32 {
    // L1 (Scalar) - STM Only
    evaluate_l1_scalar(stm_acc, ntm_acc, &mut scratch.hidden_l1);

    // Layer 1: 256 -> 32
    layer_affine_scalar(&scratch.hidden_l1, &net.l1_weights, &net.l1_biases, &mut scratch.l2_out, L1_SIZE, L2_SIZE);
    clamp_activations_scalar(&mut scratch.l2_out);

#[cfg(not(feature = "nnue_512_64"))]
{
    // Layer 2: 32 -> 32
    layer_affine_scalar(&scratch.l2_out, &net.l2_weights, &net.l2_biases, &mut scratch.l3_out, L2_SIZE, L3_SIZE);
    clamp_activations_scalar(&mut scratch.l3_out);

    // Layer 3: 32 -> 32
    layer_affine_scalar(&scratch.l3_out, &net.l3_weights, &net.l3_biases, &mut scratch.l4_out, L3_SIZE, L4_SIZE);
    clamp_activations_scalar(&mut scratch.l4_out);

    // Output Layer: 32 -> 1
    layer_affine_scalar(&scratch.l4_out, &net.l4_weights, &net.l4_biases, &mut scratch.final_out, L4_SIZE, 1);
}

#[cfg(feature = "nnue_512_64")]
{
    // Output Layer: 64 -> 1
    layer_affine_scalar(&scratch.l2_out, &net.l2_weights, &net.l2_biases, &mut scratch.final_out, L2_SIZE, 1);
}

    let output = scratch.final_out[0] as i32;
    (output * SCALE) / (QB * Q_ACTIVATION)
}

fn evaluate_l1_scalar(stm: &Accumulator, ntm: &Accumulator, output: &mut [i16]) {
    // Dual-perspective funnel: SCReLU(STM) - SCReLU(NTM) (must match trainer)
    for i in 0..L1_SIZE {
        let stm_val = stm.v[i].clamp(0, 255) as usize;
        let ntm_val = ntm.v[i].clamp(0, 255) as usize;
        output[i] = SCRELU[stm_val].wrapping_sub(SCRELU[ntm_val]);
    }
}

fn clamp_activations_scalar(buffer: &mut [i16]) {
    for x in buffer.iter_mut() {
        *x = (*x).clamp(0, Q_ACTIVATION as i16);
    }
}

fn layer_affine_scalar(input: &[i16], weights: &[i16], biases: &[i16], output: &mut [i16], in_size: usize, out_size: usize) {
    for i in 0..out_size {
        let mut sum: i32 = biases[i] as i32;
        for j in 0..in_size {
            sum += (input[j] as i32) * (weights[i * in_size + j] as i32);
        }
        // Shift Right
        output[i] = (sum >> OUTPUT_SHIFT) as i16;
    }
}

// --------------------------------------------------------
// Network & Loading
// --------------------------------------------------------

pub struct Network {
    // Layer 0 (768*32 -> 256)
    pub l0_weights: Vec<i16>, // (768*32) * 256
    pub l0_biases: Vec<i16>,  // 256

    // Layer 1 (256 -> 32)
    pub l1_weights: Vec<i16>, // 256 * 32
    pub l1_biases: Vec<i16>,  // 32

    // Layer 2
    // default: 32 -> 32 (ClippedReLU)
    // nnue_512_64: 64 -> 1 (Raw output)
    pub l2_weights: Vec<i16>,
    pub l2_biases: Vec<i16>,

    #[cfg(not(feature = "nnue_512_64"))]
    pub l3_weights: Vec<i16>,
    #[cfg(not(feature = "nnue_512_64"))]
    pub l3_biases: Vec<i16>,

    #[cfg(not(feature = "nnue_512_64"))]
    pub l4_weights: Vec<i16>,
    #[cfg(not(feature = "nnue_512_64"))]
    pub l4_biases: Vec<i16>,
}

pub fn load_network_from_reader<R: Read>(reader: &mut R) -> io::Result<Network> {
    // Magic Check
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    let magic = u32::from_le_bytes(magic_bytes);
    if magic != NETWORK_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic number - architecture mismatch"));
    }

    // Helper to read vector
    let read_vec = |reader: &mut R, len: usize| -> io::Result<Vec<i16>> {
        let mut v = vec![0i16; len];
        let mut buf = vec![0u8; len * 2];
        reader.read_exact(&mut buf)?;
        for (i, chunk) in buf.chunks(2).enumerate() {
            v[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
        }
        Ok(v)
    };

    let total_features = INPUT_SIZE * NUM_BUCKETS; // 768 * 32

    // L0 (Acc)
    let l0_weights = read_vec(reader, total_features * L1_SIZE)?;
    let l0_biases = read_vec(reader, L1_SIZE)?;

    // L1 (Dense 256->32)
    let l1_weights = read_vec(reader, L1_SIZE * L2_SIZE)?;
    let l1_biases = read_vec(reader, L2_SIZE)?;

    // L2 (Dense 32->32)
    let l2_weights = read_vec(reader, L2_SIZE * L3_SIZE)?;
    let l2_biases = read_vec(reader, L3_SIZE)?;

    // L3 (Dense 32->32)
    let l3_weights = read_vec(reader, L3_SIZE * L4_SIZE)?;
    let l3_biases = read_vec(reader, L4_SIZE)?;

    // L4 (Dense 32->1)
    let l4_weights = read_vec(reader, L4_SIZE * OUTPUT_SIZE)?;
    let l4_biases = read_vec(reader, OUTPUT_SIZE)?;

    Ok(Network {
        l0_weights,
        l0_biases,
        l1_weights,
        l1_biases,
        l2_weights,
        l2_biases,
        l3_weights,
        l3_biases,
        l4_weights,
        l4_biases,
    })
}

pub fn load_network(path: &str) -> io::Result<Network> {
    let mut file = File::open(path)?;
    load_network_from_reader(&mut file)
}

pub fn init_nnue(path: &str) {
    match load_network(path) {
        Ok(net) => {
            NETWORK.set(net).ok();
            println!("NNUE loaded successfully from {}", path);
        }
        Err(e) => {
            println!(
                "Failed to load NNUE from {}: {}. Trying embedded network...",
                path, e
            );
            match load_network_from_reader(&mut Cursor::new(EMBEDDED_NET)) {
                Ok(net) => {
                    NETWORK.set(net).ok();
                    println!("NNUE loaded successfully from embedded data");
                }
                Err(e2) => {
                    println!("Failed to load embedded NNUE: {}", e2);
                }
            }
        }
    }
}

pub fn ensure_nnue_loaded() {
    if NETWORK.get().is_some() {
        return;
    }
    // Try default path first
    if std::path::Path::new("nn-aether.nnue").exists() {
        init_nnue("nn-aether.nnue");
    } else {
        // Fallback to embedded
        match load_network_from_reader(&mut Cursor::new(EMBEDDED_NET)) {
            Ok(net) => {
                NETWORK.set(net).ok();
                println!("NNUE loaded successfully from embedded data");
            }
            Err(e) => {
                println!("Failed to load embedded NNUE: {}", e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screlu_avx2_vs_scalar() {
        if !use_avx2() {
            println!("Skipping AVX2 test on non-AVX2 machine");
            return;
        }

        // Test all inputs 0..255
        let mut inputs = [0i16; 256];
        for i in 0..256 {
            inputs[i] = i as i16;
        }

        unsafe {
            // Process in chunks of 16
            for i in (0..256).step_by(16) {
                let ptr = inputs.as_ptr().add(i);
                let val = _mm256_loadu_si256(ptr as *const __m256i);
                let res_vec = screlu_calc_avx2(val);

                let mut res_arr = [0i16; 16];
                _mm256_storeu_si256(res_arr.as_mut_ptr() as *mut __m256i, res_vec);

                for j in 0..16 {
                    let input = inputs[i + j];
                    let expected = SCRELU[input as usize];
                    let actual = res_arr[j];
                    assert_eq!(actual, expected, "Mismatch at input {}", input);
                }
            }
        }
    }

    #[test]
    fn test_determinism() {
        use crate::state::GameState;

        // Only run if we can load a network (requires valid magic)
        // Since we probably don't have one, we just check if it returns gracefully or panics.
        // We can mock a network here just for this test!

        let mock_network = Network {
            l0_weights: vec![0; INPUT_SIZE * NUM_BUCKETS * L1_SIZE],
            l0_biases: vec![10; L1_SIZE], // Use bias 10
            l1_weights: vec![1; L1_SIZE * L2_SIZE],
            l1_biases: vec![1; L2_SIZE],
            l2_weights: vec![1; L2_SIZE * L3_SIZE],
            l2_biases: vec![1; L3_SIZE],
            l3_weights: vec![1; L3_SIZE * L4_SIZE],
            l3_biases: vec![1; L4_SIZE],
            l4_weights: vec![1; L4_SIZE * OUTPUT_SIZE],
            l4_biases: vec![1; OUTPUT_SIZE],
        };

        // We can't set the global OnceLock easily if it's already set or not.
        // But we can call evaluate functions directly if we pass the network.
        // Oops, evaluate functions take `net` as argument in their impl but the public `evaluate` retrieves it from global.

        // However, `evaluate_scalar` and `evaluate_avx2` take `net`.
        // So we can test `evaluate_scalar` directly.

        let mut stm = Accumulator { v: [0; L1_SIZE], magic: ACC_MAGIC };
        stm.v.copy_from_slice(&mock_network.l0_biases);

        let mut ntm = Accumulator { v: [0; L1_SIZE], magic: ACC_MAGIC };
        ntm.v.copy_from_slice(&mock_network.l0_biases);

        let mut scratch = NNUEScratch::default();

        // Run twice
        let eval1 = evaluate_scalar(&mut stm, &mut ntm, &mock_network, &mut scratch);
        let eval2 = evaluate_scalar(&mut stm, &mut ntm, &mock_network, &mut scratch);

        assert_eq!(eval1, eval2, "Scalar eval determinism failed");

        if use_avx2() {
            unsafe {
                let eval_avx = evaluate_avx2(&mut stm, &mut ntm, &mock_network, &mut scratch);
                assert_eq!(eval1, eval_avx, "AVX2 vs Scalar mismatch");
            }
        }
    }
}
