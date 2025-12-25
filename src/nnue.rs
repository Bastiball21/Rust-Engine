// src/nnue.rs
use std::fs::File;
use std::io::{self, Read};
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Architecture Constants
pub const L1_SIZE: usize = 256;
pub const L2_SIZE: usize = 32;
pub const L3_SIZE: usize = 32;
pub const L4_SIZE: usize = 32;
pub const OUTPUT_SIZE: usize = 1;

pub const INPUT_SIZE: usize = 768;
pub const NUM_BUCKETS: usize = 32; // Matches bullet_lib ChessBuckets (Standard Mirrored)

// Quantization Constants
const QA: i32 = 255;
const QB: i32 = 64;
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
        table[i] = ((i as i32 * i as i32) / 255) as i16;
        i += 1;
    }
    table
};

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
                        let sq = bb.get_lsb_index() as usize;
                        bb.pop_bit(sq as u8);

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
        unsafe {
            let mut i = 0;
            while i < L1_SIZE {
                let v_ptr = self.v.as_mut_ptr().add(i);
                let w_ptr = weights.as_ptr().add(i);

                let v_vec = _mm256_loadu_si256(v_ptr as *const __m256i); // Unaligned load safe
                let w_vec = _mm256_loadu_si256(w_ptr as *const __m256i);

                let res = _mm256_add_epi16(v_vec, w_vec);
                _mm256_storeu_si256(v_ptr as *mut __m256i, res); // Unaligned store safe

                i += 16;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        for i in 0..L1_SIZE {
            self.v[i] = self.v[i].wrapping_add(weights[i]);
        }
    }

    #[inline(always)]
    fn sub_feature(&mut self, feature_idx: usize, net: &Network) {
        let offset = feature_idx * L1_SIZE;
        let weights = &net.l0_weights[offset..offset + L1_SIZE];

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut i = 0;
            while i < L1_SIZE {
                let v_ptr = self.v.as_mut_ptr().add(i);
                let w_ptr = weights.as_ptr().add(i);

                let v_vec = _mm256_loadu_si256(v_ptr as *const __m256i); // Unaligned load safe
                let w_vec = _mm256_loadu_si256(w_ptr as *const __m256i);

                let res = _mm256_sub_epi16(v_vec, w_vec);
                _mm256_storeu_si256(v_ptr as *mut __m256i, res); // Unaligned store safe

                i += 16;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
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
unsafe fn screlu_calc_avx2(val: __m256i, scale: __m256) -> __m256i {
    let lo_lane = _mm256_castsi256_si128(val);
    let lo_epi32 = _mm256_cvtepu16_epi32(lo_lane);
    let lo_ps = _mm256_cvtepi32_ps(lo_epi32);
    let lo_sq = _mm256_mul_ps(lo_ps, lo_ps);
    let lo_res = _mm256_mul_ps(lo_sq, scale);
    let lo_int = _mm256_cvttps_epi32(lo_res);

    let hi_lane = _mm256_extracti128_si256::<1>(val);
    let hi_epi32 = _mm256_cvtepu16_epi32(hi_lane);
    let hi_ps = _mm256_cvtepi32_ps(hi_epi32);
    let hi_sq = _mm256_mul_ps(hi_ps, hi_ps);
    let hi_res = _mm256_mul_ps(hi_sq, scale);
    let hi_int = _mm256_cvttps_epi32(hi_res);

    let packed = _mm256_packus_epi32(lo_int, hi_int);
    _mm256_permute4x64_epi64::<0b11_01_10_00>(packed)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn layer_affine_avx2(input: &[i16], weights: &[i16], biases: &[i16], output: &mut [i16], in_size: usize, out_size: usize) {
    let qb = QB as i32;

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
        output[i] = (total_sum / qb) as i16;
    }
}

// --------------------------------------------------------
// Evaluation
// --------------------------------------------------------

pub fn evaluate(stm_acc: &Accumulator, ntm_acc: &Accumulator) -> i32 {
    if let Some(net) = NETWORK.get() {
        if stm_acc.magic != ACC_MAGIC || ntm_acc.magic != ACC_MAGIC {
            panic!("CRITICAL ERROR: NNUE is loaded but Accumulators are invalid...");
        }

        let mut hidden_512 = [0i16; 512];

        // L1 Input generation
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            unsafe {
                let zero = _mm256_setzero_si256();
                let qa = _mm256_set1_epi16(QA as i16);
                let scale = _mm256_set1_ps(1.0 / 255.0);

                // STM
                for i in (0..L1_SIZE).step_by(16) {
                    let ptr = stm_acc.v.as_ptr().add(i);
                    let val = _mm256_loadu_si256(ptr as *const __m256i);
                    let clamped = _mm256_min_epi16(_mm256_max_epi16(val, zero), qa);
                    let res = screlu_calc_avx2(clamped, scale);
                    _mm256_storeu_si256(hidden_512.as_mut_ptr().add(i) as *mut __m256i, res);
                }
                // NTM
                for i in (0..L1_SIZE).step_by(16) {
                    let ptr = ntm_acc.v.as_ptr().add(i);
                    let val = _mm256_loadu_si256(ptr as *const __m256i);
                    let clamped = _mm256_min_epi16(_mm256_max_epi16(val, zero), qa);
                    let res = screlu_calc_avx2(clamped, scale);
                    _mm256_storeu_si256(hidden_512.as_mut_ptr().add(L1_SIZE + i) as *mut __m256i, res);
                }
            }
        } else {
             evaluate_l1_scalar(stm_acc, ntm_acc, &mut hidden_512);
        }

        #[cfg(not(target_arch = "x86_64"))]
        evaluate_l1_scalar(stm_acc, ntm_acc, &mut hidden_512);

        // Layer 1: 512 -> 32
        let mut l2_out = [0i16; L2_SIZE];
        affine_layer_wrapper(&hidden_512, &net.l1_weights, &net.l1_biases, &mut l2_out, 512, L2_SIZE);
        clamp_activations_wrapper(&mut l2_out);

        // Layer 2: 32 -> 32
        let mut l3_out = [0i16; L3_SIZE];
        affine_layer_wrapper(&l2_out, &net.l2_weights, &net.l2_biases, &mut l3_out, L2_SIZE, L3_SIZE);
        clamp_activations_wrapper(&mut l3_out);

        // Layer 3: 32 -> 32
        let mut l4_out = [0i16; L4_SIZE];
        affine_layer_wrapper(&l3_out, &net.l3_weights, &net.l3_biases, &mut l4_out, L3_SIZE, L4_SIZE);
        clamp_activations_wrapper(&mut l4_out);

        // Output Layer: 32 -> 1
        let mut final_out = [0i16; 1];
        affine_layer_wrapper(&l4_out, &net.l4_weights, &net.l4_biases, &mut final_out, L4_SIZE, 1);

        let output = final_out[0] as i32;
        (output * SCALE) / (QB * Q_ACTIVATION)
    } else {
        0
    }
}

fn evaluate_l1_scalar(stm: &Accumulator, ntm: &Accumulator, output: &mut [i16]) {
    for i in 0..L1_SIZE {
        let val = stm.v[i].clamp(0, 255);
        output[i] = SCRELU[val as usize];
    }
    for i in 0..L1_SIZE {
        let val = ntm.v[i].clamp(0, 255);
        output[L1_SIZE + i] = SCRELU[val as usize];
    }
}

fn affine_layer_wrapper(input: &[i16], weights: &[i16], biases: &[i16], output: &mut [i16], in_size: usize, out_size: usize) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { layer_affine_avx2(input, weights, biases, output, in_size, out_size); }
        return;
    }
    layer_affine_scalar(input, weights, biases, output, in_size, out_size);
}

fn clamp_activations_wrapper(buffer: &mut [i16]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe {
            let zero = _mm256_setzero_si256();
            let max = _mm256_set1_epi16(Q_ACTIVATION as i16);
            for i in (0..buffer.len()).step_by(16) {
                 let ptr = buffer.as_mut_ptr().add(i);
                 let v = _mm256_loadu_si256(ptr as *const __m256i);
                 let res = _mm256_min_epi16(_mm256_max_epi16(v, zero), max);
                 _mm256_storeu_si256(ptr as *mut __m256i, res);
            }
        }
        return;
    }
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
        // Normalize: Divide by 64 (QB)
        output[i] = (sum / QB) as i16;
    }
}

// --------------------------------------------------------
// Network & Loading
// --------------------------------------------------------

pub struct Network {
    // Layer 0 (768*32 -> 256)
    pub l0_weights: Vec<i16>, // (768*32) * 256
    pub l0_biases: Vec<i16>,  // 256

    // Layer 1 (512 -> 32)
    pub l1_weights: Vec<i16>, // 512 * 32
    pub l1_biases: Vec<i16>,  // 32

    // Layer 2 (32 -> 32)
    pub l2_weights: Vec<i16>, // 32 * 32
    pub l2_biases: Vec<i16>,  // 32

    // Layer 3 (32 -> 32)
    pub l3_weights: Vec<i16>, // 32 * 32
    pub l3_biases: Vec<i16>,  // 32

    // Layer 4 (32 -> 1)
    pub l4_weights: Vec<i16>, // 32 * 1
    pub l4_biases: Vec<i16>,  // 1
}

pub fn load_network(path: &str) -> io::Result<Network> {
    let mut file = File::open(path)?;

    // Helper to read vector
    let read_vec = |f: &mut File, len: usize| -> io::Result<Vec<i16>> {
        let mut v = vec![0i16; len];
        let mut buf = vec![0u8; len * 2];
        f.read_exact(&mut buf)?;
        for (i, chunk) in buf.chunks(2).enumerate() {
            v[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
        }
        Ok(v)
    };

    let total_features = INPUT_SIZE * NUM_BUCKETS; // 768 * 32

    // L0
    let l0_weights = read_vec(&mut file, total_features * L1_SIZE)?;
    let l0_biases = read_vec(&mut file, L1_SIZE)?;

    // L1
    let l1_weights = read_vec(&mut file, (2 * L1_SIZE) * L2_SIZE)?;
    let l1_biases = read_vec(&mut file, L2_SIZE)?;

    // L2
    let l2_weights = read_vec(&mut file, L2_SIZE * L3_SIZE)?;
    let l2_biases = read_vec(&mut file, L3_SIZE)?;

    // L3
    let l3_weights = read_vec(&mut file, L3_SIZE * L4_SIZE)?;
    let l3_biases = read_vec(&mut file, L4_SIZE)?;

    // L4
    let l4_weights = read_vec(&mut file, L4_SIZE * OUTPUT_SIZE)?;
    let l4_biases = read_vec(&mut file, OUTPUT_SIZE)?;

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

pub fn init_nnue(path: &str) {
    match load_network(path) {
        Ok(net) => {
            NETWORK.set(net).ok();
            println!("NNUE loaded successfully from {}", path);
        }
        Err(e) => {
            println!("Failed to load NNUE from {}: {}", path, e);
        }
    }
}
