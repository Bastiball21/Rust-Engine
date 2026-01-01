// src/nnue.rs
use std::fs::File;
use std::io::{self, Cursor, Read};
use std::sync::OnceLock;
use crate::nnue_scratch::NNUEScratch;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

static EMBEDDED_NET: &[u8] = include_bytes!("../new-net.bin");

// Architecture Constants
pub const L1_SIZE: usize = 512;
pub const L2_SIZE: usize = 64;
// Unused path (kept non-zero for scratch struct layout).
pub const L3_SIZE: usize = 1;
pub const L4_SIZE: usize = 1;
pub const OUTPUT_SIZE: usize = 1;

pub const INPUT_SIZE: usize = 768;
pub const NUM_BUCKETS: usize = 32;

pub const NETWORK_MAGIC: u32 = 0xAE74E202;

// Quantization Constants
const QA: i32 = 255;
const QB: i32 = 64;
const OUTPUT_SHIFT: i32 = 6;
const Q_ACTIVATION: i32 = 127;
const SCALE: i32 = 400;
const ACC_MAGIC: u16 = 0x1234;

pub static NETWORK: OnceLock<Network> = OnceLock::new();

const SCRELU: [i16; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let sq = i as i32 * i as i32;
        let y = sq + 128;
        let y2 = y + (y >> 8);
        table[i] = (y2 >> 8) as i16;
        i += 1;
    }
    table
};

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
    pub magic: u16,
}

impl Accumulator {
    pub fn default() -> Self {
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
            self.v.copy_from_slice(&net.l0_biases);
            self.magic = ACC_MAGIC;

            let king_bucket = get_king_bucket(perspective, king_sq);
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
                     let v_vec = _mm256_load_si256(v_ptr as *const __m256i);
                     let w_vec = _mm256_loadu_si256(w_ptr as *const __m256i);
                     let res = _mm256_add_epi16(v_vec, w_vec);
                     _mm256_store_si256(v_ptr as *mut __m256i, res);
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
                     let v_vec = _mm256_load_si256(v_ptr as *const __m256i);
                     let w_vec = _mm256_loadu_si256(w_ptr as *const __m256i);
                     let res = _mm256_sub_epi16(v_vec, w_vec);
                     _mm256_store_si256(v_ptr as *mut __m256i, res);
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

pub fn get_king_bucket(perspective: usize, king_sq: usize) -> usize {
    let rel_sq = if perspective == crate::state::WHITE { king_sq } else { king_sq ^ 56 };
    let rank = rel_sq / 8;
    let file = rel_sq % 8;
    let file_folded = if file > 3 { 7 - file } else { file };
    rank * 4 + file_folded
}

pub fn make_index(perspective: usize, piece: usize, sq: usize, king_bucket: usize) -> usize {
    let piece_color = if piece < 6 { crate::state::WHITE } else { crate::state::BLACK };
    let piece_type = piece % 6;
    let orient_sq = if perspective == crate::state::WHITE { sq } else { sq ^ 56 };
    let context_offset = if piece_color == perspective { 0 } else { 384 };
    (king_bucket * 768) + context_offset + piece_type * 64 + orient_sq
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_calc_avx2(val: __m256i) -> __m256i {
    let c128 = _mm256_set1_epi32(128);
    let lo_lane = _mm256_castsi256_si128(val);
    let lo_epi32 = _mm256_cvtepu16_epi32(lo_lane);
    let hi_lane = _mm256_extracti128_si256::<1>(val);
    let hi_epi32 = _mm256_cvtepu16_epi32(hi_lane);
    let lo_sq = _mm256_mullo_epi32(lo_epi32, lo_epi32);
    let hi_sq = _mm256_mullo_epi32(hi_epi32, hi_epi32);
    let lo_y = _mm256_add_epi32(lo_sq, c128);
    let hi_y = _mm256_add_epi32(hi_sq, c128);
    let lo_y_shr = _mm256_srli_epi32(lo_y, 8);
    let hi_y_shr = _mm256_srli_epi32(hi_y, 8);
    let lo_y2 = _mm256_add_epi32(lo_y, lo_y_shr);
    let hi_y2 = _mm256_add_epi32(hi_y, hi_y_shr);
    let lo_res = _mm256_srli_epi32(lo_y2, 8);
    let hi_res = _mm256_srli_epi32(hi_y2, 8);
    let packed = _mm256_packus_epi32(lo_res, hi_res);
    _mm256_permute4x64_epi64::<0b11_01_10_00>(packed)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
// FIX: Changed biases to &[i32]
unsafe fn layer_affine_avx2(input: &[i16], weights: &[i16], biases: &[i32], output: &mut [i16], in_size: usize, out_size: usize) {
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

        // FIX: Removed cast, biases[i] is already i32
        let total_sum = sum_part + biases[i];
        output[i] = (total_sum >> OUTPUT_SHIFT) as i16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn clamp_activations_avx2(buffer: &mut [i16]) {
    let zero = _mm256_setzero_si256();
    let max = _mm256_set1_epi16(Q_ACTIVATION as i16);
    for i in (0..buffer.len()).step_by(16) {
         let ptr = buffer.as_mut_ptr().add(i);
         let v = _mm256_loadu_si256(ptr as *const __m256i);
         let res = _mm256_min_epi16(_mm256_max_epi16(v, zero), max);
         _mm256_storeu_si256(ptr as *mut __m256i, res);
    }
}

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

    #[cfg(target_arch = "x86_64")]
    if use_avx2() {
        return unsafe { evaluate_avx2(stm_acc, ntm_acc, net, scratch) };
    }
    evaluate_scalar(stm_acc, ntm_acc, net, scratch)
}

#[cfg(debug_assertions)]
pub fn verify_accumulator(state: &crate::state::GameState, acc: &Accumulator, perspective: usize) {
    if NETWORK.get().is_none() { return; }
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
            panic!("Accumulator Verification Failed");
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn evaluate_avx2(stm_acc: &Accumulator, ntm_acc: &Accumulator, net: &Network, scratch: &mut NNUEScratch) -> i32 {
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);

    for i in (0..L1_SIZE).step_by(16) {
        let stm_ptr = stm_acc.v.as_ptr().add(i);
        let ntm_ptr = ntm_acc.v.as_ptr().add(i);
        let stm_val = _mm256_load_si256(stm_ptr as *const __m256i);
        let ntm_val = _mm256_load_si256(ntm_ptr as *const __m256i);
        let stm_clamped = _mm256_min_epi16(_mm256_max_epi16(stm_val, zero), qa);
        let ntm_clamped = _mm256_min_epi16(_mm256_max_epi16(ntm_val, zero), qa);
        let stm_act = screlu_calc_avx2(stm_clamped);
        let ntm_act = screlu_calc_avx2(ntm_clamped);
        let res = _mm256_sub_epi16(stm_act, ntm_act);
        _mm256_storeu_si256(scratch.hidden_l1.as_mut_ptr().add(i) as *mut __m256i, res);
    }

    // FIX: Using i32 biases
    layer_affine_avx2(&scratch.hidden_l1, &net.l1_weights, &net.l1_biases, &mut scratch.l2_out, L1_SIZE, L2_SIZE);
    clamp_activations_avx2(&mut scratch.l2_out);

    // FIX: Using i32 biases
    layer_affine_avx2(&scratch.l2_out, &net.l2_weights, &net.l2_biases, &mut scratch.final_out, L2_SIZE, 1);

    let output = scratch.final_out[0] as i32;
    (output * SCALE) / (QB * Q_ACTIVATION)
}

fn evaluate_scalar(stm_acc: &Accumulator, ntm_acc: &Accumulator, net: &Network, scratch: &mut NNUEScratch) -> i32 {
    evaluate_l1_scalar(stm_acc, ntm_acc, &mut scratch.hidden_l1);
    // FIX: Using i32 biases
    layer_affine_scalar(&scratch.hidden_l1, &net.l1_weights, &net.l1_biases, &mut scratch.l2_out, L1_SIZE, L2_SIZE);
    clamp_activations_scalar(&mut scratch.l2_out);
    // FIX: Using i32 biases
    layer_affine_scalar(&scratch.l2_out, &net.l2_weights, &net.l2_biases, &mut scratch.final_out, L2_SIZE, 1);
    let output = scratch.final_out[0] as i32;
    (output * SCALE) / (QB * Q_ACTIVATION)
}

fn evaluate_l1_scalar(stm: &Accumulator, ntm: &Accumulator, output: &mut [i16]) {
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

// FIX: Changed biases to &[i32]
fn layer_affine_scalar(input: &[i16], weights: &[i16], biases: &[i32], output: &mut [i16], in_size: usize, out_size: usize) {
    for i in 0..out_size {
        // FIX: Removed cast, biases[i] is already i32
        let mut sum: i32 = biases[i];
        for j in 0..in_size {
            sum += (input[j] as i32) * (weights[i * in_size + j] as i32);
        }
        output[i] = (sum >> OUTPUT_SHIFT) as i16;
    }
}

pub struct Network {
    pub l0_weights: Vec<i16>,
    pub l0_biases: Vec<i16>, // Keep L0 as i16 for accumulator
    pub l1_weights: Vec<i16>,
    pub l1_biases: Vec<i32>, // FIX: Stored as i32
    pub l2_weights: Vec<i16>,
    pub l2_biases: Vec<i32>, // FIX: Stored as i32
}

pub fn load_network_from_reader<R: Read>(reader: &mut R) -> io::Result<Network> {
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    let magic = u32::from_le_bytes(magic_bytes);
    if magic != NETWORK_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic number"));
    }

    let read_vec = |reader: &mut R, len: usize| -> io::Result<Vec<i16>> {
        let mut v = vec![0i16; len];
        let mut buf = vec![0u8; len * 2];
        reader.read_exact(&mut buf)?;
        for (i, chunk) in buf.chunks(2).enumerate() {
            v[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
        }
        Ok(v)
    };

    // FIX: New helper to read i32
    let read_vec_i32 = |reader: &mut R, len: usize| -> io::Result<Vec<i32>> {
        let mut v = vec![0i32; len];
        let mut buf = vec![0u8; len * 4];
        reader.read_exact(&mut buf)?;
        for (i, chunk) in buf.chunks(4).enumerate() {
            v[i] = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(v)
    };

    let total_features = INPUT_SIZE * NUM_BUCKETS;

    let l0_weights = read_vec(reader, total_features * L1_SIZE)?;
    let l0_biases = read_vec(reader, L1_SIZE)?;

    let l1_weights = read_vec(reader, L1_SIZE * L2_SIZE)?;
    // FIX: Read L1 biases as i16, then convert to i32 for uniformity
    let l1_biases_i16 = read_vec(reader, L2_SIZE)?;
    let l1_biases = l1_biases_i16.into_iter().map(|x| x as i32).collect();

    let l2_weights = read_vec(reader, L2_SIZE * L3_SIZE)?;
    // FIX: Read L2 biases as i32 (Trainer writes i32 for output)
    let l2_biases = read_vec_i32(reader, L3_SIZE)?;

    Ok(Network {
        l0_weights,
        l0_biases,
        l1_weights,
        l1_biases,
        l2_weights,
        l2_biases,
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
            println!("Failed to load NNUE from {}: {}. Trying embedded...", path, e);
            match load_network_from_reader(&mut Cursor::new(EMBEDDED_NET)) {
                Ok(net) => {
                    NETWORK.set(net).ok();
                    println!("NNUE loaded successfully from embedded data");
                }
                Err(e2) => println!("Failed to load embedded NNUE: {}", e2),
            }
        }
    }
}

pub fn ensure_nnue_loaded() {
    if NETWORK.get().is_some() { return; }
    if std::path::Path::new("nn-aether.nnue").exists() {
        init_nnue("nn-aether.nnue");
    } else {
        match load_network_from_reader(&mut Cursor::new(EMBEDDED_NET)) {
            Ok(net) => {
                NETWORK.set(net).ok();
                println!("NNUE loaded successfully from embedded data");
            }
            Err(e) => println!("Failed to load embedded NNUE: {}", e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screlu_avx2_vs_scalar() {
        if !use_avx2() { return; }
        let mut inputs = [0i16; 256];
        for i in 0..256 { inputs[i] = i as i16; }
        unsafe {
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
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}
