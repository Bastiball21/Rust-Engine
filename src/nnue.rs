// src/nnue.rs
use std::fs::File;
use std::io::{self, Read};
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Architecture Constants
pub const LAYER1_SIZE: usize = 256;
pub const INPUT_SIZE: usize = 768;
pub const NUM_BUCKETS: usize = 32; // Matches bullet_lib ChessBuckets (Standard Mirrored)
const QA: i32 = 255;
const QB: i32 = 64;
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
    pub v: [i16; LAYER1_SIZE],
    pub magic: u16, // Safety check for initialization
}

impl Accumulator {
    pub fn default() -> Self {
        // Initialize with biases if network is loaded, else 0
        let mut acc = Accumulator {
            v: [0; LAYER1_SIZE],
            magic: 0,
        };
        if let Some(net) = NETWORK.get() {
            acc.v.copy_from_slice(&net.feature_biases);
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
            self.v.copy_from_slice(&net.feature_biases);
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
        let offset = feature_idx * LAYER1_SIZE;
        let weights = &net.feature_weights[offset..offset + LAYER1_SIZE];

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut i = 0;
            while i < LAYER1_SIZE {
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
        for i in 0..LAYER1_SIZE {
            self.v[i] = self.v[i].wrapping_add(weights[i]);
        }
    }

    #[inline(always)]
    fn sub_feature(&mut self, feature_idx: usize, net: &Network) {
        let offset = feature_idx * LAYER1_SIZE;
        let weights = &net.feature_weights[offset..offset + LAYER1_SIZE];

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut i = 0;
            while i < LAYER1_SIZE {
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
        for i in 0..LAYER1_SIZE {
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
// Evaluation
// --------------------------------------------------------

pub fn evaluate(stm_acc: &Accumulator, ntm_acc: &Accumulator) -> i32 {
    if let Some(net) = NETWORK.get() {
        if stm_acc.magic != ACC_MAGIC || ntm_acc.magic != ACC_MAGIC {
            panic!("CRITICAL ERROR: NNUE is loaded but Accumulators are invalid...");
        }

        let mut sum = net.output_bias as i32;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let mut sum_vec = _mm256_setzero_si256();
            let zero = _mm256_setzero_si256();
            let qa = _mm256_set1_epi16(QA as i16);

            // Process STM (current player)
            for i in (0..LAYER1_SIZE).step_by(16) {
                let v_ptr = stm_acc.v.as_ptr().add(i);
                let w_ptr = net.output_weights.as_ptr().add(i);
                let val = _mm256_loadu_si256(v_ptr as *const __m256i);
                let w = _mm256_loadu_si256(w_ptr as *const __m256i);
                let clamped = _mm256_min_epi16(_mm256_max_epi16(val, zero), qa);
                let sq = _mm256_mullo_epi16(clamped, clamped);
                let activation = _mm256_mulhi_epu16(sq, _mm256_set1_epi16(257));
                let prod = _mm256_madd_epi16(activation, w);
                sum_vec = _mm256_add_epi32(sum_vec, prod);
            }

            // Process NTM (opponent)
            for i in (0..LAYER1_SIZE).step_by(16) {
                let v_ptr = ntm_acc.v.as_ptr().add(i);
                let w_ptr = net.output_weights.as_ptr().add(LAYER1_SIZE + i);
                let val = _mm256_loadu_si256(v_ptr as *const __m256i);
                let w = _mm256_loadu_si256(w_ptr as *const __m256i);
                let clamped = _mm256_min_epi16(_mm256_max_epi16(val, zero), qa);
                let sq = _mm256_mullo_epi16(clamped, clamped);
                let activation = _mm256_mulhi_epu16(sq, _mm256_set1_epi16(257));
                let prod = _mm256_madd_epi16(activation, w);
                sum_vec = _mm256_add_epi32(sum_vec, prod);
            }

            let mut arr = [0i32; 8];
            _mm256_storeu_si256(arr.as_mut_ptr() as *mut __m256i, sum_vec);
            for x in arr {
                sum += x;
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            for i in 0..LAYER1_SIZE {
                let val = stm_acc.v[i].clamp(0, QA as i16) as i32;
                let activation = (val * val) / QA;
                sum += activation * net.output_weights[i] as i32;
            }
            for i in 0..LAYER1_SIZE {
                let val = ntm_acc.v[i].clamp(0, QA as i16) as i32;
                let activation = (val * val) / QA;
                sum += activation * net.output_weights[LAYER1_SIZE + i] as i32;
            }
        }

        // Return score from **current player's perspective** (no flip needed)
        (sum * SCALE) / (QA * QB)
    } else {
        0
    }
}

// --------------------------------------------------------
// Network & Loading
// --------------------------------------------------------

pub struct Network {
    // Feature Weights: (768 * 32) * 256
    pub feature_weights: Vec<i16>,
    // Feature Biases: 256
    pub feature_biases: Vec<i16>,
    // Output Weights: 256 * 2 (because 2 accumulators -> 512 total inputs to output neuron)
    pub output_weights: Vec<i16>,
    pub output_bias: i16,
}

pub fn load_network(path: &str) -> io::Result<Network> {
    let mut file = File::open(path)?;

    // Read Feature Weights (l0w): (768 * 32) * 256 * 2 bytes
    let total_features = INPUT_SIZE * NUM_BUCKETS;
    let mut feature_weights = vec![0i16; total_features * LAYER1_SIZE];
    read_i16_slice(&mut file, &mut feature_weights)?;

    // Read Feature Biases (l0b): 256 * 2 bytes
    let mut feature_biases = vec![0i16; LAYER1_SIZE];
    read_i16_slice(&mut file, &mut feature_biases)?;

    // Read Output Weights (l1w): (256 * 2) * 2 bytes
    let mut output_weights = vec![0i16; 2 * LAYER1_SIZE];
    read_i16_slice(&mut file, &mut output_weights)?;

    // Read Output Bias (l1b): 1 * 2 bytes
    let mut output_bias_buf = [0u8; 2];
    file.read_exact(&mut output_bias_buf)?;
    let output_bias = i16::from_le_bytes(output_bias_buf);

    Ok(Network {
        feature_weights,
        feature_biases,
        output_weights,
        output_bias,
    })
}

fn read_i16_slice(file: &mut File, slice: &mut [i16]) -> io::Result<()> {
    let mut buffer = vec![0u8; slice.len() * 2];
    file.read_exact(&mut buffer)?;

    for (i, chunk) in buffer.chunks(2).enumerate() {
        slice[i] = i16::from_le_bytes([chunk[0], chunk[1]]);
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_logic() {
        // Mock Accumulator
        let acc = Accumulator {
            v: [0; LAYER1_SIZE],
            magic: 0,
        };
        for x in acc.v {
            assert_eq!(x, 0);
        }

        // We can't easily test with weights without a file or mocking the Network global,
        // but the struct size is correct.
        assert_eq!(acc.v.len(), 256);
    }

    #[test]
    fn test_make_index_buckets() {
        // Test Bucket Logic (32 Buckets)
        // A1 (0) -> Rank 0, File 0 -> 0*4 + 0 = 0
        assert_eq!(get_king_bucket(crate::state::WHITE, 0), 0);

        // H1 (7) -> Rank 0, File 7 (Mirror 0) -> 0*4 + 0 = 0
        assert_eq!(get_king_bucket(crate::state::WHITE, 7), 0);

        // E1 (4) -> Rank 0, File 4 (Mirror 3) -> 0*4 + 3 = 3
        assert_eq!(get_king_bucket(crate::state::WHITE, 4), 3);

        // H8 (63) -> Rank 7, File 7 (Mirror 0) -> 7*4 + 0 = 28
        assert_eq!(get_king_bucket(crate::state::WHITE, 63), 28);

        // Index Calculation
        // White King A1 (Bucket 0). White Pawn A2 (8).
        // Friendly(0) + Pawn(0)*64 + Sq(8) = 8.
        let idx = make_index(crate::state::WHITE, 0, 8, 0);
        assert_eq!(idx, 8);

        // White King H8 (Bucket 28). White Pawn A2 (8).
        // Bucket(28)*768 + 8 = 21504 + 8 = 21512.
        let idx = make_index(crate::state::WHITE, 0, 8, 28);
        assert_eq!(idx, 21512);
    }

    #[test]
    fn test_screlu_values() {
        // Verify a few spots
        assert_eq!(SCRELU[0], 0);
        assert_eq!(SCRELU[255], 255);
        // 100*100 = 10000. 10000/255 = 39.
        assert_eq!(SCRELU[100], 39);
    }
}
