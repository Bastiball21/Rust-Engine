// src/nnue.rs
use std::fs::File;
use std::io::{self, Read};
use std::sync::OnceLock;

// Architecture Constants
pub const LAYER1_SIZE: usize = 128;
pub const INPUT_SIZE: usize = 768;
const QA: i32 = 255;
const QB: i32 = 64;
const SCALE: i32 = 400; // Eval scale from trainer

// SAFE GLOBAL NETWORK
pub static NETWORK: OnceLock<Network> = OnceLock::new();

#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
pub struct Accumulator {
    pub v: [i16; LAYER1_SIZE],
}

impl Accumulator {
    pub fn default() -> Self {
        // Initialize with biases if network is loaded, else 0
        let mut acc = Accumulator {
            v: [0; LAYER1_SIZE],
        };
        if let Some(net) = NETWORK.get() {
            acc.v.copy_from_slice(&net.feature_biases);
        }
        acc
    }

    pub fn refresh(&mut self, bitboards: &[crate::bitboard::Bitboard; 12], perspective: usize) {
        if let Some(net) = NETWORK.get() {
            // Start with biases
            self.v.copy_from_slice(&net.feature_biases);

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

                        let idx = make_index(perspective, piece, sq);
                        self.add_feature(idx, net);
                    }
                }
            }
        }
    }

    pub fn update(&mut self, added: &[(usize, usize)], removed: &[(usize, usize)], perspective: usize) {
        if let Some(net) = NETWORK.get() {
            for &(piece, sq) in removed {
                let idx = make_index(perspective, piece, sq);
                self.sub_feature(idx, net);
            }
            for &(piece, sq) in added {
                let idx = make_index(perspective, piece, sq);
                self.add_feature(idx, net);
            }
        }
    }

    // Helper to add a feature's weights
    #[inline(always)]
    fn add_feature(&mut self, feature_idx: usize, net: &Network) {
        let offset = feature_idx * LAYER1_SIZE;
        let weights = &net.feature_weights[offset..offset + LAYER1_SIZE];

        // Auto-vectorization should handle this loop
        for i in 0..LAYER1_SIZE {
            self.v[i] = self.v[i].wrapping_add(weights[i]);
        }
    }

    #[inline(always)]
    fn sub_feature(&mut self, feature_idx: usize, net: &Network) {
        let offset = feature_idx * LAYER1_SIZE;
        let weights = &net.feature_weights[offset..offset + LAYER1_SIZE];

        for i in 0..LAYER1_SIZE {
            self.v[i] = self.v[i].wrapping_sub(weights[i]);
        }
    }
}

// --------------------------------------------------------
// Feature Indexer (Chess768)
// --------------------------------------------------------

pub fn make_index(perspective: usize, piece: usize, sq: usize) -> usize {
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

    // Index = Context + PieceType * 64 + Square
    context_offset + piece_type * 64 + orient_sq
}

// --------------------------------------------------------
// Evaluation
// --------------------------------------------------------

pub fn evaluate(stm_acc: &Accumulator, ntm_acc: &Accumulator) -> i32 {
    if let Some(net) = NETWORK.get() {
        let mut sum = net.output_bias as i32;

        // Perspective 1 (STM)
        for i in 0..LAYER1_SIZE {
            let val = stm_acc.v[i];
            let clamped = val.clamp(0, QA as i16) as i32;
            let activation = (clamped * clamped) / QA; // SCReLU / QA

            // Output weight index for STM: 0..127
            let weight = net.output_weights[i] as i32;
            sum += activation * weight;
        }

        // Perspective 2 (NTM)
        for i in 0..LAYER1_SIZE {
            let val = ntm_acc.v[i];
            let clamped = val.clamp(0, QA as i16) as i32;
            let activation = (clamped * clamped) / QA; // SCReLU / QA

            // Output weight index for NTM: 128..255
            let weight = net.output_weights[LAYER1_SIZE + i] as i32;
            sum += activation * weight;
        }

        // Scale to centipawns
        // Output scale is QA * QB
        // We want (sum / (QA * QB)) * SCALE
        // = sum * SCALE / (QA * QB)
        // = sum * 400 / (255 * 64)
        // = sum * 400 / 16320
        // = sum / 40.8

        // Integer approx: sum * 400 / 16320
        (sum * SCALE) / (QA * QB)
    } else {
        0
    }
}

// --------------------------------------------------------
// Network & Loading
// --------------------------------------------------------

pub struct Network {
    pub feature_weights: Vec<i16>, // 768 * 128
    pub feature_biases: Vec<i16>,  // 128
    pub output_weights: Vec<i16>,  // 256
    pub output_bias: i16,
}

pub fn load_network(path: &str) -> io::Result<Network> {
    let mut file = File::open(path)?;

    // Read Feature Weights (l0w): 768 * 128 * 2 bytes
    let mut feature_weights = vec![0i16; INPUT_SIZE * LAYER1_SIZE];
    read_i16_slice(&mut file, &mut feature_weights)?;

    // Read Feature Biases (l0b): 128 * 2 bytes
    let mut feature_biases = vec![0i16; LAYER1_SIZE];
    read_i16_slice(&mut file, &mut feature_biases)?;

    // Read Output Weights (l1w): 256 * 2 bytes
    let mut output_weights = vec![0i16; 2 * LAYER1_SIZE];
    read_i16_slice(&mut file, &mut output_weights)?;

    // Read Output Bias (l1b): 1 * 2 bytes (SavedFormat::quantise::<i16>)
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
    use std::io::Write;

    #[test]
    fn test_accumulator_logic() {
        // Test basic i16 wrapping logic manually since we can't easily mock Network global
        // without init_nnue which requires a file.

        let mut acc = Accumulator::default();
        // Should be 0s if no net loaded
        for x in acc.v { assert_eq!(x, 0); }

        // Manual update simulation
        let mut v = [0i16; 128];
        let weight = [10i16; 128];

        // Add
        for i in 0..128 { v[i] = v[i].wrapping_add(weight[i]); }
        for i in 0..128 { assert_eq!(v[i], 10); }

        // Sub
        for i in 0..128 { v[i] = v[i].wrapping_sub(weight[i]); }
        for i in 0..128 { assert_eq!(v[i], 0); }
    }

    #[test]
    fn test_make_index() {
        // Perspective White (0)
        // White Pawn (0) on A1 (0) -> Friendly (0) + 0*64 + 0 = 0
        assert_eq!(make_index(crate::state::WHITE, 0, 0), 0);

        // Black Pawn (6) on A1 (0) -> Enemy (384) + 0*64 + 0 = 384
        assert_eq!(make_index(crate::state::WHITE, 6, 0), 384);

        // Perspective Black (1)
        // Black Pawn (6) on A1 (0). Relative square A1 for Black is A8? No, A1 (0) ^ 56 = A8 (56).
        // Friendly (0) + 0*64 + 56 = 56.
        assert_eq!(make_index(crate::state::BLACK, 6, 0), 56);

        // White Pawn (0) on A1 (0). Relative square 56.
        // Enemy (384) + 0*64 + 56 = 440.
        assert_eq!(make_index(crate::state::BLACK, 0, 0), 384 + 56);
    }
}
