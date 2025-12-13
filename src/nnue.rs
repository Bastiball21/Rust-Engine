// src/nnue.rs
use std::sync::RwLock;

// Architecture Constants
pub const LAYER1_SIZE: usize = 128; // Hidden size
pub const INPUT_SIZE: usize = 768; // Input size (Chess768)

// SAFE GLOBAL
pub static NNUE: RwLock<NnueWeights> = RwLock::new(NnueWeights::new());

#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
pub struct Accumulator {
    pub v: [i16; LAYER1_SIZE],
}

impl Accumulator {
    pub fn default() -> Self {
        Accumulator {
            v: [0; LAYER1_SIZE],
        }
    }

    pub fn refresh(&mut self, _state: &crate::state::GameState, _perspective: usize) {
        // Disabled
    }

    pub fn update(&mut self, _added: &[usize], _removed: &[usize]) {
        // Disabled
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

pub fn evaluate_nnue(_acc_us: &Accumulator, _acc_them: &Accumulator) -> i32 {
    0 // Placeholder
}

// --------------------------------------------------------
// Weights & Loading
// --------------------------------------------------------

pub struct NnueWeights {
    pub feature_biases: [i16; LAYER1_SIZE],
    pub feature_weights: Vec<i16>, // 768 * 128
    pub output_weights: Vec<i16>,  // 256 (128 + 128)
    pub output_bias: i16,
}

impl NnueWeights {
    pub const fn new() -> Self {
        NnueWeights {
            feature_biases: [0; LAYER1_SIZE],
            feature_weights: Vec::new(),
            output_weights: Vec::new(),
            output_bias: 0,
        }
    }
}

pub fn init_nnue() {
    println!("NNUE Disabled.");
}
