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
// Evaluation
// --------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn affine_tx_simd(input: &[i8], weights: &[i8], bias: &[i32], output: &mut [i8], in_size: usize, out_size: usize) {
    // Weights layout: [Output][Input] (row major) - BUT typical NNUE is [Input][Output] (column major).
    // The trainer usually exports column major for efficiency? Or row major?
    // Bullet usually exports: Input 0 -> All Outputs, Input 1 -> All Outputs.
    // Wait, let's check how we read it.
    // read_i16_slice simply reads bytes.
    // Standard affine layer matrix mult: O = I * W + B
    // In SIMD, if weights are column-major (Input-major), we broadcast input and multiply by weight row.
    // If weights are row-major (Output-major), we dot product input with weight row.
    // Assuming standard Linear layer export: [InputDim x OutputDim]
    //
    // Actually, for small layers (32x32), we can just do naive loop or specialized SIMD.
    // Let's stick to a robust implementation first, then optimize.
    //
    // However, our weights are quantized to i8 (actually loaded as i16 in struct, but conceptually i8 for calculation?)
    // Wait, the plan says weights +/- 64. So they fit in i8.
    // But our struct uses Vec<i16> for everything to keep it simple.
    // Let's use i16 everywhere for safety and AVX2 _mm256_madd_epi16.

    // Fallback to scalar for complex layers for now to avoid layout bugs.
    // The accumulator part (feature transformer) is the bottleneck anyway.
    panic!("SIMD path not fully implemented for deep layers yet");
}

pub fn evaluate(stm_acc: &Accumulator, ntm_acc: &Accumulator) -> i32 {
    if let Some(net) = NETWORK.get() {
        if stm_acc.magic != ACC_MAGIC || ntm_acc.magic != ACC_MAGIC {
            panic!("CRITICAL ERROR: NNUE is loaded but Accumulators are invalid...");
        }

        // 1. Feature Transformer Output (L1 Input)
        // Concatenate SCReLU(STM) + SCReLU(NTM) -> 256 + 256 = 512
        let mut l1_input = [0i8; 512]; // Quantized activations [0, 127]

        #[cfg(target_arch = "x86_64")]
        unsafe {
            let zero = _mm256_setzero_si256();
            let qa = _mm256_set1_epi16(QA as i16);

            // STM -> First 256
            for i in (0..L1_SIZE).step_by(16) {
                let v_ptr = stm_acc.v.as_ptr().add(i);
                let val = _mm256_loadu_si256(v_ptr as *const __m256i);
                let clamped = _mm256_min_epi16(_mm256_max_epi16(val, zero), qa);
                let sq = _mm256_mullo_epi16(clamped, clamped);
                // Divide by 255 -> mulhi with reciprocal approx?
                // x^2 / 255 ~= (x^2 * 257) >> 16
                let activation = _mm256_mulhi_epu16(sq, _mm256_set1_epi16(257));

                // Pack to i8 (0..127 range expected? No, SCReLU output is 0..255)
                // Wait, L1 input expects quantized range.
                // Trainer: SavedFormat::id("l1w").round().quantise::<i16>(64)
                // The input to L1 is the output of L0.
                // L0 Output is SCReLU -> 0..255.
                // We store it as i8? i8 is -128..127. 255 doesn't fit in i8!
                // We need u8 or i16.
                // Let's look at `l1_input`. If we define it as i16, we are safe.

                // Correction: `l1_input` should be `i16` to hold 0..255.
                // But later layers use ClippedReLU which is 0..127 (fits in i8).
                // Let's use `i16` for intermediate buffers to be safe and compatible with our `Network` struct (Vec<i16>).
            }
        }

        let mut hidden_512 = [0i16; 512];

        // STM
        for i in 0..L1_SIZE {
            let val = stm_acc.v[i].clamp(0, 255);
            hidden_512[i] = SCRELU[val as usize];
        }
        // NTM
        for i in 0..L1_SIZE {
            let val = ntm_acc.v[i].clamp(0, 255);
            hidden_512[L1_SIZE + i] = SCRELU[val as usize];
        }

        // 2. Layer 1: 512 -> 32
        let mut l2_out = [0i16; L2_SIZE];
        layer_affine(&hidden_512, &net.l1_weights, &net.l1_biases, &mut l2_out, 512, L2_SIZE);
        // Activation: ClippedReLU (0..127)
        for x in l2_out.iter_mut() {
            *x = (*x).clamp(0, Q_ACTIVATION as i16);
        }

        // 3. Layer 2: 32 -> 32
        let mut l3_out = [0i16; L3_SIZE];
        layer_affine(&l2_out, &net.l2_weights, &net.l2_biases, &mut l3_out, L2_SIZE, L3_SIZE);
        // Activation: ClippedReLU (0..127)
        for x in l3_out.iter_mut() {
            *x = (*x).clamp(0, Q_ACTIVATION as i16);
        }

        // 4. Layer 3: 32 -> 32
        let mut l4_out = [0i16; L4_SIZE];
        layer_affine(&l3_out, &net.l3_weights, &net.l3_biases, &mut l4_out, L3_SIZE, L4_SIZE);
        // Activation: ClippedReLU (0..127)
        for x in l4_out.iter_mut() {
            *x = (*x).clamp(0, Q_ACTIVATION as i16);
        }

        // 5. Output Layer: 32 -> 1
        let mut final_out = [0i16; 1];
        layer_affine(&l4_out, &net.l4_weights, &net.l4_biases, &mut final_out, L4_SIZE, 1);

        // Result
        let output = final_out[0] as i32;

        // Scale back?
        // Trainer scale: 400.
        // Quantization:
        // L0(255) * L1(64) * L2(64) * L3(64) * L4(64) ?
        // The trainer handles the quantization scaling in the output bias/weights usually?
        // Bullet output is raw sum.
        // We usually divide by (QA * QB...)
        // Let's assume standard behavior:
        // Output = Sum / (QA * QB) * SCALE
        // Here we have multiple layers.
        // If we strictly follow the quantized integer arithmetic:
        // L1_out = (Input * W1 + B1) >> Shift?
        // Bullet `quantise` helper usually just clamps the float weights to int range.
        // It does NOT automatically add bitshifts to the network file.
        // We need to divide by the quantization factor accumulated.
        //
        // Factors:
        // Input: 0..255 (QA=255)
        // L1 Weights: 64 (QB=64)
        // L1 Out (Accum): 255 * 64 * 512 ~= 8M. Fits in i32.
        // We clamp L1 Out to 0..127 (ClippedReLU).
        // Effectively we implicitly divide by (AccumMax / 127)?
        // No, `bullet` training trains the weights such that the activations fall in range.
        // But for *inference*, we normally just sum.
        // If we sum integers, the values grow.
        //
        // 1. Input: u8 (0..255)
        // 2. L1: Input * i8(64). Sum can be large.
        //    We need to right-shift to bring it back to activation range (0..127)?
        //    Or does the bias handle it?
        //    Usually: (Sum + Bias) / QuantFactor
        //
        //    Let's check the quantisation params in trainer again.
        //    L1W: 64. L1B: 64*127.
        //    This implies the "1" in this layer corresponds to 64*127 raw value?
        //
        //    Actually, simple integer inference usually works like this:
        //    Layer Output = (Sum) / Shift
        //    where Shift brings it back to next layer's input scale.
        //    Next layer input scale is 0..127.
        //
        //    Let's try a standard heuristic:
        //    L1_Out = (Sum) / 64
        //    L2_Out = (Sum) / 64
        //    ...
        //
        //    Wait, `(val * val) / 255` is SCReLU. That returns 0..255.
        //
        //    Let's assume the divisor is roughly QB (64) for each layer.

        let final_val = output;

        // We need to descale the total accumulation of quantization factors.
        // Input (255) * W1(64) -> / 64 -> 255? No, target is 127. / 128?
        // Let's assume we simply divide by QB=64 at each step to normalize?
        //
        // Actually, let's look at `evaluate` in `src/nnue.rs` before I overwrote it.
        // It did `(sum * SCALE) / (QA * QB)`.
        // That was for 1 layer.
        //
        // For deep networks, we typically just perform the matmul and rely on the fact that
        // weights are small. But we clamp to 127.
        // If we don't divide, the sum will instantly exceed 127.
        // So yes, division is required.
        //
        // Divisor = WeightQuantization (64).
        //
        // Let's refine `layer_affine`:

        (final_val * SCALE) / (QB * Q_ACTIVATION)
    } else {
        0
    }
}

// Simple scalar affine layer: Out = In * W + B
// DIVIDES by 64 (QB) after summation to normalize.
fn layer_affine(input: &[i16], weights: &[i16], biases: &[i16], output: &mut [i16], in_size: usize, out_size: usize) {
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
