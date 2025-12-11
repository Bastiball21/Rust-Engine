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

    pub fn refresh(&mut self, state: &crate::state::GameState, perspective: usize) {
        let net = NNUE.read().unwrap();

        // Initialize with biases
        self.v.copy_from_slice(&net.feature_biases);

        // Iterate all pieces
        for piece in 0..12 {
            let mut bb = state.bitboards[piece];
            while bb.0 != 0 {
                let sq = bb.get_lsb_index() as usize;
                bb.pop_bit(sq as u8);

                let idx = make_index(perspective, piece, sq);
                self.add_feature(idx, &net);
            }
        }
    }

    fn add_feature(&mut self, idx: usize, net: &NnueWeights) {
        let offset = idx * LAYER1_SIZE;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            use std::arch::x86_64::*;
            let dst = self.v.as_mut_ptr();
            let src = net.feature_weights.as_ptr().add(offset);

            for i in (0..LAYER1_SIZE).step_by(16) {
                let v_acc = _mm256_loadu_si256(dst.add(i) as *const __m256i);
                let v_weight = _mm256_loadu_si256(src.add(i) as *const __m256i);
                let v_sum = _mm256_add_epi16(v_acc, v_weight);
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, v_sum);
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        for i in 0..LAYER1_SIZE {
            self.v[i] = self.v[i].wrapping_add(net.feature_weights[offset + i]);
        }
    }

    fn sub_feature(&mut self, idx: usize, net: &NnueWeights) {
        let offset = idx * LAYER1_SIZE;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            use std::arch::x86_64::*;
            let dst = self.v.as_mut_ptr();
            let src = net.feature_weights.as_ptr().add(offset);

            for i in (0..LAYER1_SIZE).step_by(16) {
                let v_acc = _mm256_loadu_si256(dst.add(i) as *const __m256i);
                let v_weight = _mm256_loadu_si256(src.add(i) as *const __m256i);
                let v_sub = _mm256_sub_epi16(v_acc, v_weight);
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, v_sub);
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        for i in 0..LAYER1_SIZE {
            self.v[i] = self.v[i].wrapping_sub(net.feature_weights[offset + i]);
        }
    }

    pub fn update(&mut self, added: &[usize], removed: &[usize]) {
        let net = NNUE.read().unwrap();
        for &idx in removed {
            self.sub_feature(idx, &net);
        }
        for &idx in added {
            self.add_feature(idx, &net);
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
// SIMD Inference Logic (SCReLU)
// --------------------------------------------------------

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

pub fn evaluate_nnue(acc_us: &Accumulator, acc_them: &Accumulator) -> i32 {
    let net = NNUE.read().unwrap();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        // SCReLU Implementation: (clamp(x, 0, 255)^2) / 255
        let mut sum_vec = _mm256_setzero_si256();

        // Process 'Us' (0..128)
        for i in (0..LAYER1_SIZE).step_by(16) {
            let v_us = _mm256_loadu_si256(acc_us.v.as_ptr().add(i) as *const __m256i);

            // Clamp 0..255
            let v_us_clamped = _mm256_max_epi16(v_us, _mm256_setzero_si256());
            let v_us_clamped = _mm256_min_epi16(v_us_clamped, _mm256_set1_epi16(255));

            // Square: (x * x)
            let v_us_sq = _mm256_mullo_epi16(v_us_clamped, v_us_clamped);

            let v_us_sq_lo = _mm256_unpacklo_epi16(v_us_sq, _mm256_setzero_si256());
            let v_us_sq_hi = _mm256_unpackhi_epi16(v_us_sq, _mm256_setzero_si256());

            // Divide by 255.0f
            let v_inv_255 = _mm256_set1_ps(1.0 / 255.0);

            let v_us_f_lo = _mm256_cvtepi32_ps(v_us_sq_lo);
            let v_us_f_hi = _mm256_cvtepi32_ps(v_us_sq_hi);

            let v_us_scaled_lo = _mm256_cvtps_epi32(_mm256_mul_ps(v_us_f_lo, v_inv_255));
            let v_us_scaled_hi = _mm256_cvtps_epi32(_mm256_mul_ps(v_us_f_hi, v_inv_255));

            let v_act = _mm256_packus_epi32(v_us_scaled_lo, v_us_scaled_hi);

            // Multiply by L1 weights (Us: 0..128)
            let w_us = _mm256_loadu_si256(net.output_weights.as_ptr().add(i) as *const __m256i);

            let prod = _mm256_madd_epi16(v_act, w_us);
            sum_vec = _mm256_add_epi32(sum_vec, prod);
        }

        // Process 'Them' (0..128)
        for i in (0..LAYER1_SIZE).step_by(16) {
            let v_them = _mm256_loadu_si256(acc_them.v.as_ptr().add(i) as *const __m256i);

            let v_clamped = _mm256_max_epi16(v_them, _mm256_setzero_si256());
            let v_clamped = _mm256_min_epi16(v_clamped, _mm256_set1_epi16(255));

            let v_sq = _mm256_mullo_epi16(v_clamped, v_clamped);

            let v_sq_lo = _mm256_unpacklo_epi16(v_sq, _mm256_setzero_si256());
            let v_sq_hi = _mm256_unpackhi_epi16(v_sq, _mm256_setzero_si256());

            let v_inv_255 = _mm256_set1_ps(1.0 / 255.0);

            let v_f_lo = _mm256_cvtepi32_ps(v_sq_lo);
            let v_f_hi = _mm256_cvtepi32_ps(v_sq_hi);

            let v_scaled_lo = _mm256_cvtps_epi32(_mm256_mul_ps(v_f_lo, v_inv_255));
            let v_scaled_hi = _mm256_cvtps_epi32(_mm256_mul_ps(v_f_hi, v_inv_255));

            let v_act = _mm256_packus_epi32(v_scaled_lo, v_scaled_hi);

            // Multiply by L1 weights (Them: 128..256)
            let w_them =
                _mm256_loadu_si256(net.output_weights.as_ptr().add(128 + i) as *const __m256i);

            let prod = _mm256_madd_epi16(v_act, w_them);
            sum_vec = _mm256_add_epi32(sum_vec, prod);
        }

        let total = hsum_256_epi32(sum_vec) + (net.output_bias as i32);

        return total / 64;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        // Fallback
        let mut sum = net.output_bias as i32;

        // Us
        for i in 0..LAYER1_SIZE {
            let val = acc_us.v[i].clamp(0, 255) as i32;
            let act = (val * val) / 255;
            sum += act * (net.output_weights[i] as i32);
        }

        // Them
        for i in 0..LAYER1_SIZE {
            let val = acc_them.v[i].clamp(0, 255) as i32;
            let act = (val * val) / 255;
            sum += act * (net.output_weights[128 + i] as i32);
        }

        return sum / 64;
    }
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

// Embed the binary
const NNUE_DATA: &[u8] = include_bytes!("resources/quantised.bin");

pub fn init_nnue() {
    let mut reader = std::io::Cursor::new(NNUE_DATA);
    use std::io::Read;

    // Helper to read i16
    fn read_i16<R: Read>(r: &mut R) -> std::io::Result<i16> {
        let mut buf = [0u8; 2];
        r.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    // Helper to read buffer
    fn read_buf<R: Read>(r: &mut R, buf: &mut [i16]) -> std::io::Result<()> {
        let ptr = buf.as_mut_ptr() as *mut u8;
        let len = buf.len() * 2;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, len) };
        r.read_exact(slice)?;
        // Ensure LE
        if cfg!(target_endian = "big") {
            for x in buf {
                *x = x.to_le();
            }
        }
        Ok(())
    }

    // 1. Read Feature Weights (l0w): 768 * 128
    let count = INPUT_SIZE * LAYER1_SIZE;
    let mut feature_weights = vec![0i16; count];
    read_buf(&mut reader, &mut feature_weights).expect("Failed to read feature weights");

    // 2. Read Feature Biases (l0b): 128
    let mut feature_biases = [0i16; LAYER1_SIZE];
    read_buf(&mut reader, &mut feature_biases).expect("Failed to read feature biases");

    // 3. Read Output Weights (l1w): 256 (128 * 2)
    let mut output_weights = vec![0i16; 2 * LAYER1_SIZE];
    read_buf(&mut reader, &mut output_weights).expect("Failed to read output weights");

    // 4. Read Output Bias (l1b): 1
    let output_bias = read_i16(&mut reader).expect("Failed to read output bias");

    println!("NNUE Embedded Architecture Loaded: 768 -> 128x2 -> 1");

    let weights = NnueWeights {
        feature_biases,
        feature_weights,
        output_weights,
        output_bias,
    };

    *NNUE.write().unwrap() = weights;
}
