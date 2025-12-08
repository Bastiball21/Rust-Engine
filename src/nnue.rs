// src/nnue.rs
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::path::PathBuf;
use std::env;
use std::sync::RwLock;

// Architecture Constants
pub const LAYER1_SIZE: usize = 256;
pub const INPUT_SIZE: usize = 41024;
pub const HIDDEN_SIZE: usize = 32;

// SAFE GLOBAL
pub static NNUE: RwLock<Option<NnueWeights>> = RwLock::new(None);

#[derive(Clone, Copy, Debug)]
#[repr(align(64))]
pub struct Accumulator {
    pub v: [i16; LAYER1_SIZE],
}

impl Accumulator {
    pub fn default() -> Self {
        Accumulator { v: [0; LAYER1_SIZE] }
    }

    pub fn refresh(&mut self, state: &crate::state::GameState, perspective: usize) {
        if let Ok(guard) = NNUE.read() {
            if let Some(net) = guard.as_ref() {
                // FIXED: Use Unaligned Store (storeu) to prevent stack misalignment crashes
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                unsafe {
                     use std::arch::x86_64::*;
                     let dst = self.v.as_mut_ptr();
                     let src = net.feature_biases.as_ptr();
                     for i in (0..LAYER1_SIZE).step_by(16) {
                         // loadu is safer for src (though heap is usually aligned)
                         let reg = _mm256_loadu_si256(src.add(i) as *const __m256i);
                         // storeu is CRITICAL for dst (stack memory)
                         _mm256_storeu_si256(dst.add(i) as *mut __m256i, reg);
                     }
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                self.v.copy_from_slice(&net.feature_biases);

                let king_sq = state.bitboards[if perspective == crate::state::WHITE { crate::state::K } else { crate::state::k }].get_lsb_index() as usize;

                for piece in 0..12 {
                    let mut bb = state.bitboards[piece];
                    while bb.0 != 0 {
                        let sq = bb.get_lsb_index() as usize;
                        bb.pop_bit(sq as u8);
                        if let Some(idx) = make_halfkp_index(perspective, king_sq, piece, sq) {
                            self.add_feature(idx, net);
                        }
                    }
                }
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

            // Unroll loop 16 items (256 bits) at a time
            for i in (0..LAYER1_SIZE).step_by(16) {
                // FIXED: Use loadu/storeu everywhere
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

            // Unroll loop 16 items (256 bits) at a time
            for i in (0..LAYER1_SIZE).step_by(16) {
                // FIXED: Use loadu/storeu everywhere
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
        if let Ok(guard) = NNUE.read() {
            if let Some(net) = guard.as_ref() {
                for &idx in removed {
                    self.sub_feature(idx, net);
                }
                for &idx in added {
                    self.add_feature(idx, net);
                }
            }
        }
    }
}

// --------------------------------------------------------
// SIMD Inference Logic
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

pub fn evaluate_nnue_avx2(acc_us: &Accumulator, acc_them: &Accumulator, net: &NnueWeights) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        let mut input = [0i8; 512];
        let input_ptr = input.as_mut_ptr();

        // Clamp and Pack
        for i in 0..256 {
            *input_ptr.add(i) = acc_us.v[i].clamp(0, 255) as i8;
            *input_ptr.add(256+i) = acc_them.v[i].clamp(0, 255) as i8;
        }

        // --- LAYER 1 ---
        let mut layer1_out = [0i8; 32];

        for i in 0..32 {
            let mut sum_vec = _mm256_setzero_si256();
            let row_offset = i * 512;
            let weights_ptr = net.layer1_weights.as_ptr().add(row_offset);

            for k in (0..512).step_by(32) {
                // Safe Implementation for QA=255 (Avoid maddubs saturation)
                let inp = _mm256_loadu_si256(input_ptr.add(k) as *const __m256i);
                let w = _mm256_loadu_si256(weights_ptr.add(k) as *const __m256i);

                // Lower 128 bits
                let inp_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(inp));
                let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
                let prod_lo = _mm256_madd_epi16(inp_lo, w_lo);

                // Upper 128 bits
                let inp_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(inp, 1));
                let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
                let prod_hi = _mm256_madd_epi16(inp_hi, w_hi);

                sum_vec = _mm256_add_epi32(sum_vec, _mm256_add_epi32(prod_lo, prod_hi));
            }

            let total = hsum_256_epi32(sum_vec) + net.layer1_biases[i];
            layer1_out[i] = (total >> 6).clamp(0, 255) as i8;
        }

        // --- LAYER 2 ---
        let mut layer2_out = [0i8; 32];
        let l1_vec = _mm256_loadu_si256(layer1_out.as_ptr() as *const __m256i);

        for i in 0..32 {
            let row_offset = i * 32;
            let w = _mm256_loadu_si256(net.layer2_weights.as_ptr().add(row_offset) as *const __m256i);

            // Safe Implementation
            let inp_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(l1_vec));
            let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w));
            let prod_lo = _mm256_madd_epi16(inp_lo, w_lo);

            let inp_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(l1_vec, 1));
            let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w, 1));
            let prod_hi = _mm256_madd_epi16(inp_hi, w_hi);

            let sum_i32 = _mm256_add_epi32(prod_lo, prod_hi);

            let total = hsum_256_epi32(sum_i32) + net.layer2_biases[i];
            layer2_out[i] = (total >> 6).clamp(0, 255) as i8;
        }

        // --- OUTPUT ---
        let l2_vec = _mm256_loadu_si256(layer2_out.as_ptr() as *const __m256i);
        let out_w = _mm256_loadu_si256(net.output_weights.as_ptr() as *const __m256i);

        // Safe Implementation
        let inp_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(l2_vec));
        let w_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(out_w));
        let prod_lo = _mm256_madd_epi16(inp_lo, w_lo);

        let inp_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(l2_vec, 1));
        let w_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(out_w, 1));
        let prod_hi = _mm256_madd_epi16(inp_hi, w_hi);

        let sum_i32 = _mm256_add_epi32(prod_lo, prod_hi);

        let final_sum = hsum_256_epi32(sum_i32) + net.output_bias;

        // Scaling for QA=255/QB=64 to approx centipawns
        // 16 was too small, yielding very high scores (e.g. 166cp for startpos).
        // 64 is more reasonable (166 / 4 ~= 41).
        return final_sum / 64; // Removed +10 bias
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    return 0;
}

// --------------------------------------------------------
// Standard Loading Code
// --------------------------------------------------------

#[repr(align(64))]
pub struct NnueWeights {
    pub feature_biases: [i16; LAYER1_SIZE],
    pub feature_weights: Vec<i16>,
    pub layer1_biases: [i32; HIDDEN_SIZE],
    pub layer1_weights: [i8; HIDDEN_SIZE * 512],
    pub layer2_biases: [i32; HIDDEN_SIZE],
    pub layer2_weights: [i8; HIDDEN_SIZE * 32],
    pub output_bias: i32,
    pub output_weights: [i8; HIDDEN_SIZE],
}

pub fn make_halfkp_index(perspective: usize, king_sq: usize, piece: usize, sq: usize) -> Option<usize> {
    let orient_sq = if perspective == crate::state::WHITE { sq } else { sq ^ 56 };
    let orient_king = if perspective == crate::state::WHITE { king_sq } else { king_sq ^ 56 };
    let piece_color = if piece < 6 { crate::state::WHITE } else { crate::state::BLACK };
    let piece_type = piece % 6;

    if piece_type == 5 { return None; }

    let kp_idx = if piece_color == perspective { piece_type } else { piece_type + 5 };
    Some(orient_king * 641 + kp_idx * 64 + orient_sq)
}

pub fn init_nnue(filename: &str) {
    let path = resolve_path(filename);
    println!("Loading NNUE from: {:?}", path);

    match load_nnue_file(&path) {
        Ok(weights) => {
             if weights.feature_weights.iter().take(100).all(|&x| x == 0) {
                 println!("WARNING: Weights look empty. Check generation.");
             }
             if let Ok(mut guard) = NNUE.write() {
                *guard = Some(weights);
            }
            println!("NNUE Loaded Successfully (AVX2 Enabled).");
        },
        Err(e) => {
            println!("Error Loading NNUE: {}. Defaulting to HCE.", e);
        }
    }
}

fn resolve_path(filename: &str) -> PathBuf {
    let path = PathBuf::from(filename);
    if path.exists() { return path; }
    if let Ok(exe_path) = env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let alt_path = exe_dir.join(filename);
            if alt_path.exists() { return alt_path; }
        }
    }
    path
}

fn load_nnue_file(path: &PathBuf) -> io::Result<NnueWeights> {
    let f = File::open(path)?;
    let mut reader = BufReader::new(f);

    let _version = read_u32(&mut reader)?;
    let _hash = read_u32(&mut reader)?;
    let desc_len = read_u32(&mut reader)?;
    for _ in 0..desc_len { read_u8(&mut reader)?; }
    let _hash_transformer = read_u32(&mut reader)?;

    let mut feature_biases = [0i16; LAYER1_SIZE];
    read_i16_buf(&mut reader, &mut feature_biases)?;

    let count = INPUT_SIZE * LAYER1_SIZE;
    let mut feature_weights = vec![0i16; count];
    read_i16_buf(&mut reader, &mut feature_weights)?;

    let _hash_network = read_u32(&mut reader)?;

    let mut layer1_biases = [0i32; HIDDEN_SIZE];
    read_i32_buf(&mut reader, &mut layer1_biases)?;
    let mut layer1_weights = [0i8; HIDDEN_SIZE * 512];
    read_i8_buf(&mut reader, &mut layer1_weights)?;

    let mut layer2_biases = [0i32; HIDDEN_SIZE];
    read_i32_buf(&mut reader, &mut layer2_biases)?;
    let mut layer2_weights = [0i8; HIDDEN_SIZE * 32];
    read_i8_buf(&mut reader, &mut layer2_weights)?;

    let mut output_bias_buf = [0i32; 1];
    read_i32_buf(&mut reader, &mut output_bias_buf)?;
    let mut output_weights = [0i8; HIDDEN_SIZE];
    read_i8_buf(&mut reader, &mut output_weights)?;

    Ok(NnueWeights {
        feature_biases, feature_weights,
        layer1_biases, layer1_weights,
        layer2_biases, layer2_weights,
        output_bias: output_bias_buf[0], output_weights,
    })
}

// Low-level readers
fn read_u32<R: Read>(reader: &mut R) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}
fn read_u8<R: Read>(reader: &mut R) -> io::Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}
fn read_i16_buf<R: Read>(reader: &mut R, buf: &mut [i16]) -> io::Result<()> {
    let byte_count = buf.len() * 2;
    let ptr = buf.as_mut_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, byte_count) };
    reader.read_exact(slice)?;
    if cfg!(target_endian = "big") { for x in buf { *x = x.to_le(); } }
    Ok(())
}
fn read_i32_buf<R: Read>(reader: &mut R, buf: &mut [i32]) -> io::Result<()> {
    let byte_count = buf.len() * 4;
    let ptr = buf.as_mut_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, byte_count) };
    reader.read_exact(slice)?;
    if cfg!(target_endian = "big") { for x in buf { *x = x.to_le(); } }
    Ok(())
}
fn read_i8_buf<R: Read>(reader: &mut R, buf: &mut [i8]) -> io::Result<()> {
    let ptr = buf.as_mut_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, buf.len()) };
    reader.read_exact(slice)
}