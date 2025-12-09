// src/nnue.rs
use std::fs::File;
use std::io::{self, Read, BufReader};
use std::path::PathBuf;
use std::env;
use std::sync::RwLock;

// Architecture Constants
pub const LAYER1_SIZE: usize = 128;
pub const INPUT_SIZE: usize = 768;

// GLOBAL
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
                // Initialize with biases
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                unsafe {
                     use std::arch::x86_64::*;
                     let dst = self.v.as_mut_ptr();
                     let src = net.feature_biases.as_ptr();
                     for i in (0..LAYER1_SIZE).step_by(16) {
                         let reg = _mm256_loadu_si256(src.add(i) as *const __m256i);
                         _mm256_storeu_si256(dst.add(i) as *mut __m256i, reg);
                     }
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                self.v.copy_from_slice(&net.feature_biases);

                for piece in 0..12 {
                    let mut bb = state.bitboards[piece];
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
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, _mm256_add_epi16(v_acc, v_weight));
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
                _mm256_storeu_si256(dst.add(i) as *mut __m256i, _mm256_sub_epi16(v_acc, v_weight));
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

pub fn make_index(perspective: usize, piece: usize, sq: usize) -> usize {
    let orient_sq = if perspective == crate::state::WHITE { sq } else { sq ^ 56 };
    let piece_color = if piece < 6 { crate::state::WHITE } else { crate::state::BLACK };
    let piece_type = piece % 6;
    let rel_piece = if piece_color == perspective { piece_type } else { piece_type + 6 };
    rel_piece * 64 + orient_sq
}

#[repr(align(64))]
pub struct NnueWeights {
    pub feature_biases: [i16; LAYER1_SIZE],
    pub feature_weights: Vec<i16>,
    pub output_bias: i16,
    pub output_weights: [i16; 2 * LAYER1_SIZE],
}

pub fn evaluate(acc_us: &Accumulator, acc_them: &Accumulator, net: &NnueWeights) -> i32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    unsafe {
        use std::arch::x86_64::*;
        let mut sum_vec = _mm256_setzero_si256();
        let zero = _mm256_setzero_si256();
        let max = _mm256_set1_epi16(255);

        // Process Us (First 128 inputs to L1)
        for i in (0..LAYER1_SIZE).step_by(16) {
             let v_acc = _mm256_loadu_si256(acc_us.v.as_ptr().add(i) as *const __m256i);
             let v_clamped = _mm256_min_epi16(_mm256_max_epi16(v_acc, zero), max);

             let v_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_clamped));
             let v_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_clamped, 1));

             let v_sq_lo = _mm256_mullo_epi32(v_lo, v_lo);
             let v_sq_hi = _mm256_mullo_epi32(v_hi, v_hi);

             let w = _mm256_loadu_si256(net.output_weights.as_ptr().add(i) as *const __m256i);
             let w_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
             let w_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));

             let prod_lo = _mm256_mullo_epi32(v_sq_lo, w_lo);
             let prod_hi = _mm256_mullo_epi32(v_sq_hi, w_hi);

             sum_vec = _mm256_add_epi32(sum_vec, _mm256_add_epi32(prod_lo, prod_hi));
        }

        // Process Them (Second 128 inputs to L1)
        for i in (0..LAYER1_SIZE).step_by(16) {
             let v_acc = _mm256_loadu_si256(acc_them.v.as_ptr().add(i) as *const __m256i);
             let v_clamped = _mm256_min_epi16(_mm256_max_epi16(v_acc, zero), max);

             let v_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v_clamped));
             let v_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v_clamped, 1));

             let v_sq_lo = _mm256_mullo_epi32(v_lo, v_lo);
             let v_sq_hi = _mm256_mullo_epi32(v_hi, v_hi);

             let w = _mm256_loadu_si256(net.output_weights.as_ptr().add(LAYER1_SIZE + i) as *const __m256i);
             let w_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(w));
             let w_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(w, 1));

             let prod_lo = _mm256_mullo_epi32(v_sq_lo, w_lo);
             let prod_hi = _mm256_mullo_epi32(v_sq_hi, w_hi);

             sum_vec = _mm256_add_epi32(sum_vec, _mm256_add_epi32(prod_lo, prod_hi));
        }

        let total = hsum_256_epi32(sum_vec) + net.output_bias as i32;
        return total / 4161600;
    }

    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        let mut sum: i32 = net.output_bias as i32;
        // Us
        for i in 0..LAYER1_SIZE {
            let v = acc_us.v[i].clamp(0, 255) as i32;
            let sq = v * v;
            sum += sq * (net.output_weights[i] as i32);
        }
        // Them
        for i in 0..LAYER1_SIZE {
            let v = acc_them.v[i].clamp(0, 255) as i32;
            let sq = v * v;
            sum += sq * (net.output_weights[LAYER1_SIZE + i] as i32);
        }
        return sum / 4161600;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn hsum_256_epi32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let v128 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    let v64 = _mm_add_epi32(v128, _mm_shuffle_epi32(v128, 0b00_00_11_10));
    let v32 = _mm_add_epi32(v64, _mm_shuffle_epi32(v64, 0b00_00_00_01));
    _mm_cvtsi128_si32(v32)
}

pub fn init_nnue(filename: &str) {
    let path = resolve_path(filename);
    println!("Loading NNUE from: {:?}", path);

    match load_nnue_file(&path) {
        Ok(weights) => {
             if let Ok(mut guard) = NNUE.write() {
                *guard = Some(weights);
            }
            println!("NNUE Loaded Successfully (768->128->1 SCReLU).");
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
    let metadata = f.metadata()?;
    let expected_size = (INPUT_SIZE * LAYER1_SIZE + LAYER1_SIZE + 2 * LAYER1_SIZE + 1) as u64 * 2;
    if metadata.len() != expected_size {
         return Err(io::Error::new(io::ErrorKind::InvalidData, format!("File size mismatch. Expected {}, got {}", expected_size, metadata.len())));
    }

    let mut reader = BufReader::new(f);

    let count = INPUT_SIZE * LAYER1_SIZE;
    let mut feature_weights = vec![0i16; count];
    read_i16_buf(&mut reader, &mut feature_weights)?;

    let mut feature_biases = [0i16; LAYER1_SIZE];
    read_i16_buf(&mut reader, &mut feature_biases)?;

    let mut output_weights = [0i16; 2 * LAYER1_SIZE];
    read_i16_buf(&mut reader, &mut output_weights)?;

    let mut output_bias = [0i16; 1];
    read_i16_buf(&mut reader, &mut output_bias)?;

    Ok(NnueWeights {
        feature_biases, feature_weights,
        output_weights, output_bias: output_bias[0],
    })
}

fn read_i16_buf<R: Read>(reader: &mut R, buf: &mut [i16]) -> io::Result<()> {
    let byte_count = buf.len() * 2;
    let ptr = buf.as_mut_ptr() as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, byte_count) };
    reader.read_exact(slice)?;
    if cfg!(target_endian = "big") { for x in buf { *x = x.to_le(); } }
    Ok(())
}
