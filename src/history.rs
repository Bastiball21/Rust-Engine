
use std::ops::{Index, IndexMut};

const C_DIM: usize = 768; // 12 pieces * 64 squares
const BUCKETS: usize = 4; // 2 (in_check) * 2 (is_capture)

#[derive(Clone)]
pub struct ContinuationHistory {
    // Flat buffer: [Bucket][PrevMove][CurrMove]
    pub table: Vec<i16>,
}

impl ContinuationHistory {
    pub fn new() -> Self {
        Self {
            table: vec![0; BUCKETS * C_DIM * C_DIM],
        }
    }

    #[inline(always)]
    fn idx(bucket: usize, prev: usize, curr: usize) -> usize {
        // Linear indexing: Bucket -> Prev -> Curr
        (bucket * C_DIM + prev) * C_DIM + curr
    }

    #[inline(always)]
    pub fn get(&self, in_check: bool, is_capture: bool, prev_piece: usize, prev_to: usize, curr_piece: usize, curr_to: usize) -> i16 {
        // Explicit bucket mapping: (Check * 2) + Capture
        let bucket = (in_check as usize) * 2 + (is_capture as usize);
        let prev = prev_piece * 64 + prev_to;
        let curr = curr_piece * 64 + curr_to;
        self.table[Self::idx(bucket, prev, curr)]
    }

    pub fn update(&mut self, in_check: bool, is_capture: bool, prev_piece: usize, prev_to: usize, curr_piece: usize, curr_to: usize, bonus: i32) {
        let bucket = (in_check as usize) * 2 + (is_capture as usize);
        let prev = prev_piece * 64 + prev_to;
        let curr = curr_piece * 64 + curr_to;

        let index = Self::idx(bucket, prev, curr);
        // SAFETY: We trust indices are within bounds (0..12, 0..64)
        debug_assert!(index < self.table.len());
        let entry = &mut self.table[index];

        // Gravity Update: entry += bonus - (entry * |bonus|) / 16384
        let penalty = (*entry as i32 * bonus.abs()) / 16384;
        let new_val = (*entry as i32 + bonus - penalty).clamp(-16000, 16000);
        *entry = new_val as i16;
    }

    pub fn clear(&mut self) {
        self.table.fill(0);
    }
}
