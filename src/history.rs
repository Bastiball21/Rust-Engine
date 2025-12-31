use std::ops::{Index, IndexMut};

use crate::dense_history::MoveIndexer;

const BUCKETS: usize = 4; // 2 (in_check) * 2 (is_capture)

#[derive(Clone)]
pub struct ContinuationHistory {
    // Flat buffer: [Bucket][PrevMove][CurrMove]
    pub table: Vec<i16>,
    dim: usize,
}

impl ContinuationHistory {
    pub fn new() -> Self {
        let dim = MoveIndexer::max_dense_moves().max(1);
        Self {
            table: vec![0; BUCKETS * dim * dim],
            dim,
        }
    }

    #[inline(always)]
    fn idx(&self, bucket: usize, prev: usize, curr: usize) -> usize {
        // Linear indexing: Bucket -> Prev -> Curr
        (bucket * self.dim + prev) * self.dim + curr
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.table.fill(0);
    }

    /// Get continuation history score for (prev_from->prev_to) followed by (curr_from->curr_to).
    /// Returns 0 if either move is invalid under dense indexing.
    #[inline(always)]
    pub fn get(
        &self,
        in_check: bool,
        is_capture: bool,
        prev_from: usize,
        prev_to: usize,
        curr_from: usize,
        curr_to: usize,
    ) -> i16 {
        let bucket = (in_check as usize) * 2 + (is_capture as usize);

        let Some(prev) = MoveIndexer::get_index(prev_from as u8, prev_to as u8) else {
            return 0;
        };
        let Some(curr) = MoveIndexer::get_index(curr_from as u8, curr_to as u8) else {
            return 0;
        };

        self.table[self.idx(bucket, prev, curr)]
    }

    /// Update continuation history for the (prev move) -> (curr move) pair.
    /// If either move is invalid under dense indexing, does nothing.
    #[inline(always)]
    pub fn update(
        &mut self,
        in_check: bool,
        is_capture: bool,
        prev_from: usize,
        prev_to: usize,
        curr_from: usize,
        curr_to: usize,
        bonus: i32,
    ) {
        let bucket = (in_check as usize) * 2 + (is_capture as usize);

        let Some(prev) = MoveIndexer::get_index(prev_from as u8, prev_to as u8) else {
            return;
        };
        let Some(curr) = MoveIndexer::get_index(curr_from as u8, curr_to as u8) else {
            return;
        };

        let idx = self.idx(bucket, prev, curr);

        // Standard "history update" clamp behavior
        let entry = &mut self.table[idx];
        let val = (*entry as i32) + bonus;
        *entry = val.clamp(-32767, 32767) as i16;
    }
}

impl Index<usize> for ContinuationHistory {
    type Output = i16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.table[index]
    }
}

impl IndexMut<usize> for ContinuationHistory {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.table[index]
    }
}
