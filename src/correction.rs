#[derive(Clone)]
pub struct CorrectionHistory {
    // [side][index] where index = pawn_key & 16383
    data: [[i16; 16384]; 2],
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self { data: [[0; 16384]; 2] }
    }
}

impl CorrectionHistory {
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    fn idx(pawn_key: u64) -> usize {
        (pawn_key as usize) & 0x3FFF // 16384 entries
    }

    /// Get the current correction (centipawns).
    #[inline(always)]
    pub fn get(&self, pawn_key: u64, side: usize) -> i32 {
        debug_assert!(side < 2);
        self.data[side][Self::idx(pawn_key)] as i32
    }

    /// Update correction based on observed error: best_score - static_eval.
    ///
    /// We interpolate towards the new error with a depth-dependent weight.
    /// Values are clamped to +/-512 to avoid instability.
    #[inline(always)]
    pub fn update(&mut self, pawn_key: u64, side: usize, static_eval: i32, best_score: i32, depth: u8) {
        debug_assert!(side < 2);

        let idx = Self::idx(pawn_key);

        // Error in centipawns
        let mut error = best_score - static_eval;

        // Clamp the target error to keep updates stable
        error = error.clamp(-512, 512);

        let cur = self.data[side][idx] as i32;

        // Weight: deeper nodes get a slightly stronger update (1..16).
        // Equivalent to: cur + (error - cur) * weight / 16
        let w = (depth as i32).clamp(1, 16);
        let newv = cur + ((error - cur) * w) / 16;

        self.data[side][idx] = (newv.clamp(-512, 512) as i16);
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data = [[0; 16384]; 2];
    }
}
