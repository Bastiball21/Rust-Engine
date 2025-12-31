use std::io::Write;
use serde::{Deserialize, Serialize};

fn default_lmr_table() -> [[u8; 64]; 64] {
    [[0; 64]; 64]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchParameters {
    // LMR
    pub lmr_base: f64,
    pub lmr_divisor: f64,

    // NMP
    pub nmp_base: i32,
    pub nmp_divisor: i32,

    // RFP
    pub rfp_margin: i32,

    // Razoring
    pub razoring_base: i32,
    pub razoring_multiplier: i32,

    // Tactical bonuses (move ordering)
    pub bonus_safe_check: i32,
    pub bonus_check: i32,
    pub bonus_fork: i32,
    pub bonus_pin: i32,
    pub bonus_skewer: i32,
    pub bonus_discovered: i32,

    // Gating
    pub tactical_topk_quiets: usize,

    // Extension / pruning guard thresholds
    pub extend_safe_check: bool,
    pub extend_near_mate: bool,

    // LMP (Kept as table for now, fixed in struct)
    // We skip serialization for the table to avoid clutter and Default issue,
    // relying on default initialization which matches the hardcoded values.
    #[serde(skip)]
    pub lmp_table: [usize; 16],

    // Precomputed LMR Table
    #[serde(skip, default = "default_lmr_table")]
    pub lmr_table: [[u8; 64]; 64],
}

impl Default for SearchParameters {
    fn default() -> Self {
        let mut params = Self {
            lmr_base: 1.0,
            lmr_divisor: 4.0,
            nmp_base: 3,
            nmp_divisor: 6,
            rfp_margin: 60,
            razoring_base: 300,
            razoring_multiplier: 150,

            bonus_safe_check: 200_000,
            bonus_check: 60_000,
            bonus_fork: 120_000,
            bonus_pin: 90_000,
            bonus_skewer: 90_000,
            bonus_discovered: 70_000,

            tactical_topk_quiets: 20,

            extend_safe_check: true,
            extend_near_mate: true,

            lmp_table: [
                0, 2, 4, 7, 10, 15, 20, 28, 38, 50, 65, 80, 100, 120, 150, 200,
            ],
            lmr_table: [[0; 64]; 64],
        };
        params.recalculate_tables();
        params
    }
}

impl SearchParameters {
    pub fn recalculate_tables(&mut self) {
        // LMR Table
        for d in 0..64 {
            for m in 0..64 {
                if d > 2 && m > 2 {
                    let lmr = self.lmr_base + (d as f64).ln() * (m as f64).ln() / self.lmr_divisor;
                    self.lmr_table[d][m] = lmr.max(0.0) as u8;
                } else {
                    self.lmr_table[d][m] = 0;
                }
            }
        }
    }

    pub fn save_to_json(&self, path: &str) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        let mut file = std::fs::File::create(path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    pub fn load_from_json(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut params: SearchParameters = serde_json::from_reader(reader)?;
        // Ensure tables are computed
        params.recalculate_tables();
        // Reset LMP table to default since it's skipped
        params.lmp_table = [
             0, 2, 4, 7, 10, 15, 20, 28, 38, 50, 65, 80, 100, 120, 150, 200,
        ];
        Ok(params)
    }
}
