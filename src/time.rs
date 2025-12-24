use std::time::Instant;
use crate::state::Move;

#[derive(Clone, Copy)]
pub enum TimeControl {
    Infinite,
    MoveTime(u128),
    GameTime {
        wtime: u128,
        btime: u128,
        winc: u128,
        binc: u128,
        moves_to_go: Option<u32>,
    },
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum ForcedMoveType {
    None,
    Weak,
    Strong,
    OneLegal,
}

#[derive(Clone, Copy)]
pub struct TimeManager {
    pub start_time: Instant,
    pub hard_limit: u128,
    pub soft_limit: u128,
    pub base_soft_limit: u128, // Store original base to apply multipliers to
    pub max_time: u128,

    // Search State for Time Management
    pub prev_score: i32,
    pub prev_move: Option<Move>,
    pub stability: usize,
    pub failed_low: i32,
    pub found_forced_move: ForcedMoveType,
}

impl TimeManager {
    pub fn new(limit: TimeControl, side: usize, overhead: u128) -> Self {
        let start_time = Instant::now();
        let mut max_time = u128::MAX;

        let (hard, soft) = match limit {
            TimeControl::Infinite => (u128::MAX, u128::MAX),
            TimeControl::MoveTime(t) => {
                let effective_t = t.saturating_sub(overhead);
                let val = if effective_t == 0 { t } else { effective_t };
                max_time = val;
                (val, val)
            }
            TimeControl::GameTime {
                wtime,
                btime,
                winc,
                binc,
                moves_to_go,
            } => {
                let (mut time, inc) = if side == 0 {
                    (wtime, winc)
                } else {
                    (btime, binc)
                };

                // Max time we ever want to spend is 95% of current time - overhead
                max_time = (time * 95 / 100).saturating_sub(overhead);

                time = time.saturating_sub(overhead);
                if time == 0 {
                    time = 50;
                }

                let mtg = moves_to_go.unwrap_or(24).clamp(2, 50) as u128;

                // Basic time management formula
                // Aether default was roughly time/20 + inc/2
                // Viridithas uses a more complex window calculation.
                // We'll stick to a robust default:
                // Base = (Time / MovesToGo) + Inc
                let base = (time / mtg) + (inc * 3 / 4);

                // Hard limit: roughly 5x base or 50% of time?
                // Viridithas uses hard_window_frac = 46%.
                let hard_cap = time * 46 / 100;
                let hard = (base * 4).min(hard_cap).min(max_time);

                // Optimal (soft) limit: roughly 70% of base?
                // Viridithas uses optimal_window_frac = 73%
                let soft = (base * 73 / 100).min(hard);

                (hard, soft)
            }
        };

        Self {
            start_time,
            hard_limit: hard,
            soft_limit: soft,
            base_soft_limit: soft,
            max_time,
            prev_score: 0,
            prev_move: None,
            stability: 0,
            failed_low: 0,
            found_forced_move: ForcedMoveType::None,
        }
    }

    pub fn set_stability_factor(&mut self, factor: f64) {
        // Legacy support if needed, but report_completed_depth handles this better now
        let f = factor.clamp(0.5, 4.0);
        let new_soft = (self.base_soft_limit as f64 * f) as u128;
        self.soft_limit = new_soft.min(self.hard_limit);
    }

    pub fn report_completed_depth(&mut self, depth: i32, eval: i32, best_move: Move) {
        if self.hard_limit == u128::MAX { return; } // Infinite time

        if Some(best_move) == self.prev_move {
            self.stability += 1;
        } else {
            self.stability = 0;
        }

        // Stability Multiplier
        let stability_mult = match self.stability.min(4) {
            0 => 2.50,
            1 => 1.20,
            2 => 0.90,
            3 => 0.80,
            _ => 0.75,
        };

        // Failed Low Bonus (0.34 * failed_low)
        let fail_low_mult = 1.0 + (self.failed_low as f64 * 0.34);

        // Forced Move Multiplier
        let forced_mult = match self.found_forced_move {
            ForcedMoveType::OneLegal => 0.01,
            ForcedMoveType::Strong => 0.386,
            ForcedMoveType::Weak => 0.627,
            ForcedMoveType::None => 1.0,
        };

        let total_mult = stability_mult * fail_low_mult * forced_mult;

        // Apply to base soft limit
        let new_soft = (self.base_soft_limit as f64 * total_mult) as u128;
        let new_hard = (self.hard_limit as f64 * total_mult).min(self.max_time as f64) as u128;

        self.soft_limit = new_soft.min(new_hard);
        self.hard_limit = new_hard; // We update hard limit too based on Viridithas logic (scaling window)

        self.prev_move = Some(best_move);
        self.prev_score = eval;
    }

    pub fn report_aspiration_fail(&mut self, bound: u8) {
         // Bound::Upper is 2 (FLAG_ALPHA) ?
         // Viridithas: if bound == Bound::Upper (Fail Low) -> failed_low++
         // TT flags: 2=Alpha (Fail Low), 3=Beta (Fail High), 1=Exact
         // In Aether: FLAG_ALPHA (2) means score <= alpha (Fail Low)
         if bound == 2 {
             self.failed_low += 1;
         }
    }

    pub fn check_for_forced_move(&mut self, depth: i32) -> Option<i32> {
        if self.found_forced_move == ForcedMoveType::None && self.hard_limit != u128::MAX {
            if depth >= 12 {
                return Some(400); // Strong forced check margin
            } else if depth >= 8 {
                 return Some(170); // Weak forced check margin
            }
        }
        None
    }

    pub fn report_forced_move(&mut self, depth: i32) {
        if depth >= 12 {
            self.found_forced_move = ForcedMoveType::Strong;
        } else {
            self.found_forced_move = ForcedMoveType::Weak;
        }
    }

    pub fn notify_one_legal_move(&mut self) {
        self.found_forced_move = ForcedMoveType::OneLegal;
        self.soft_limit = 0; // Stop immediately
    }

    #[inline(always)]
    pub fn check_soft_limit(&self) -> bool {
        self.start_time.elapsed().as_millis() >= self.soft_limit
    }

    #[inline(always)]
    pub fn check_hard_limit(&self) -> bool {
        self.start_time.elapsed().as_millis() >= self.hard_limit
    }
}
