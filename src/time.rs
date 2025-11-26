use std::time::{Duration, Instant};

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

pub struct TimeManager {
    pub start_time: Instant, // Made public to fix E0616
    pub hard_limit: u128,    // Made public
    pub soft_limit: u128,    // Made public
}

impl TimeManager {
    pub fn new(limit: TimeControl, side: usize) -> Self {
        let start_time = Instant::now();
        let (hard, soft) = match limit {
            TimeControl::Infinite => (u128::MAX, u128::MAX),
            TimeControl::MoveTime(t) => (t.saturating_sub(50), t.saturating_sub(50)), // Safety buffer
            TimeControl::GameTime { wtime, btime, winc, binc, moves_to_go } => {
                let (time, inc) = if side == 0 { (wtime, winc) } else { (btime, binc) };
                
                // Stockfish/Viridithas style heuristics
                let mtg = moves_to_go.unwrap_or(40).clamp(20, 50) as u128;
                
                // Basic calculation: Time / Moves + Increment
                let base = (time / mtg) + (inc * 3 / 4);
                
                // Safety: Never use more than 80% of remaining time
                let max_alloc = time * 8 / 10;
                
                let soft = base.min(max_alloc);
                let hard = (base * 2).min(max_alloc); // Allow up to 2x for critical moves
                
                (hard, soft)
            }
        };

        Self {
            start_time,
            hard_limit: hard,
            soft_limit: soft,
        }
    }

    #[inline(always)]
    pub fn check_hard_limit(&self) -> bool {
        if self.hard_limit == u128::MAX { return false; }
        self.start_time.elapsed().as_millis() >= self.hard_limit
    }

    #[inline(always)]
    pub fn check_soft_limit(&self) -> bool {
        if self.soft_limit == u128::MAX { return false; }
        self.start_time.elapsed().as_millis() >= self.soft_limit
    }
}