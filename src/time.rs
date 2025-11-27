use std::time::Instant;

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

#[derive(Clone, Copy)]
pub struct TimeManager {
    pub start_time: Instant,
    pub hard_limit: u128,
    pub soft_limit: u128,
}

impl TimeManager {
    pub fn new(limit: TimeControl, side: usize, overhead: u128) -> Self {
        let start_time = Instant::now();
        let (mut hard, mut soft) = match limit {
            TimeControl::Infinite => (u128::MAX, u128::MAX),
            TimeControl::MoveTime(t) => {
                // If movetime is very small, don't subtract overhead to avoid underflow/zero
                let effective_t = t.saturating_sub(overhead);
                if effective_t == 0 { (t, t) } else { (effective_t, effective_t) }
            },
            TimeControl::GameTime { wtime, btime, winc, binc, moves_to_go } => {
                let (mut time, inc) = if side == 0 { (wtime, winc) } else { (btime, binc) };
                
                // Subtract overhead from current time to be safe
                time = time.saturating_sub(overhead);
                if time == 0 { time = 50; } // Emergency buffer if less than overhead

                let mtg = moves_to_go.unwrap_or(40).clamp(20, 50) as u128;
                
                // Basic time management formula
                let base = (time / mtg) + (inc * 3 / 4);
                
                // Don't use more than 80% of remaining time
                let max_alloc = time * 8 / 10;
                
                let soft = base.min(max_alloc);
                let hard = (base * 2).min(max_alloc);
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
    pub fn check_soft_limit(&self) -> bool {
        self.start_time.elapsed().as_millis() >= self.soft_limit
    }

    #[inline(always)]
    pub fn check_hard_limit(&self) -> bool {
        self.start_time.elapsed().as_millis() >= self.hard_limit
    }
}