use crate::state::Move;
use std::alloc::{alloc_zeroed, Layout};
use std::sync::atomic::{AtomicU64, Ordering};

// --- PACKED ENTRY (Atomic) ---
pub struct TTEntry {
    pub key: AtomicU64,  
    pub data: AtomicU64, 
}

pub const FLAG_NONE: u8 = 0;
pub const FLAG_EXACT: u8 = 1;
pub const FLAG_ALPHA: u8 = 2;
pub const FLAG_BETA: u8 = 3;

#[repr(C, align(64))] 
pub struct Cluster {
    pub entries: [TTEntry; 4],
}

// The Wrapper
pub struct TranspositionTable {
    pub table: *mut Cluster,
    pub count: usize,
}

// CRITICAL: Tell Rust "I promise this raw pointer is safe to share across threads"
unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let cluster_size = std::mem::size_of::<Cluster>();
        let desired_bytes = mb * 1024 * 1024;
        let count = desired_bytes / cluster_size;
        
        let layout = Layout::from_size_align(count * cluster_size, 64).unwrap();
        // We use alloc_zeroed so the AtomicU64s start at 0
        let ptr = unsafe { alloc_zeroed(layout) as *mut Cluster };

        println!("TT: {} MB / {} Clusters / Shared Atomic Access", mb, count);
        Self { table: ptr, count }
    }

    pub fn prefetch(&self, hash: u64) {
        let index = (hash as usize) % self.count;
        unsafe {
            let ptr = self.table.add(index);
            #[cfg(target_arch = "x86_64")]
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    pub fn store(&self, hash: u64, score: i32, best_move: Option<Move>, depth: u8, flag: u8) {
        let index = (hash as usize) % self.count;
        let move_u16 = if let Some(m) = best_move {
            ((m.source as u16) << 6) | (m.target as u16)
        } else { 0 };
        
        let score_u16 = (score + 32000) as u16; 
        let data_val = (move_u16 as u64) | ((score_u16 as u64) << 16) | ((depth as u64) << 32) | ((flag as u64) << 40);

        unsafe {
            let cluster = &*self.table.add(index);
            let mut replace_idx = 0;
            let mut min_depth = 255;

            // We use Relaxed ordering for speed. In chess, if we read stale data,
            // it's not a crash, just a slightly worse search.
            for i in 0..4 {
                let entry = &cluster.entries[i];
                let key = entry.key.load(Ordering::Relaxed);
                
                if key == hash {
                    // Found match, overwrite logic
                    // (Simple overwrite for now to avoid CompareExchange loops)
                    entry.data.store(data_val, Ordering::Relaxed);
                    return;
                }
                
                let data = entry.data.load(Ordering::Relaxed);
                let d = (data >> 32) as u8;
                if d < min_depth {
                    min_depth = d;
                    replace_idx = i;
                }
            }

            if depth >= min_depth {
                let entry = &cluster.entries[replace_idx];
                entry.key.store(hash, Ordering::Relaxed);
                entry.data.store(data_val, Ordering::Relaxed);
            }
        }
    }

    pub fn probe(&self, hash: u64, depth: u8, alpha: i32, beta: i32) -> Option<i32> {
        let index = (hash as usize) % self.count;
        unsafe {
            let cluster = &*self.table.add(index);
            for i in 0..4 {
                let entry = &cluster.entries[i];
                // Atomic Load
                let key = entry.key.load(Ordering::Relaxed);
                
                if key == hash {
                    let data = entry.data.load(Ordering::Relaxed);
                    // Verify key again to avoid "tearing" (race condition where key changed but data didn't)
                    if entry.key.load(Ordering::Relaxed) != hash { continue; }

                    let entry_depth = (data >> 32) as u8;
                    if entry_depth >= depth {
                        let entry_score = ((data >> 16) as u16) as i32 - 32000;
                        let entry_flag = (data >> 40) as u8;
                        match entry_flag {
                            FLAG_EXACT => return Some(entry_score),
                            FLAG_ALPHA => if entry_score <= alpha { return Some(alpha); },
                            FLAG_BETA  => if entry_score >= beta  { return Some(beta); },
                            _ => {}
                        }
                    }
                }
            }
        }
        None
    }

    pub fn get_move(&self, hash: u64) -> Option<Move> {
        let index = (hash as usize) % self.count;
        unsafe {
            let cluster = &*self.table.add(index);
            for i in 0..4 {
                let entry = &cluster.entries[i];
                if entry.key.load(Ordering::Relaxed) == hash {
                    let data = entry.data.load(Ordering::Relaxed);
                    let move_u16 = (data & 0xFFFF) as u16;
                    if move_u16 == 0 { return None; }
                    let from = (move_u16 >> 6) as u8;
                    let to = (move_u16 & 0x3F) as u8;
                    return Some(Move { source: from, target: to, promotion: None, is_capture: false });
                }
            }
        }
        None
    }

    pub fn hashfull(&self) -> usize {
        let mut used = 0;
        let scan_limit = if self.count > 1000 { 1000 } else { self.count };
        for i in 0..scan_limit {
            unsafe {
                let cluster = &*self.table.add(i);
                for entry in &cluster.entries {
                    if entry.key.load(Ordering::Relaxed) != 0 { used += 1; }
                }
            }
        }
        (used * 1000) / (scan_limit * 4)
    }
}