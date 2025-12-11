#![allow(non_upper_case_globals)]

use crate::bitboard;
use crate::state::{b, k, n, p, q, r, Move, B, K, N, P, Q, R, WHITE};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

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

pub struct TranspositionTable {
    pub table: *mut Cluster,
    pub count: usize,
    pub generation: AtomicU8, // Tracks the "age" of the search
}

unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl Drop for TranspositionTable {
    fn drop(&mut self) {
        if !self.table.is_null() {
            let cluster_size = std::mem::size_of::<Cluster>();
            let layout = Layout::from_size_align(self.count * cluster_size, 64).unwrap();
            unsafe {
                dealloc(self.table as *mut u8, layout);
            }
        }
    }
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let cluster_size = std::mem::size_of::<Cluster>();
        let desired_bytes = mb * 1024 * 1024;
        let count = desired_bytes / cluster_size;
        let layout = Layout::from_size_align(count * cluster_size, 64).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut Cluster };
        println!("TT: {} MB / {} Clusters / 4-Way Associative", mb, count);
        Self {
            table: ptr,
            count,
            generation: AtomicU8::new(0),
        }
    }

    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.table, 0, self.count);
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    pub fn prefetch(&self, hash: u64) {
        let index = (hash as usize) % self.count;
        unsafe {
            let ptr = self.table.add(index);
            #[cfg(target_arch = "x86_64")]
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    // Pack: Move(16) | Score(16) | Depth(8) | Flag(8) | Age(8) = 56 bits
    fn pack(score: i32, depth: u8, flag: u8, age: u8, mv: Option<Move>) -> u64 {
        let move_u16 = if let Some(m) = mv {
            let promo_bits = match m.promotion {
                Some(Q) | Some(q) => 1,
                Some(R) | Some(r) => 2,
                Some(B) | Some(b) => 3,
                Some(N) | Some(n) => 4,
                _ => 0,
            };
            ((m.source as u16) << 6) | (m.target as u16) | (promo_bits << 12)
        } else {
            0
        };

        let score_u16 = (score.clamp(-32000, 32000) + 32000) as u16;

        (move_u16 as u64)
            | ((score_u16 as u64) << 16)
            | ((depth as u64) << 32)
            | ((flag as u64) << 40)
            | ((age as u64) << 48)
    }

    // Unpack: Returns (score, depth, flag, age, move)
    fn unpack(data: u64) -> (i32, u8, u8, u8, Option<Move>) {
        let move_u16 = (data & 0xFFFF) as u16;
        let score = ((data >> 16) & 0xFFFF) as i32 - 32000;
        let depth = ((data >> 32) & 0xFF) as u8;
        let flag = ((data >> 40) & 0xFF) as u8;
        let age = ((data >> 48) & 0xFF) as u8;

        let mv = if move_u16 != 0 {
            let from = (move_u16 >> 6) & 0x3F;
            let to = move_u16 & 0x3F;
            let promo_bits = (move_u16 >> 12) & 0xF;
            let promotion = match promo_bits {
                1 => Some(Q),
                2 => Some(R),
                3 => Some(B),
                4 => Some(N),
                _ => None,
            };
            if from != to {
                Some(Move {
                    source: from as u8,
                    target: to as u8,
                    promotion,
                    is_capture: false,
                })
            } else {
                None
            }
        } else {
            None
        };

        (score, depth, flag, age, mv)
    }

    pub fn store(&self, hash: u64, score: i32, best_move: Option<Move>, depth: u8, flag: u8) {
        let index = (hash as usize) % self.count;
        let current_gen = self.generation.load(Ordering::Relaxed);
        let new_data = Self::pack(score, depth, flag, current_gen, best_move);

        unsafe {
            let cluster = &*self.table.add(index);
            let mut best_victim_idx = 0;
            let mut min_score = i32::MAX;

            for i in 0..4 {
                let entry = &cluster.entries[i];
                let key = entry.key.load(Ordering::Relaxed);

                // 1. Found exact match? Update it.
                if key == hash {
                    let old_data = entry.data.load(Ordering::Relaxed);
                    let (_, _, _, _, old_move) = Self::unpack(old_data);

                    // Keep old move if new one is missing
                    let final_move = if best_move.is_none() {
                        old_move
                    } else {
                        best_move
                    };
                    let final_data = Self::pack(score, depth, flag, current_gen, final_move);

                    entry.data.store(final_data, Ordering::Relaxed);
                    // entry.age is implicitly updated by pack() using current_gen
                    return;
                }

                // 2. Found empty slot? Take it.
                if key == 0 {
                    entry.key.store(hash, Ordering::Relaxed);
                    entry.data.store(new_data, Ordering::Relaxed);
                    return;
                }

                // 3. Evaluate replacement candidates (Aging logic)
                let data = entry.data.load(Ordering::Relaxed);
                let (_, d, _, age, _) = Self::unpack(data);

                // Calculate "Value": Deep entries are valuable, Old entries lose value.
                // Age diff is wrapped u8, handled by masking.
                let age_diff = current_gen.wrapping_sub(age);
                let replace_score = (d as i32) - (age_diff as i32 * 2);

                if replace_score < min_score {
                    min_score = replace_score;
                    best_victim_idx = i;
                }
            }

            // 4. Replace the worst entry if our new entry is better (or if we just force it)
            // Here we use a strategy: Always replace the worst candidate in the bucket.
            let entry = &cluster.entries[best_victim_idx];
            entry.key.store(hash, Ordering::Relaxed);
            entry.data.store(new_data, Ordering::Relaxed);
        }
    }

    pub fn probe_data(&self, hash: u64) -> Option<(i32, u8, u8, Option<Move>)> {
        let index = (hash as usize) % self.count;
        unsafe {
            let cluster = &*self.table.add(index);
            for i in 0..4 {
                let entry = &cluster.entries[i];
                if entry.key.load(Ordering::Relaxed) == hash {
                    let data = entry.data.load(Ordering::Relaxed);
                    let (score, depth, flag, _, mv) = Self::unpack(data);
                    return Some((score, depth, flag, mv));
                }
            }
        }
        None
    }

    pub fn get_move(&self, hash: u64) -> Option<Move> {
        self.probe_data(hash).and_then(|(_, _, _, m)| m)
    }

    pub fn is_pseudo_legal(&self, state: &crate::state::GameState, mv: Move) -> bool {
        let from = mv.source;
        let to = mv.target;
        let side = state.side_to_move;
        let occ = state.occupancies[2]; // BOTH

        if from >= 64 || to >= 64 || from == to {
            return false;
        }

        let mut piece_type = 12;
        let start = if side == WHITE { P } else { p };
        let end = if side == WHITE { K } else { k };

        for piece in start..=end {
            if state.bitboards[piece].get_bit(from) {
                piece_type = piece;
                break;
            }
        }
        if piece_type == 12 {
            return false;
        }

        // Basic capture check (own piece)
        if state.occupancies[side].get_bit(to) {
            return false;
        }

        match piece_type {
            N | n => bitboard::mask_knight_attacks(from).get_bit(to),
            K | k => {
                let attacks = bitboard::mask_king_attacks(from);
                if attacks.get_bit(to) {
                    return true;
                }
                if (from as i8 - to as i8).abs() == 2 {
                    return true;
                }
                false
            }
            P => {
                if to == from + 8 && !occ.get_bit(to) {
                    return true;
                }
                if from >= 8
                    && from <= 15
                    && to == from + 16
                    && !occ.get_bit(from + 8)
                    && !occ.get_bit(to)
                {
                    return true;
                }
                if (to == from + 7 || to == from + 9)
                    && (state.occupancies[1].get_bit(to) || to == state.en_passant)
                {
                    return true;
                }
                false
            }
            p => {
                if to == from.wrapping_sub(8) && !occ.get_bit(to) {
                    return true;
                }
                if from >= 48
                    && from <= 55
                    && to == from.wrapping_sub(16)
                    && !occ.get_bit(from.wrapping_sub(8))
                    && !occ.get_bit(to)
                {
                    return true;
                }
                if (to == from.wrapping_sub(7) || to == from.wrapping_sub(9))
                    && (state.occupancies[0].get_bit(to) || to == state.en_passant)
                {
                    return true;
                }
                false
            }
            R | r => bitboard::get_rook_attacks(from, occ).get_bit(to),
            B | b => bitboard::get_bishop_attacks(from, occ).get_bit(to),
            Q | q => bitboard::get_queen_attacks(from, occ).get_bit(to),
            _ => false,
        }
    }

    pub fn hashfull(&self) -> usize {
        let mut used = 0;
        let scan_limit = if self.count > 1000 { 1000 } else { self.count };
        for i in 0..scan_limit {
            unsafe {
                let cluster = &*self.table.add(i);
                for entry in &cluster.entries {
                    if entry.key.load(Ordering::Relaxed) != 0 {
                        used += 1;
                    }
                }
            }
        }
        (used * 1000) / (scan_limit * 4)
    }
}
