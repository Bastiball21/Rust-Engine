#![allow(non_upper_case_globals)]

use crate::bitboard;
use crate::state::{b, k, n, p, q, r, GameState, Move, B, K, N, P, Q, R, WHITE, NO_PIECE};
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
            let promo_bits = match m.promotion() {
                Some(Q) | Some(q) => 1,
                Some(R) | Some(r) => 2,
                Some(B) | Some(b) => 3,
                Some(N) | Some(n) => 4,
                _ => 0,
            };
            ((m.source() as u16) << 6) | (m.target() as u16) | (promo_bits << 12)
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
                Some(Move::new(
                    from as u8,
                    to as u8,
                    promotion,
                    false, // Will be fixed by call sites via fix_move
                ))
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

                    // Use Release to ensure the update is visible
                    entry.data.store(final_data, Ordering::Release);
                    // entry.age is implicitly updated by pack() using current_gen
                    return;
                }

                // 2. Found empty slot? Take it.
                if key == 0 {
                    // CRITICAL FIX: Write Data FIRST, then Key.
                    // Use Release on the Key to ensure the Data write is visible to readers.
                    entry.data.store(new_data, Ordering::Relaxed);
                    entry.key.store(hash, Ordering::Release);
                    return;
                }

                // 3. Evaluate replacement candidates (Aging logic)
                let data = entry.data.load(Ordering::Relaxed);
                let (_, d, _, age, _) = Self::unpack(data);

                // "Depth-Preferred" Replacement Strategy
                // 1. If we are deeper than the existing entry, we almost always want to replace it.
                // 2. If the existing entry is old (age_diff > 0), its value drops significantly.
                // 3. Score = Depth - (Age * Boost)

                let age_diff = current_gen.wrapping_sub(age);
                let replace_score = (d as i32) - (age_diff as i32 * 4); // Boost age penalty

                if replace_score < min_score {
                    min_score = replace_score;
                    best_victim_idx = i;
                }
            }

            // 4. Replace strategy
            // We replace if:
            // A. The victim is empty (key == 0) - handled above
            // B. The new entry is deeper than the victim's score (conceptually)
            // C. The victim is the "worst" in the cluster
            // To avoid thrashing deep PV nodes with shallow searches, we check:
            // New Depth > Min Score? Or just always replace the worst?
            // Let's replace the worst, but ONLY if the new entry is "better" or the victim is old.

            let entry = &cluster.entries[best_victim_idx];
            let old_data = entry.data.load(Ordering::Relaxed);
            let (_, old_d, _, old_age, _) = Self::unpack(old_data);
            let old_age_diff = current_gen.wrapping_sub(old_age);

            // Allow replacement if:
            // 1. New depth >= Old depth
            // 2. Old entry is old (age_diff > 0)
            // 3. New depth is reasonably close to old depth (e.g. old_d - new_d < 5)
            if depth >= old_d || old_age_diff > 0 || (old_d < depth + 5) {
                // CRITICAL FIX: Write Data FIRST, then Key.
                entry.data.store(new_data, Ordering::Relaxed);
                entry.key.store(hash, Ordering::Release);
            }
        }
    }

    pub fn probe_data(&self, hash: u64, state: &GameState) -> Option<(i32, u8, u8, Option<Move>)> {
        let index = (hash as usize) % self.count;
        unsafe {
            let cluster = &*self.table.add(index);
            for i in 0..4 {
                let entry = &cluster.entries[i];

                // CRITICAL FIX: Load Key with Acquire.
                // This guarantees we see the data associated with this key.
                if entry.key.load(Ordering::Acquire) == hash {
                    let data = entry.data.load(Ordering::Relaxed);
                    let (score, depth, flag, _, mut mv) = Self::unpack(data);

                    if let Some(ref mut m) = mv {
                        Self::fix_move(m, state);
                    }

                    return Some((score, depth, flag, mv));
                }
            }
        }
        None
    }

    pub fn get_move(&self, hash: u64, state: &GameState) -> Option<Move> {
        self.probe_data(hash, state).and_then(|(_, _, _, m)| m)
    }

    // Since `Move` is now Copy/Clone/Immutable mostly (we use `Move(u16)`),
    // and `is_capture` is packed, we can't easily mutate it in place unless we recreate it.
    // Wait, `mv` in `fix_move` is `&mut Move`. I need to handle that.
    // The previous `Move` struct had pub fields. The new one is opaque.
    // I can't set `mv.is_capture = true`.
    // I need to add a method to `Move` or recreate it.
    // Since `Move` is just a u16, I can recreate it.
    // But `fix_move` takes `&mut Move`.
    // I should probably add a `set_capture` method to `Move` or similar, OR change `fix_move` to return a new `Move`.
    // Let's modify `Move` in `src/state.rs` to allow mutation or reconstruction?
    // Actually, I can just update the underlying u16 if I knew the layout.
    // But it's cleaner to just replace the value. `*mv = Move::new(...)`.

    fn fix_move(mv: &mut Move, state: &GameState) {
        let to = mv.target() as usize;
        let captured = state.board[to];
        let mut is_capture = false;

        if captured != NO_PIECE as u8 {
            is_capture = true;
        } else if mv.target() == state.en_passant {
            // Check if it's a pawn move
            let piece = state.board[mv.source() as usize];
            if piece == P as u8 || piece == p as u8 {
                is_capture = true;
            }
        }

        // Reconstruct move with correct capture flag
        *mv = Move::new(mv.source(), mv.target(), mv.promotion(), is_capture);
    }

    pub fn is_pseudo_legal(&self, state: &crate::state::GameState, mv: Move) -> bool {
        let from = mv.source();
        let to = mv.target();
        let side = state.side_to_move;

        if from >= 64 || to >= 64 || from == to {
            return false;
        }

        // MAILBOX OPTIMIZATION & STRICT VALIDATION
        let piece_type = state.board[from as usize] as usize;
        if piece_type == 12 {
            return false;
        }

        // Verify with Bitboards (Hardening against Board Desync)
        if !state.bitboards[piece_type].get_bit(from) {
            return false;
        }

        // Validate Side ownership
        // White pieces: 0..5, Black pieces: 6..11
        if side == WHITE {
             if piece_type > K { return false; }
        } else {
             if piece_type < p || piece_type > k { return false; }
        }

        let target_piece = state.board[to as usize] as usize;

        // Basic capture check (own piece)
        if target_piece != 12 {
             if side == WHITE {
                  if target_piece <= K { return false; }
             } else {
                  if target_piece >= p && target_piece <= k { return false; }
             }
        }

        // STRICT CAPTURE FLAG VALIDATION
        let is_occupied = target_piece != 12;

        // Verify Victim with Bitboards (Hardening)
        if is_occupied && !state.bitboards[target_piece].get_bit(to) {
            return false; // Ghost Victim
        }

        let is_ep = to == state.en_passant && (piece_type == P || piece_type == p);

        if mv.is_capture() {
            if !is_occupied && !is_ep {
                return false; // Ghost Capture
            }
        } else {
            if is_occupied {
                return false; // State Corruption (Capture as Non-Capture)
            }
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
                if to == from + 8 && !is_occupied {
                    return true;
                }
                if from >= 8
                    && from <= 15
                    && to == from + 16
                    && !state.occupancies[2].get_bit(from + 8)
                    && !is_occupied
                {
                    return true;
                }
                if (to == from + 7 || to == from + 9)
                    && (is_occupied || to == state.en_passant)
                {
                    return true;
                }
                false
            }
            p => {
                if to == from.wrapping_sub(8) && !is_occupied {
                    return true;
                }
                if from >= 48
                    && from <= 55
                    && to == from.wrapping_sub(16)
                    && !state.occupancies[2].get_bit(from.wrapping_sub(8))
                    && !is_occupied
                {
                    return true;
                }
                if (to == from.wrapping_sub(7) || to == from.wrapping_sub(9))
                    && (is_occupied || to == state.en_passant)
                {
                    return true;
                }
                false
            }
            R | r => bitboard::get_rook_attacks(from, state.occupancies[2]).get_bit(to),
            B | b => bitboard::get_bishop_attacks(from, state.occupancies[2]).get_bit(to),
            Q | q => bitboard::get_queen_attacks(from, state.occupancies[2]).get_bit(to),
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
