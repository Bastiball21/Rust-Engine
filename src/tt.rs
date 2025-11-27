use crate::state::{Move, Q, R, B, N, P, K, q, r, b, n, p, k, WHITE}; 
use std::alloc::{alloc_zeroed, Layout};
use std::sync::atomic::{AtomicU64, Ordering};
use crate::bitboard; 

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
}

unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let cluster_size = std::mem::size_of::<Cluster>();
        let desired_bytes = mb * 1024 * 1024;
        let count = desired_bytes / cluster_size;
        let layout = Layout::from_size_align(count * cluster_size, 64).unwrap();
        let ptr = unsafe { alloc_zeroed(layout) as *mut Cluster };
        println!("TT: {} MB / {} Clusters / Shared Atomic Access", mb, count);
        Self { table: ptr, count }
    }

    pub fn clear(&mut self) {
        unsafe { std::ptr::write_bytes(self.table, 0, self.count); }
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
            let promo_bits = match m.promotion {
                Some(Q) | Some(q) => 1, Some(R) | Some(r) => 2, Some(B) | Some(b) => 3, Some(N) | Some(n) => 4, _ => 0
            };
            ((m.source as u16) << 6) | (m.target as u16) | (promo_bits << 12)
        } else { 0 };

        let score_u16 = (score + 32000) as u16; 
        let data_val = (move_u16 as u64) | ((score_u16 as u64) << 16) | ((depth as u64) << 32) | ((flag as u64) << 40);

        unsafe {
            let cluster = &*self.table.add(index);
            let mut replace_idx = 0;
            let mut min_depth = 255;
            for i in 0..4 {
                let entry = &cluster.entries[i];
                let key = entry.key.load(Ordering::Relaxed);
                if key == hash {
                    entry.data.store(data_val, Ordering::Relaxed);
                    return;
                }
                let data = entry.data.load(Ordering::Relaxed);
                let d = (data >> 32) as u8;
                if d < min_depth { min_depth = d; replace_idx = i; }
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
                if entry.key.load(Ordering::Relaxed) == hash {
                    let data = entry.data.load(Ordering::Relaxed);
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
                    let from = (move_u16 >> 6) & 0x3F;
                    let to = move_u16 & 0x3F;
                    if from == to { return None; } 
                    let promo_bits = (move_u16 >> 12) & 0xF;
                    let promotion = match promo_bits {
                        1 => Some(Q), 2 => Some(R), 3 => Some(B), 4 => Some(N), _ => None
                    };
                    return Some(Move { source: from as u8, target: to as u8, promotion, is_capture: false });
                }
            }
        }
        None
    }
    
    // --- FAST MOVE VALIDATION (Pseudo-Legal Check) ---
    // Essential for preventing illegal moves from hash collisions without slow generation
    pub fn is_pseudo_legal(&self, state: &crate::state::GameState, mv: Move) -> bool {
        let from = mv.source;
        let to = mv.target;
        let side = state.side_to_move;
        let occ = state.occupancies[2];

        let mut piece_type = 12;
        let start = if side == WHITE { P } else { p };
        let end = if side == WHITE { K } else { k };
        
        // Renamed loop variable to 'piece' to avoid conflict
        for piece in start..=end {
            if state.bitboards[piece].get_bit(from) { piece_type = piece; break; }
        }
        if piece_type == 12 { return false; }

        if state.occupancies[side].get_bit(to) { return false; }

        match piece_type {
            N | n => {
                let attacks = bitboard::mask_knight_attacks(from);
                if !attacks.get_bit(to) { return false; }
            },
            K | k => {
                let attacks = bitboard::mask_king_attacks(from);
                if !attacks.get_bit(to) { 
                    // Allow castling jump distance in pseudo-check
                    if (from as i8 - to as i8).abs() == 2 { return true; } 
                    return false; 
                }
            },
            P => { 
                if to == from + 8 && !occ.get_bit(to) { return true; }
                if from >= 8 && from <= 15 && to == from + 16 && !occ.get_bit(from+8) && !occ.get_bit(to) { return true; }
                if (to == from + 7 || to == from + 9) && (state.occupancies[1].get_bit(to) || to == state.en_passant) { return true; }
                return false;
            },
            p => { 
                if to == from.wrapping_sub(8) && !occ.get_bit(to) { return true; }
                if from >= 48 && from <= 55 && to == from.wrapping_sub(16) && !occ.get_bit(from.wrapping_sub(8)) && !occ.get_bit(to) { return true; }
                if (to == from.wrapping_sub(7) || to == from.wrapping_sub(9)) && (state.occupancies[0].get_bit(to) || to == state.en_passant) { return true; }
                return false;
            },
            R | r => {
                let attacks = bitboard::get_rook_attacks(from, occ);
                if !attacks.get_bit(to) { return false; }
            },
            B | b => {
                let attacks = bitboard::get_bishop_attacks(from, occ);
                if !attacks.get_bit(to) { return false; }
            },
            Q | q => {
                let attacks = bitboard::get_queen_attacks(from, occ);
                if !attacks.get_bit(to) { return false; }
            },
            _ => return false
        }
        true
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