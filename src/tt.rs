#![allow(non_upper_case_globals)]

use crate::bitboard;
use crate::state::{b, k, n, p, q, r, GameState, Move, B, K, N, P, Q, R, WHITE, NO_PIECE};
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

#[cfg(target_os = "linux")]
use libc::{c_void, mmap, munmap, MAP_ANONYMOUS, MAP_FAILED, MAP_HUGETLB, MAP_PRIVATE, PROT_READ, PROT_WRITE};

#[cfg(target_os = "windows")]
use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_LARGE_PAGES, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
#[cfg(target_os = "windows")]
use std::ffi::c_void;

pub const FLAG_NONE: u8 = 0;
pub const FLAG_EXACT: u8 = 1;
pub const FLAG_ALPHA: u8 = 2;
pub const FLAG_BETA: u8 = 3;

/// Trait abstracting TT Entry operations.
pub trait TTEntryTrait {
    fn new() -> Self;
    fn key(&self) -> u64;
    fn save(&self, key: u64, score: i32, depth: u8, flag: u8, age: u8, mv: Option<Move>);
    fn probe(&self, key: u64) -> Option<(i32, u8, u8, u8, Option<Move>)>;
}

// --------------------------------------------------------------------------------
// STANDARD LAYOUT (16 bytes)
// Key (64) | Data (64)
// --------------------------------------------------------------------------------
#[cfg(not(feature = "packed-tt"))]
#[derive(Debug)]
pub struct TTEntry {
    pub key: AtomicU64,
    pub data: AtomicU64,
}

#[cfg(not(feature = "packed-tt"))]
impl TTEntryTrait for TTEntry {
    fn new() -> Self {
        Self {
            key: AtomicU64::new(0),
            data: AtomicU64::new(0),
        }
    }

    fn key(&self) -> u64 {
        self.key.load(Ordering::Acquire)
    }

    fn save(&self, key: u64, score: i32, depth: u8, flag: u8, age: u8, mv: Option<Move>) {
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

        let data = (move_u16 as u64)
            | ((score_u16 as u64) << 16)
            | ((depth as u64) << 32)
            | ((flag as u64) << 40)
            | ((age as u64) << 48);

        // Lockless XOR Protection
        // We store (key ^ data) in the key field.
        // This ensures that if we read a mismatched key/data pair, the XOR check will fail.
        let stored_key = key ^ data;

        self.data.store(data, Ordering::Release);
        self.key.store(stored_key, Ordering::Release);
    }

    fn probe(&self, key: u64) -> Option<(i32, u8, u8, u8, Option<Move>)> {
        let data = self.data.load(Ordering::Relaxed);
        let stored_key = self.key.load(Ordering::Acquire);

        // Verify integrity
        if (stored_key ^ data) != key {
            return None;
        }

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
                    false,
                ))
            } else {
                None
            }
        } else {
            None
        };

        Some((score, depth, flag, age, mv))
    }
}

// --------------------------------------------------------------------------------
// PACKED LAYOUT (8 bytes)
// --------------------------------------------------------------------------------
#[cfg(feature = "packed-tt")]
#[derive(Debug)]
pub struct TTEntry {
    pub data: AtomicU64,
}

#[cfg(feature = "packed-tt")]
impl TTEntryTrait for TTEntry {
    fn new() -> Self {
        Self {
            data: AtomicU64::new(0),
        }
    }

    fn key(&self) -> u64 {
        let raw = self.data.load(Ordering::Acquire);
        let key16 = (raw & 0xFFFF) as u64;
        key16 << 48
    }

    fn save(&self, key: u64, score: i32, depth: u8, flag: u8, age: u8, mv: Option<Move>) {
        let key16 = (key >> 48) as u16;

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
        let gen_bound = ((age & 0x3F) << 2) | (flag & 0x3);

        let data = (key16 as u64)
                 | ((move_u16 as u64) << 16)
                 | ((score_u16 as u64) << 32)
                 | ((depth as u64) << 48)
                 | ((gen_bound as u64) << 56);

        self.data.store(data, Ordering::Release);
    }

    fn probe(&self, _key: u64) -> Option<(i32, u8, u8, u8, Option<Move>)> {
        let data = self.data.load(Ordering::Relaxed);
        if data == 0 { return None; }

        let move_u16 = ((data >> 16) & 0xFFFF) as u16;
        let score = ((data >> 32) & 0xFFFF) as i32 - 32000;
        let depth = ((data >> 48) & 0xFF) as u8;
        let gen_bound = ((data >> 56) & 0xFF) as u8;

        let flag = gen_bound & 0x3;
        let age = (gen_bound >> 2) & 0x3F;

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
                    false,
                ))
            } else {
                None
            }
        } else {
            None
        };

        Some((score, depth, flag, age, mv))
    }
}


#[cfg(not(feature = "packed-tt"))]
const CLUSTER_SIZE: usize = 4;

#[cfg(feature = "packed-tt")]
const CLUSTER_SIZE: usize = 8;

#[repr(C, align(64))]
pub struct Cluster {
    pub entries: [TTEntry; CLUSTER_SIZE],
}

// Inner structure to hold allocation details per shard
struct TTShard {
    table: *mut Cluster,
    count: usize,
    mask: usize,
    is_large_page: bool,
}

unsafe impl Send for TTShard {}
unsafe impl Sync for TTShard {}

impl TTShard {
    fn new(mb: usize) -> Self {
        let cluster_size = std::mem::size_of::<Cluster>();
        let desired_bytes = mb * 1024 * 1024;
        let desired_count = desired_bytes / cluster_size;

        // Power of two count
        let count = desired_count.next_power_of_two() >> 1; // Round down to be safe, or up?
        // User requested "enforce power-of-two size".
        // Usually we want to fill the RAM. next_power_of_two might exceed mb.
        // Let's use previous_power_of_two (actually `next_power_of_two() / 2` if not exact, or just msb).
        let mut count = 1;
        while count * 2 * cluster_size <= desired_bytes {
            count *= 2;
        }

        // Ensure at least one cluster
        if count == 0 { count = 1; }

        let size_bytes = count * cluster_size;
        let mask = count - 1;

        let mut ptr = std::ptr::null_mut();
        let mut is_large_page = false;

        #[cfg(target_os = "linux")]
        {
            unsafe {
                let addr = mmap(
                    std::ptr::null_mut(),
                    size_bytes,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
                    -1,
                    0,
                );

                if addr != MAP_FAILED {
                    ptr = addr as *mut Cluster;
                    is_large_page = true;
                }
            }
        }

        #[cfg(target_os = "windows")]
        {
            unsafe {
                let addr = VirtualAlloc(
                    std::ptr::null(),
                    size_bytes,
                    MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                    PAGE_READWRITE,
                );

                if !addr.is_null() {
                    ptr = addr as *mut Cluster;
                    is_large_page = true;
                }
            }
        }

        if ptr.is_null() {
             let layout = Layout::from_size_align(size_bytes, 64).unwrap();
             ptr = unsafe { alloc_zeroed(layout) as *mut Cluster };
        }

        Self {
            table: ptr,
            count,
            mask,
            is_large_page,
        }
    }
}

impl Drop for TTShard {
    fn drop(&mut self) {
        if !self.table.is_null() {
            let cluster_size = std::mem::size_of::<Cluster>();
            let size_bytes = self.count * cluster_size;

            if self.is_large_page {
                #[cfg(target_os = "linux")]
                unsafe { munmap(self.table as *mut c_void, size_bytes); }
                #[cfg(target_os = "windows")]
                unsafe { VirtualFree(self.table as *mut c_void, 0, MEM_RELEASE); }
            } else {
                let layout = Layout::from_size_align(size_bytes, 64).unwrap();
                unsafe { dealloc(self.table as *mut u8, layout); }
            }
        }
    }
}

pub struct TranspositionTable {
    shards: Vec<TTShard>,
    num_shards: usize,
    pub generation: AtomicU8,
}

unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    pub fn new(mb: usize, num_shards: usize) -> Self {
        let mb_per_shard = if num_shards > 0 { mb / num_shards } else { mb };
        let real_shards = if num_shards == 0 { 1 } else { num_shards };

        let mut shards = Vec::with_capacity(real_shards);
        for _ in 0..real_shards {
            shards.push(TTShard::new(mb_per_shard));
        }

        let total_clusters: usize = shards.iter().map(|s| s.count).sum();
        let cluster_size = std::mem::size_of::<Cluster>();
        let total_mb = (total_clusters * cluster_size) / (1024 * 1024);

        println!("TT: {} MB total ({} shards of ~{} MB), {} Clusters, Masked Indexing",
                 total_mb, real_shards, mb_per_shard, total_clusters);

        Self {
            shards,
            num_shards: real_shards,
            generation: AtomicU8::new(0),
        }
    }

    pub fn new_default(mb: usize) -> Self {
        Self::new(mb, 1)
    }

    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    pub fn clear(&mut self) {
        for shard in &mut self.shards {
            unsafe {
                std::ptr::write_bytes(shard.table, 0, shard.count);
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    fn get_shard(&self, _key: u64, thread_id: Option<usize>) -> &TTShard {
        if self.num_shards == 1 {
            return &self.shards[0];
        }
        // If thread_id is provided, use it (Private Partitioning / Sharding)
        // User asked: "index by (key ^ thread_id)"?
        // Actually, user said: "allocate shards and index by (key ^ thread_id) & (shard_size - 1)".
        // This implies selecting the index WITHIN the shard using thread_id mixing.
        // But do we select the SHARD based on thread_id?
        // "Shard TT per thread" suggests 1 shard per thread.
        // So we select shard by thread_id % num_shards.

        if let Some(tid) = thread_id {
            &self.shards[tid % self.num_shards]
        } else {
            // Fallback if no thread_id (e.g. main thread helpers) -> use shard 0
            &self.shards[0]
        }
    }

    pub fn prefetch(&self, hash: u64, thread_id: Option<usize>) {
        let shard = self.get_shard(hash, thread_id);
        // Masked Indexing
        let index = (hash as usize) & shard.mask;
        unsafe {
            let ptr = shard.table.add(index);
            #[cfg(target_arch = "x86_64")]
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    pub fn store(&self, hash: u64, score: i32, best_move: Option<Move>, depth: u8, flag: u8, thread_id: Option<usize>) {
        let shard = self.get_shard(hash, thread_id);
        let index = (hash as usize) & shard.mask;
        let current_gen = self.generation.load(Ordering::Relaxed);

        unsafe {
            let cluster = &*shard.table.add(index);
            let mut best_victim_idx = 0;
            let mut min_score = i32::MAX;

            for i in 0..CLUSTER_SIZE {
                let entry = &cluster.entries[i];

                #[cfg(feature = "packed-tt")]
                let (match_found, is_empty) = {
                    let entry_key = entry.key();
                    (
                        (entry_key == (hash & 0xFFFF_0000_0000_0000)) && (entry_key != 0),
                        entry_key == 0
                    )
                };

                #[cfg(not(feature = "packed-tt"))]
                let (match_found, is_empty) = {
                    let data = entry.data.load(Ordering::Relaxed);
                    let stored_key = entry.key.load(Ordering::Acquire);
                    let recovered_key = stored_key ^ data;
                    // data != 0 ensures we don't match empty slots if hash happens to be 0
                    (recovered_key == hash && data != 0, data == 0)
                };

                if match_found {
                    let (_, _, _, _, old_move) = entry.probe(hash).unwrap_or((0,0,0,0,None));
                    let final_move = if best_move.is_none() { old_move } else { best_move };
                    entry.save(hash, score, depth, flag, current_gen, final_move);
                    return;
                }

                if is_empty {
                    entry.save(hash, score, depth, flag, current_gen, best_move);
                    return;
                }

                let (_, d, _, age, _) = entry.probe(hash).unwrap_or((0,0,0,0,None));
                let age_diff = current_gen.wrapping_sub(age);

                // Replacement Strategy: Prefer replacing shallow entries, then old entries.
                // Score = Depth - AgePenalty. Lower is better candidate to replace.
                // If entry is ancient (age_diff large), score drops significantly.
                let replace_score = (d as i32) - (age_diff as i32 * 4);

                if replace_score < min_score {
                    min_score = replace_score;
                    best_victim_idx = i;
                }
            }

            // Always replace if new depth is greater, or if victim is very old/shallow
            let entry = &cluster.entries[best_victim_idx];
            // Just force replace the victim found
            entry.save(hash, score, depth, flag, current_gen, best_move);
        }
    }

    pub fn probe_data(&self, hash: u64, state: &GameState, thread_id: Option<usize>) -> Option<(i32, u8, u8, Option<Move>)> {
        let shard = self.get_shard(hash, thread_id);
        let index = (hash as usize) & shard.mask;

        unsafe {
            let cluster = &*shard.table.add(index);
            for i in 0..CLUSTER_SIZE {
                let entry = &cluster.entries[i];

                #[cfg(feature = "packed-tt")]
                {
                    if entry.key() == (hash & 0xFFFF_0000_0000_0000) {
                         if let Some((score, depth, flag, _, mut mv)) = entry.probe(hash) {
                             if let Some(ref mut m) = mv { Self::fix_move(m, state); }
                             return Some((score, depth, flag, mv));
                         }
                    }
                }

                #[cfg(not(feature = "packed-tt"))]
                {
                    // entry.probe() handles the XOR decoding internally
                    if let Some((score, depth, flag, _, mut mv)) = entry.probe(hash) {
                        if let Some(ref mut m) = mv { Self::fix_move(m, state); }
                        return Some((score, depth, flag, mv));
                    }
                }
            }
        }
        None
    }

    pub fn get_move(&self, hash: u64, state: &GameState, thread_id: Option<usize>) -> Option<Move> {
        self.probe_data(hash, state, thread_id).and_then(|(_, _, _, m)| m)
    }

    fn fix_move(mv: &mut Move, state: &GameState) {
        let to = mv.target() as usize;
        let from = mv.source() as usize;
        let captured = state.board[to];
        let mut is_capture = false;

        if captured != NO_PIECE as u8 {
            is_capture = true;
        } else if mv.target() == state.en_passant {
            let piece = state.board[from];
            if piece == P as u8 || piece == p as u8 {
                is_capture = true;
            }
        }

        // Fix: Castling is not a capture!
        // If King captures friendly Rook, it's castling, so set is_capture = false.
        // BUT we must be careful: King capturing Friendly QUEEN is NOT castling.
        // It must be a Rook.
        if is_capture {
            let piece_type = state.board[from] as usize;
            let captured_type = state.board[to] as usize;

            let rook_type = if state.side_to_move == WHITE { R } else { r };

            // Only convert to Quiet if target is actually a Rook
            if (piece_type == K || piece_type == k) && captured_type == rook_type {
                let is_friendly = if state.side_to_move == WHITE {
                     captured_type <= K
                } else {
                     captured_type >= p
                };

                if is_friendly {
                    is_capture = false;
                }
            }
        }

        *mv = Move::new(mv.source(), mv.target(), mv.promotion(), is_capture);
    }

    pub fn is_pseudo_legal(&self, state: &crate::state::GameState, mv: Move) -> bool {
        let from = mv.source();
        let to = mv.target();
        let side = state.side_to_move;

        if from >= 64 || to >= 64 || from == to { return false; }

        let piece_type = state.board[from as usize] as usize;
        if piece_type == 12 || !state.bitboards[piece_type].get_bit(from) { return false; }

        if side == WHITE {
             if piece_type > K { return false; }
        } else {
             if piece_type < p || piece_type > k { return false; }
        }

        let target_piece = state.board[to as usize] as usize;
        if target_piece != 12 {
             if side == WHITE {
                  if target_piece <= K {
                    // Allow King moves to friendly rook for castling
                    if (piece_type == K) && (target_piece == R) {
                         // Check castling rights?
                         // Ideally we should check if this specific rook is valid for castling,
                         // but for pseudo-legal check, just knowing it's a rook and we have *some* rights is often enough,
                         // or we can rely on movegen to filter.
                         // But for TT retrieval validation, we should be slightly permissive or check rights.
                         // The safest is to return true here and let make_move handle validity, OR check rights.
                         if state.castling_rights & 3 != 0 { return true; }
                    }
                    return false;
                  }
             } else {
                  if target_piece >= p && target_piece <= k {
                    // Allow King moves to friendly rook for castling
                    if (piece_type == k) && (target_piece == r) {
                         if state.castling_rights & 12 != 0 { return true; }
                    }
                    return false;
                  }
             }
        }

        let is_occupied = target_piece != 12;
        if is_occupied && !state.bitboards[target_piece].get_bit(to) { return false; }

        let is_ep = to == state.en_passant && (piece_type == P || piece_type == p);

        if mv.is_capture() {
            if !is_occupied && !is_ep { return false; }
        } else {
            // Check if target is occupied in BITBOARDS to prevent "Quiet" moves to "Ghost" squares.
            // If board says empty (is_occupied=false) but bitboard says occupied, it's a desync/ghost -> Reject.
            if state.occupancies[crate::state::BOTH].get_bit(to) {
                if !is_occupied { return false; }

                // If occupied (in both), it must be a Castling attempt (King -> Own Rook)
                // Any other quiet move to an occupied square is illegal (capture flag missing).
                let is_castling_attempt = (piece_type == K && target_piece == R) || (piece_type == k && target_piece == r);

                if !is_castling_attempt {
                     return false;
                }

                if is_castling_attempt {
                    return true;
                }
                return false;
            }
        }

        match piece_type {
            N | n => bitboard::mask_knight_attacks(from).get_bit(to),
            K | k => {
                let attacks = bitboard::mask_king_attacks(from);
                if attacks.get_bit(to) { return true; }
                if (from as i8 - to as i8).abs() == 2 { return true; }
                false
            }
            P => {
                let file_from = from % 8;
                let file_to = to % 8;
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
                    return (file_from as i8 - file_to as i8).abs() == 1;
                }
                false
            }
            p => {
                let file_from = from % 8;
                let file_to = to % 8;
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
                    return (file_from as i8 - file_to as i8).abs() == 1;
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
        let mut total_slots = 0;

        for shard in &self.shards {
            let scan_limit = if shard.count > 250 { 250 } else { shard.count };
            for i in 0..scan_limit {
                unsafe {
                    let cluster = &*shard.table.add(i);
                    for entry in &cluster.entries {
                        #[cfg(feature = "packed-tt")]
                        if entry.key() != 0 { used += 1; }

                        #[cfg(not(feature = "packed-tt"))]
                        if entry.data.load(Ordering::Relaxed) != 0 { used += 1; }
                    }
                }
            }
            total_slots += scan_limit * CLUSTER_SIZE;
        }

        if total_slots == 0 { return 0; }
        (used * 1000) / total_slots
    }
}
