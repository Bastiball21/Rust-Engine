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
/// This allows the TranspositionTable to work with different layouts seamlessly.
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
        self.key.load(Ordering::Acquire) // Acquire to sync with data
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

        // Standard: Store Data (Relaxed) then Key (Release)
        self.data.store(data, Ordering::Relaxed);
        self.key.store(key, Ordering::Release);
    }

    fn probe(&self, _key: u64) -> Option<(i32, u8, u8, u8, Option<Move>)> {
        // Note: Key check is done by caller via self.key()
        let data = self.data.load(Ordering::Relaxed);
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
// Key(16) | Move(16) | Score(16) | Depth(8) | Gen(6) + Bound(2)
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
        // In packed mode, we don't return the full key.
        // The caller handles full key logic, but here we can only verify the upper 16 bits.
        // We return a "partial" check or 0.
        // Actually, to implement `probe_data` cleanly, `TranspositionTable` needs to handle the logic.
        // But `TTEntryTrait` assumes we can get the key.
        // We will return the stored 16-bit key shifted up.
        // The caller (TranspositionTable) must compare only the upper 16 bits.
        let raw = self.data.load(Ordering::Acquire);
        let key16 = (raw & 0xFFFF) as u64;
        key16 << 48
    }

    fn save(&self, key: u64, score: i32, depth: u8, flag: u8, age: u8, mv: Option<Move>) {
        let key16 = (key >> 48) as u16; // Upper 16 bits

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

        // Gen (6 bits) | Bound (2 bits) -> 8 bits
        // Bound: 0=None, 1=Exact, 2=Alpha, 3=Beta. Fits in 2 bits.
        let gen_bound = ((age & 0x3F) << 2) | (flag & 0x3);

        let data = (key16 as u64)
                 | ((move_u16 as u64) << 16)
                 | ((score_u16 as u64) << 32)
                 | ((depth as u64) << 48)
                 | ((gen_bound as u64) << 56);

        self.data.store(data, Ordering::Release);
    }

    fn probe(&self, _key: u64) -> Option<(i32, u8, u8, u8, Option<Move>)> {
        // Assume key check passed (upper 16 bits matched)
        let data = self.data.load(Ordering::Relaxed);
        let key16 = (data & 0xFFFF) as u16; // Not needed

        // If empty
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
const CLUSTER_SIZE: usize = 8; // 64 bytes / 8 bytes = 8 entries

#[repr(C, align(64))]
pub struct Cluster {
    pub entries: [TTEntry; CLUSTER_SIZE],
}

pub struct TranspositionTable {
    pub table: *mut Cluster,
    pub count: usize,
    pub generation: AtomicU8, // Tracks the "age" of the search
    pub is_large_page: bool,
}

unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl Drop for TranspositionTable {
    fn drop(&mut self) {
        if !self.table.is_null() {
            let cluster_size = std::mem::size_of::<Cluster>();
            let size_bytes = self.count * cluster_size;

            if self.is_large_page {
                #[cfg(target_os = "linux")]
                unsafe {
                    munmap(self.table as *mut c_void, size_bytes);
                }

                #[cfg(target_os = "windows")]
                unsafe {
                    // dwSize must be 0 if MEM_RELEASE is used
                    VirtualFree(self.table as *mut c_void, 0, MEM_RELEASE);
                }
            } else {
                let layout = Layout::from_size_align(size_bytes, 64).unwrap();
                unsafe {
                    dealloc(self.table as *mut u8, layout);
                }
            }
        }
    }
}

impl TranspositionTable {
    pub fn new(mb: usize) -> Self {
        let cluster_size = std::mem::size_of::<Cluster>();
        // Assert compile-time alignment
        assert_eq!(cluster_size, 64, "Cluster must be 64 bytes");

        let desired_bytes = mb * 1024 * 1024;
        let count = desired_bytes / cluster_size;
        let size_bytes = count * cluster_size;

        let mut ptr = std::ptr::null_mut();
        let mut is_large_page = false;

        // Try Linux Large Pages
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
                    println!("info string Transposition Table allocated using large pages");
                }
            }
        }

        // Try Windows Large Pages
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
                    println!("info string Transposition Table allocated using large pages");
                }
            }
        }

        // Fallback
        if ptr.is_null() {
             #[cfg(any(target_os = "linux", target_os = "windows"))]
             {
                 println!("info string Failed to allocate large pages, falling back to standard pages");
             }

             let layout = Layout::from_size_align(size_bytes, 64).unwrap();
             ptr = unsafe { alloc_zeroed(layout) as *mut Cluster };
        }

        println!("TT: {} MB / {} Clusters / {}-Way Associative", mb, count, CLUSTER_SIZE);
        Self {
            table: ptr,
            count,
            generation: AtomicU8::new(0),
            is_large_page,
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

    pub fn store(&self, hash: u64, score: i32, best_move: Option<Move>, depth: u8, flag: u8) {
        let index = (hash as usize) % self.count;
        let current_gen = self.generation.load(Ordering::Relaxed);

        unsafe {
            let cluster = &*self.table.add(index);
            let mut best_victim_idx = 0;
            let mut min_score = i32::MAX;

            for i in 0..CLUSTER_SIZE {
                let entry = &cluster.entries[i];
                let entry_key = entry.key();

                // 1. Found exact match? Update it.
                // Packed: Check upper 16 bits
                #[cfg(feature = "packed-tt")]
                let match_found = (entry_key == (hash & 0xFFFF_0000_0000_0000)) && (entry_key != 0);
                #[cfg(not(feature = "packed-tt"))]
                let match_found = entry_key == hash;

                if match_found {
                    // Load old data to preserve move if needed
                    let (_, _, _, _, old_move) = entry.probe(hash).unwrap_or((0,0,0,0,None));

                    // Keep old move if new one is missing
                    let final_move = if best_move.is_none() {
                        old_move
                    } else {
                        best_move
                    };

                    entry.save(hash, score, depth, flag, current_gen, final_move);
                    return;
                }

                // 2. Found empty slot?
                // Packed: key() returns 0 for empty? Yes (data is 0)
                if entry_key == 0 {
                    entry.save(hash, score, depth, flag, current_gen, best_move);
                    return;
                }

                // 3. Replacement candidates
                let (_, d, _, age, _) = entry.probe(hash).unwrap_or((0,0,0,0,None));

                let age_diff = current_gen.wrapping_sub(age);
                let replace_score = (d as i32) - (age_diff as i32 * 4); // Boost age penalty

                if replace_score < min_score {
                    min_score = replace_score;
                    best_victim_idx = i;
                }
            }

            // 4. Replace worst
            let entry = &cluster.entries[best_victim_idx];
            let (_, old_d, _, old_age, _) = entry.probe(hash).unwrap_or((0,0,0,0,None));
            let old_age_diff = current_gen.wrapping_sub(old_age);

            if depth >= old_d || old_age_diff > 0 || (old_d < depth + 5) {
                entry.save(hash, score, depth, flag, current_gen, best_move);
            }
        }
    }

    pub fn probe_data(&self, hash: u64, state: &GameState) -> Option<(i32, u8, u8, Option<Move>)> {
        let index = (hash as usize) % self.count;
        unsafe {
            let cluster = &*self.table.add(index);
            for i in 0..CLUSTER_SIZE {
                let entry = &cluster.entries[i];

                #[cfg(feature = "packed-tt")]
                {
                    // Packed: Check upper 16 bits.
                    // This is "probabilistic" but with 16 bits + bucket index (approx 20 bits), it's 36 bits.
                    // Actually, bucket index is derived from hash.
                    // So we are checking bits [48..64].
                    // The lower bits determined the bucket.
                    // So we are verifying the hash is consistent.
                    if entry.key() == (hash & 0xFFFF_0000_0000_0000) {
                         if let Some((score, depth, flag, _, mut mv)) = entry.probe(hash) {
                             if let Some(ref mut m) = mv {
                                Self::fix_move(m, state);
                             }
                             return Some((score, depth, flag, mv));
                         }
                    }
                }

                #[cfg(not(feature = "packed-tt"))]
                {
                    if entry.key() == hash {
                        // Safe probe
                        if let Some((score, depth, flag, _, mut mv)) = entry.probe(hash) {
                            if let Some(ref mut m) = mv {
                                Self::fix_move(m, state);
                            }
                            return Some((score, depth, flag, mv));
                        }
                    }
                }
            }
        }
        None
    }

    pub fn get_move(&self, hash: u64, state: &GameState) -> Option<Move> {
        self.probe_data(hash, state).and_then(|(_, _, _, m)| m)
    }

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
                    if entry.key() != 0 {
                        used += 1;
                    }
                }
            }
        }
        (used * 1000) / (scan_limit * CLUSTER_SIZE)
    }
}
