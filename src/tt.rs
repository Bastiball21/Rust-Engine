#![allow(non_upper_case_globals)]

use crate::state::{GameState, Move, Q, R, B, N};
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

#[cfg(feature = "packed-tt")]
compile_error!("packed-tt uses a 16-bit key and is unsafe. Do not enable unless key size is increased.");

/// Trait abstracting TT Entry operations.
pub trait TTEntryTrait {
    fn new() -> Self;
    fn key(&self) -> u64;
    fn save(&self, key: u64, score: i32, depth: u8, flag: u8, age: u8, mv: Option<Move>);
    fn probe(&self, key: u64) -> Option<(i32, u8, u8, u8, Option<Move>)>;
}

#[inline(always)]
fn unpack_depth_age(data: u64) -> (u8, u8) {
    let depth = ((data >> 32) & 0xFF) as u8;
    let age = ((data >> 48) & 0xFF) as u8;
    (depth, age)
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
                Some(4) => 1,
                Some(3) => 2,
                Some(2) => 3,
                Some(1) => 4,
                _ => 0,
            };
            let mut bits = ((m.source() as u16) << 6) | (m.target() as u16) | (promo_bits << 12);
            if m.is_capture() {
                bits |= 0x8000;
            }
            bits
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

        self.key.store(stored_key, Ordering::Relaxed);
        self.data.store(data, Ordering::Release);
    }

    fn probe(&self, key: u64) -> Option<(i32, u8, u8, u8, Option<Move>)> {
        let data = self.data.load(Ordering::Acquire);
        let stored_key = self.key.load(Ordering::Relaxed);

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
            let is_capture = (move_u16 & 0x8000) != 0;
            if from != to {
                Some(Move::new(from as u8, to as u8, promotion, is_capture))
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
// CLUSTER (2-way bucket)
// --------------------------------------------------------------------------------

#[cfg(not(feature = "packed-tt"))]
const CLUSTER_SIZE: usize = 4;

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

    fn get_shard(&self, hash: u64, _thread_id: Option<usize>) -> &TTShard {
        if self.num_shards == 1 {
            return &self.shards[0];
        }

        let shard_idx = if self.num_shards.is_power_of_two() {
            ((hash >> 48) as usize) & (self.num_shards - 1)
        } else {
            ((hash >> 48) as usize) % self.num_shards
        };

        &self.shards[shard_idx]
    }

    pub fn prefetch(&self, hash: u64, thread_id: Option<usize>) {
        let shard = self.get_shard(hash, thread_id);
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

            // 1. Search for existing entry to overwrite
            for i in 0..CLUSTER_SIZE {
                let entry = &cluster.entries[i];
                let data = entry.data.load(Ordering::Relaxed);

                if data != 0 {
                    let stored_key = entry.key.load(Ordering::Relaxed);
                    if (stored_key ^ data) == hash {
                        // Found match: Overwrite
                        // Replacement Strategy:
                        // 1. Prefer deeper searches (depth > old_depth)
                        // 2. If depth is equal, prefer Exact bounds over Alpha/Beta
                        // 3. Always update if the entry is from an old generation (stale)
                        // 4. Update if we have a better move (or same move with better score context?)

                        let old_depth = ((data >> 32) & 0xFF) as u8;
                        let old_flag = ((data >> 40) & 0xFF) as u8;
                        let old_age = ((data >> 48) & 0xFF) as u8;

                        let overwrite = if old_age != current_gen {
                            true // Always refresh stale entries (to prevent eviction of useful nodes)
                        } else if depth > old_depth {
                            true // Deeper is better
                        } else if depth == old_depth {
                            // Tie-break: Prefer Exact, or if same, prefer new (update score/age)
                            if flag == FLAG_EXACT {
                                true
                            } else if old_flag == FLAG_EXACT {
                                false
                            } else {
                                true
                            }
                        } else {
                            false // Shallower search, keep old info
                        };

                        if overwrite {
                            let old_move = if best_move.is_none() {
                                 entry.probe(hash).and_then(|(_,_,_,_,m)| m)
                            } else {
                                 None
                            };
                            let final_move = best_move.or(old_move);
                            entry.save(hash, score, depth, flag, current_gen, final_move);
                        }
                        return;
                    }
                }
            }

            // 2. Find replacement victim (Worst Slot)
            // Metric: Lowest Depth. Tie-break: Oldest Gen (largest age diff).
            // Quality Q = (depth << 8) + (255 - age_diff).
            // We want to replace the entry with MINIMUM Quality.

            let mut victim_idx = 0;
            let mut min_quality = u32::MAX;

            for i in 0..CLUSTER_SIZE {
                let entry = &cluster.entries[i];
                let data = entry.data.load(Ordering::Relaxed);

                if data == 0 {
                    // Empty slot is quality 0, immediate replace
                    victim_idx = i;
                    min_quality = 0;
                    break;
                }

                let (d, age) = unpack_depth_age(data);
                let age_diff = current_gen.wrapping_sub(age);

                // Quality: Higher depth = Better. Lower age diff (More recent) = Better.
                // We want to MINIMIZE quality to find victim.
                // Depth is dominant.
                let quality = ((d as u32) << 8) | (255 - age_diff as u32);

                if quality < min_quality {
                    min_quality = quality;
                    victim_idx = i;
                }
            }

            // 3. Replace Victim
            let entry = &cluster.entries[victim_idx];
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
                if let Some((score, depth, flag, _, mv)) = entry.probe(hash) {
                    return Some((score, depth, flag, mv));
                }
            }
        }
        None
    }

    pub fn get_move(&self, hash: u64, state: &GameState, thread_id: Option<usize>) -> Option<Move> {
        self.probe_data(hash, state, thread_id).and_then(|(_, _, _, m)| m)
    }

    pub fn is_pseudo_legal(&self, state: &crate::state::GameState, mv: Move) -> bool {
        crate::movegen::is_move_pseudo_legal(state, mv)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tt_replacement_policy() {
        crate::zobrist::init_zobrist();
        crate::bitboard::init_magic_tables();

        // Mock TT with 1 Cluster
        // CLUSTER_SIZE is 4 now
        let mut tt = TranspositionTable::new(1, 1);
        let current_gen = 10;
        tt.generation.store(current_gen, Ordering::Relaxed);

        // Ensure collision by using same index bits (lower bits)
        // TT size 1MB ~ 16384 clusters. Mask is 16383 (0x3FFF).
        // We shift high bits to differentiate keys but keep low bits identical.
        let base_index = 5;
        let hash_a = (1u64 << 48) | base_index;
        let hash_b = (2u64 << 48) | base_index;
        let hash_c = (3u64 << 48) | base_index;
        let hash_d = (4u64 << 48) | base_index;
        let hash_e = (5u64 << 48) | base_index;

        // Dummy GameState for probe
        let dummy_fen = "8/8/8/8/8/8/8/8 w - - 0 1";
        let dummy_state = GameState::parse_fen(dummy_fen);

        // 1. Fill Cluster with 4 entries
        // A: Depth 5
        // B: Depth 4
        // C: Depth 6
        // D: Depth 5
        tt.store(hash_a, 100, None, 5, FLAG_EXACT, None);
        tt.store(hash_b, 200, None, 4, FLAG_EXACT, None);
        tt.store(hash_c, 300, None, 6, FLAG_EXACT, None);
        tt.store(hash_d, 400, None, 5, FLAG_EXACT, None);

        // Verify all are there
        assert!(tt.probe_data(hash_a, &dummy_state, None).is_some());
        assert!(tt.probe_data(hash_b, &dummy_state, None).is_some());
        assert!(tt.probe_data(hash_c, &dummy_state, None).is_some());
        assert!(tt.probe_data(hash_d, &dummy_state, None).is_some());

        // 2. Store E: Depth 6. Should replace B (Depth 4 is lowest)
        tt.store(hash_e, 500, None, 6, FLAG_EXACT, None);

        assert!(tt.probe_data(hash_a, &dummy_state, None).is_some(), "A (Depth 5) kept");
        assert!(tt.probe_data(hash_b, &dummy_state, None).is_none(), "B (Depth 4) replaced");
        assert!(tt.probe_data(hash_c, &dummy_state, None).is_some(), "C (Depth 6) kept");
        assert!(tt.probe_data(hash_d, &dummy_state, None).is_some(), "D (Depth 5) kept");
        assert!(tt.probe_data(hash_e, &dummy_state, None).is_some(), "E (Depth 6) stored");

        // 3. Test Aging Tiebreak
        tt.clear();

        // Setup 4 slots full
        tt.generation.store(5, Ordering::Relaxed);
        tt.store(hash_a, 100, None, 5, FLAG_EXACT, None); // Old (Gen 5)

        tt.generation.store(10, Ordering::Relaxed);
        tt.store(hash_b, 200, None, 5, FLAG_EXACT, None); // New (Gen 10)
        tt.store(hash_c, 300, None, 5, FLAG_EXACT, None); // New (Gen 10)
        tt.store(hash_d, 400, None, 5, FLAG_EXACT, None); // New (Gen 10)

        // Store E (Depth 5). Should replace A (Oldest)
        tt.store(hash_e, 500, None, 5, FLAG_EXACT, None);

        assert!(tt.probe_data(hash_a, &dummy_state, None).is_none(), "A (Old) should be replaced");
        assert!(tt.probe_data(hash_b, &dummy_state, None).is_some(), "B (New) kept");
        assert!(tt.probe_data(hash_c, &dummy_state, None).is_some(), "C (New) kept");
        assert!(tt.probe_data(hash_d, &dummy_state, None).is_some(), "D (New) kept");
        assert!(tt.probe_data(hash_e, &dummy_state, None).is_some(), "E stored");
    }
}
