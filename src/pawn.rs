// src/pawn.rs
use crate::bitboard::{self, Bitboard, FILE_A, FILE_H};
use crate::state::{p, GameState, BLACK, P, WHITE};
use std::cell::RefCell;

// --- PAWN HASH TABLE ---
const PAWN_HASH_SIZE: usize = 16384; // 16K entries, Power of two

#[derive(Clone, Copy, Default, Debug)]
struct PawnHashEntry {
    key: u64,
    pawn_entry: PawnEntry,
}

struct PawnHashTable {
    table: Box<[PawnHashEntry; PAWN_HASH_SIZE]>,
}

impl PawnHashTable {
    fn new() -> Self {
        PawnHashTable {
            table: vec![PawnHashEntry::default(); PAWN_HASH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
        }
    }

    #[inline(always)]
    fn get(&self, key: u64) -> Option<PawnEntry> {
        let idx = (key as usize) & (PAWN_HASH_SIZE - 1);
        let entry = &self.table[idx];
        if entry.key == key {
            Some(entry.pawn_entry)
        } else {
            None
        }
    }

    #[inline(always)]
    fn put(&mut self, key: u64, entry: PawnEntry) {
        let idx = (key as usize) & (PAWN_HASH_SIZE - 1);
        self.table[idx] = PawnHashEntry {
            key,
            pawn_entry: entry,
        };
    }
}

// Thread-Local Pawn Hash
thread_local! {
    static PAWN_TABLE: RefCell<PawnHashTable> = RefCell::new(PawnHashTable::new());
}

#[derive(Clone, Copy, Default, Debug)]
pub struct PawnEntry {
    pub score_mg: i32,
    pub score_eg: i32,
    pub passed_pawns: [Bitboard; 2],
    pub pawn_attacks: [Bitboard; 2],
    // Removed large king_safety_scores array if unused to save space/copy
}

pub fn evaluate_pawns(state: &GameState) -> PawnEntry {
    let key = state.pawn_key;

    // Probe Cache
    // We use a block to scope the borrow
    let cached = PAWN_TABLE.with(|pt| {
        pt.borrow().get(key)
    });

    if let Some(entry) = cached {
        return entry;
    }

    // Compute
    let mut entry = PawnEntry::default();
    let w_pawns = state.bitboards[P];
    let b_pawns = state.bitboards[p];

    entry.pawn_attacks[WHITE] = bitboard::pawn_attacks(w_pawns, WHITE);
    entry.pawn_attacks[BLACK] = bitboard::pawn_attacks(b_pawns, BLACK);

    // --- WHITE PAWNS ---
    let mut bb = w_pawns;
    while bb.0 != 0 {
        let sq = bb.pop_lsb() as usize; // Use fast pop_lsb
        let rank = sq / 8;

        // Connected
        if entry.pawn_attacks[WHITE].get_bit(sq as u8) {
            entry.score_mg += 10;
            entry.score_eg += 15;
        }

        // Isolated
        let file_mask = bitboard::file_mask(sq);
        let adj_mask = ((file_mask.0 << 1) & !FILE_A) | ((file_mask.0 >> 1) & !FILE_H);
        if (w_pawns.0 & adj_mask) == 0 {
            entry.score_mg -= 15;
            entry.score_eg -= 20;
        }

        // Doubled
        if (w_pawns.0 & file_mask.0).count_ones() > 1 {
            entry.score_mg -= 10;
            entry.score_eg -= 15;
        }

        // Passed
        let passed_mask = bitboard::passed_pawn_mask(WHITE, sq);
        if (passed_mask.0 & b_pawns.0) == 0 {
            entry.passed_pawns[WHITE].set_bit(sq as u8);
            let bonus = [0, 10, 20, 40, 70, 120, 200, 0];
            entry.score_mg += bonus[rank] / 2;
            entry.score_eg += bonus[rank];
        }
    }

    // --- BLACK PAWNS ---
    let mut bb = b_pawns;
    while bb.0 != 0 {
        let sq = bb.pop_lsb() as usize; // Use fast pop_lsb
        let rank = sq / 8;
        let rel_rank = 7 - rank;

        if entry.pawn_attacks[BLACK].get_bit(sq as u8) {
            entry.score_mg -= 10;
            entry.score_eg -= 15;
        }

        let file_mask = bitboard::file_mask(sq);
        let adj_mask = ((file_mask.0 << 1) & !FILE_A) | ((file_mask.0 >> 1) & !FILE_H);
        if (b_pawns.0 & adj_mask) == 0 {
            entry.score_mg += 15;
            entry.score_eg += 20;
        }

        if (b_pawns.0 & file_mask.0).count_ones() > 1 {
            entry.score_mg += 10;
            entry.score_eg += 15;
        }

        let passed_mask = bitboard::passed_pawn_mask(BLACK, sq);
        if (passed_mask.0 & w_pawns.0) == 0 {
            entry.passed_pawns[BLACK].set_bit(sq as u8);
            let bonus = [0, 10, 20, 40, 70, 120, 200, 0];
            entry.score_mg -= bonus[rel_rank] / 2;
            entry.score_eg -= bonus[rel_rank];
        }
    }

    // Save to Cache
    PAWN_TABLE.with(|pt| {
        pt.borrow_mut().put(key, entry);
    });

    entry
}
