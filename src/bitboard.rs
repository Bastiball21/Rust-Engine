use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// --- GLOBAL CONSTANTS ---
pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_H: u64 = 0x8080808080808080;
pub const RANK_1: u64 = 0x00000000000000FF;
pub const RANK_8: u64 = 0xFF00000000000000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Bitboard(pub u64);

// --- TABLES ---
// PEXT Tables: Flat arrays for performance
// Rook table size: 102,400 entries (sum of 2^bits for all squares) - roughly 800KB
// Bishop table size: 5,248 entries - roughly 40KB
static ROOK_TABLE: OnceLock<Vec<Bitboard>> = OnceLock::new();
static BISHOP_TABLE: OnceLock<Vec<Bitboard>> = OnceLock::new();
static ROOK_OFFSETS: OnceLock<[usize; 64]> = OnceLock::new();
static BISHOP_OFFSETS: OnceLock<[usize; 64]> = OnceLock::new();

// Masks
static ROOK_MASKS: OnceLock<[Bitboard; 64]> = OnceLock::new();
static BISHOP_MASKS: OnceLock<[Bitboard; 64]> = OnceLock::new();

static FILE_MASKS: OnceLock<[Bitboard; 64]> = OnceLock::new();
static RANK_MASKS: OnceLock<[Bitboard; 64]> = OnceLock::new();
static PASSED_PAWN_MASKS: OnceLock<[[Bitboard; 64]; 2]> = OnceLock::new();
static KING_ZONE_MASKS: OnceLock<[Bitboard; 64]> = OnceLock::new();

// --- SAFE ACCESSORS ---
#[inline(always)]
pub fn file_mask(sq: usize) -> Bitboard {
    FILE_MASKS.get().expect("Tables not init")[sq]
}

#[inline(always)]
pub fn rank_mask(sq: usize) -> Bitboard {
    RANK_MASKS.get().expect("Tables not init")[sq]
}

#[inline(always)]
pub fn passed_pawn_mask(side: usize, sq: usize) -> Bitboard {
    PASSED_PAWN_MASKS.get().expect("Tables not init")[side][sq]
}

#[inline(always)]
pub fn king_zone_mask(sq: usize) -> Bitboard {
    KING_ZONE_MASKS.get().expect("Tables not init")[sq]
}

impl Bitboard {
    pub fn new() -> Self {
        Bitboard(0)
    }
    #[inline(always)]
    pub fn set_bit(&mut self, square: u8) {
        self.0 |= 1u64 << square;
    }
    #[inline(always)]
    pub fn get_bit(&self, square: u8) -> bool {
        (self.0 & (1u64 << square)) != 0
    }
    #[inline(always)]
    pub fn pop_bit(&mut self, square: u8) {
        self.0 &= !(1u64 << square);
    }
    #[inline(always)]
    pub fn count_bits(&self) -> u32 {
        self.0.count_ones()
    }
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.0.count_ones() as usize
    }
    #[inline(always)]
    pub fn get_lsb_index(&self) -> u32 {
        self.0.trailing_zeros()
    }

    pub fn print(&self) {
        println!("  a b c d e f g h");
        println!(" -----------------");
        for rank in (0..8).rev() {
            print!("{}|", rank + 1);
            for file in 0..8 {
                let square = rank * 8 + file;
                let bit = if self.get_bit(square) { "X" } else { "." };
                print!("{} ", bit);
            }
            println!("|{}", rank + 1);
        }
        println!(" -----------------");
        println!("  Bitboard: {}", self.0);
    }
}

impl std::ops::BitOr for Bitboard {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Bitboard(self.0 | rhs.0)
    }
}
impl std::ops::BitAnd for Bitboard {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Bitboard(self.0 & rhs.0)
    }
}
impl std::ops::BitXor for Bitboard {
    type Output = Self;
    fn bitxor(self, rhs: Self) -> Self {
        Bitboard(self.0 ^ rhs.0)
    }
}
impl std::ops::Not for Bitboard {
    type Output = Self;
    fn not(self) -> Self {
        Bitboard(!self.0)
    }
}

// --- INITIALIZATION ---
pub fn init_magic_tables() {
    if ROOK_TABLE.get().is_some() {
        return;
    }

    init_eval_masks();

    println!("Initializing PEXT Bitboards...");

    // 1. Initialize Masks
    let mut r_masks = [Bitboard(0); 64];
    let mut b_masks = [Bitboard(0); 64];
    for sq in 0..64 {
        r_masks[sq] = mask_rook_attacks(sq as u8);
        b_masks[sq] = mask_bishop_attacks(sq as u8);
    }
    // Check if set is needed to avoid race condition panic in tests
    if ROOK_MASKS.get().is_none() {
        let _ = ROOK_MASKS.set(r_masks);
    }
    if BISHOP_MASKS.get().is_none() {
        let _ = BISHOP_MASKS.set(b_masks);
    }

    // 2. Initialize PEXT Tables
    let mut r_table = Vec::new();
    let mut b_table = Vec::new();
    let mut r_offsets = [0; 64];
    let mut b_offsets = [0; 64];

    let mut r_current_offset = 0;
    let mut b_current_offset = 0;

    for sq in 0..64 {
        // ROOKS
        r_offsets[sq] = r_current_offset;
        let r_mask = r_masks[sq];
        let r_bits = r_mask.count_bits();
        let r_size = 1 << r_bits;

        // Resize table
        r_table.resize(r_current_offset + r_size, Bitboard(0));

        // Iterate all submasks
        let mut map = Bitboard(0);
        loop {
            let idx = pext_safe(map.0, r_mask.0) as usize;
            let attacks = generate_rook_attacks_slow(sq as u8, map);
            r_table[r_current_offset + idx] = attacks;

            map.0 = map.0.wrapping_sub(r_mask.0) & r_mask.0;
            if map.0 == 0 {
                break;
            }
        }
        r_current_offset += r_size;

        // BISHOPS
        b_offsets[sq] = b_current_offset;
        let b_mask = b_masks[sq];
        let b_bits = b_mask.count_bits();
        let b_size = 1 << b_bits;

        b_table.resize(b_current_offset + b_size, Bitboard(0));

        let mut map = Bitboard(0);
        loop {
            let idx = pext_safe(map.0, b_mask.0) as usize;
            let attacks = generate_bishop_attacks_slow(sq as u8, map);
            b_table[b_current_offset + idx] = attacks;

            map.0 = map.0.wrapping_sub(b_mask.0) & b_mask.0;
            if map.0 == 0 {
                break;
            }
        }
        b_current_offset += b_size;
    }

    if ROOK_TABLE.get().is_none() {
        let _ = ROOK_TABLE.set(r_table);
    }
    if BISHOP_TABLE.get().is_none() {
        let _ = BISHOP_TABLE.set(b_table);
    }
    if ROOK_OFFSETS.get().is_none() {
        let _ = ROOK_OFFSETS.set(r_offsets);
    }
    if BISHOP_OFFSETS.get().is_none() {
        let _ = BISHOP_OFFSETS.set(b_offsets);
    }
}

fn init_eval_masks() {
    let mut f_masks = [Bitboard(0); 64];
    let mut r_masks = [Bitboard(0); 64];
    let mut passed = [[Bitboard(0); 64]; 2];
    let mut k_zones = [Bitboard(0); 64];

    for r in 0..8 {
        for f in 0..8 {
            let sq = r * 8 + f;
            for r2 in 0..8 {
                f_masks[sq].set_bit((r2 * 8 + f) as u8);
            }
            for f2 in 0..8 {
                r_masks[sq].set_bit((r * 8 + f2) as u8);
            }

            let mut w_passed = Bitboard(0);
            for r2 in (r + 1)..8 {
                w_passed.set_bit((r2 * 8 + f) as u8);
                if f > 0 {
                    w_passed.set_bit((r2 * 8 + (f - 1)) as u8);
                }
                if f < 7 {
                    w_passed.set_bit((r2 * 8 + (f + 1)) as u8);
                }
            }
            passed[0][sq] = w_passed;

            let mut b_passed = Bitboard(0);
            for r2 in 0..r {
                b_passed.set_bit((r2 * 8 + f) as u8);
                if f > 0 {
                    b_passed.set_bit((r2 * 8 + (f - 1)) as u8);
                }
                if f < 7 {
                    b_passed.set_bit((r2 * 8 + (f + 1)) as u8);
                }
            }
            passed[1][sq] = b_passed;

            let mut zone = Bitboard(0);
            let king_bb = 1u64 << sq;
            let attacks = mask_king_attacks(sq as u8);
            zone.0 |= attacks.0 | king_bb;
            if r < 6 {
                zone.0 |= (zone.0 << 8) | (zone.0 << 16);
            }
            k_zones[sq] = zone;
        }
    }

    let _ = FILE_MASKS.set(f_masks);
    let _ = RANK_MASKS.set(r_masks);
    let _ = PASSED_PAWN_MASKS.set(passed);
    let _ = KING_ZONE_MASKS.set(k_zones);
}

// --- ATTACK GETTERS ---
#[inline(always)]
pub fn get_rook_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    let mask = ROOK_MASKS.get().expect("Rook masks not init")[square as usize];
    let offset = ROOK_OFFSETS.get().expect("Rook offsets not init")[square as usize];
    let idx = pext_safe(occupancy.0, mask.0) as usize;
    ROOK_TABLE.get().expect("Rook table not init")[offset + idx]
}

#[inline(always)]
pub fn get_bishop_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    let mask = BISHOP_MASKS.get().expect("Bishop masks not init")[square as usize];
    let offset = BISHOP_OFFSETS.get().expect("Bishop offsets not init")[square as usize];
    let idx = pext_safe(occupancy.0, mask.0) as usize;
    BISHOP_TABLE.get().expect("Bishop table not init")[offset + idx]
}

#[inline(always)]
pub fn get_queen_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    get_rook_attacks(square, occupancy) | get_bishop_attacks(square, occupancy)
}

// --- PEXT HELPER ---
#[inline(always)]
fn pext_safe(val: u64, mask: u64) -> u64 {
    #[cfg(feature = "bmi2")]
    unsafe {
        #[cfg(target_arch = "x86_64")]
        {
            _pext_u64(val, mask)
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            // Fallback if bmi2 feature is on but not on x86_64 (should strictly not happen if correctly configured)
            // But we provide fallback to be safe for cross-compilation quirks
            software_pext(val, mask)
        }
    }

    #[cfg(not(feature = "bmi2"))]
    {
        software_pext(val, mask)
    }
}

fn software_pext(val: u64, mask: u64) -> u64 {
    let mut res = 0;
    let mut bb = val;
    let mut m = mask;
    let mut bit = 1;
    while m != 0 {
        if (m & 1) != 0 {
            if (bb & 1) != 0 {
                res |= bit;
            }
            bit <<= 1;
        }
        bb >>= 1;
        m >>= 1;
    }
    res
}

// --- GENERATORS (SLOW) ---
pub fn mask_rook_attacks(square: u8) -> Bitboard {
    let mut attacks = Bitboard(0);
    let r = square / 8;
    let f = square % 8;
    for r_itr in 1..7 {
        if r_itr != r {
            attacks.set_bit(r_itr * 8 + f);
        }
    }
    for f_itr in 1..7 {
        if f_itr != f {
            attacks.set_bit(r * 8 + f_itr);
        }
    }
    attacks
}

pub fn mask_bishop_attacks(square: u8) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8;
    let file = (square % 8) as i8;
    let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for (r_step, f_step) in directions {
        let mut r = rank + r_step;
        let mut f = file + f_step;
        while r > 0 && r < 7 && f > 0 && f < 7 {
            attacks.set_bit((r * 8 + f) as u8);
            r += r_step;
            f += f_step;
        }
    }
    attacks
}

pub fn mask_knight_attacks(square: u8) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8;
    let file = (square % 8) as i8;
    let jumps = [
        (2, 1),
        (1, 2),
        (-1, 2),
        (-2, 1),
        (-2, -1),
        (-1, -2),
        (1, -2),
        (2, -1),
    ];
    for (r_off, f_off) in jumps {
        let tr = rank + r_off;
        let tf = file + f_off;
        if tr >= 0 && tr <= 7 && tf >= 0 && tf <= 7 {
            attacks.set_bit((tr * 8 + tf) as u8);
        }
    }
    attacks
}

pub fn mask_king_attacks(square: u8) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8;
    let file = (square % 8) as i8;
    let steps = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];
    for (r_off, f_off) in steps {
        let tr = rank + r_off;
        let tf = file + f_off;
        if tr >= 0 && tr <= 7 && tf >= 0 && tf <= 7 {
            attacks.set_bit((tr * 8 + tf) as u8);
        }
    }
    attacks
}

pub fn generate_rook_attacks_slow(square: u8, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8;
    let file = (square % 8) as i8;
    let directions = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for (r_step, f_step) in directions {
        let mut r = rank + r_step;
        let mut f = file + f_step;
        while r >= 0 && r <= 7 && f >= 0 && f <= 7 {
            let target = (r * 8 + f) as u8;
            attacks.set_bit(target);
            if blockers.get_bit(target) {
                break;
            }
            r += r_step;
            f += f_step;
        }
    }
    attacks
}

pub fn generate_bishop_attacks_slow(square: u8, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8;
    let file = (square % 8) as i8;
    let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for (r_step, f_step) in directions {
        let mut r = rank + r_step;
        let mut f = file + f_step;
        while r >= 0 && r <= 7 && f >= 0 && f <= 7 {
            let target = (r * 8 + f) as u8;
            attacks.set_bit(target);
            if blockers.get_bit(target) {
                break;
            }
            r += r_step;
            f += f_step;
        }
    }
    attacks
}

pub fn pawn_attacks(pawns: Bitboard, side: usize) -> Bitboard {
    if side == crate::state::WHITE {
        let left = (pawns.0 & !FILE_A) << 7;
        let right = (pawns.0 & !FILE_H) << 9;
        Bitboard(left | right)
    } else {
        let left = (pawns.0 & !FILE_A) >> 9;
        let right = (pawns.0 & !FILE_H) >> 7;
        Bitboard(left | right)
    }
}
