use std::fmt;
use std::fs::File;
use std::io::{Write, BufReader, BufWriter, BufRead};
use std::sync::OnceLock;

// --- GLOBAL CONSTANTS ---
pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_H: u64 = 0x8080808080808080;
pub const RANK_1: u64 = 0x00000000000000FF;
pub const RANK_8: u64 = 0xFF00000000000000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bitboard(pub u64);

// --- SAFE GLOBAL STORAGE ---

// 1. Magics Storage
static ROOK_MAGICS: OnceLock<[u64; 64]> = OnceLock::new();
static ROOK_SHIFTS: OnceLock<[u8; 64]> = OnceLock::new();
static ROOK_TABLE: OnceLock<Vec<Vec<Bitboard>>> = OnceLock::new();

static BISHOP_MAGICS: OnceLock<[u64; 64]> = OnceLock::new();
static BISHOP_SHIFTS: OnceLock<[u8; 64]> = OnceLock::new();
static BISHOP_TABLE: OnceLock<Vec<Vec<Bitboard>>> = OnceLock::new();

// 2. Evaluation Masks Storage (Bundled in a struct to fix type errors)
struct EvalData {
    file_masks: [Bitboard; 64],
    rank_masks: [Bitboard; 64],
    passed_pawn_masks: [[Bitboard; 64]; 2],
    king_zone_masks: [Bitboard; 64],
}

static EVAL_DATA: OnceLock<EvalData> = OnceLock::new();

// --- PUBLIC SAFE GETTERS ---
pub fn file_mask(sq: usize) -> Bitboard {
    EVAL_DATA.get_or_init(init_eval_data).file_masks[sq]
}
pub fn rank_mask(sq: usize) -> Bitboard {
    EVAL_DATA.get_or_init(init_eval_data).rank_masks[sq]
}
pub fn passed_pawn_mask(side: usize, sq: usize) -> Bitboard {
    EVAL_DATA.get_or_init(init_eval_data).passed_pawn_masks[side][sq]
}
pub fn king_zone_mask(sq: usize) -> Bitboard {
    EVAL_DATA.get_or_init(init_eval_data).king_zone_masks[sq]
}

impl Bitboard {
    pub fn new() -> Self { Bitboard(0) }
    #[inline(always)] pub fn set_bit(&mut self, square: u8) { self.0 |= 1u64 << square; }
    #[inline(always)] pub fn get_bit(&self, square: u8) -> bool { (self.0 & (1u64 << square)) != 0 }
    #[inline(always)] pub fn pop_bit(&mut self, square: u8) { self.0 &= !(1u64 << square); }
    #[inline(always)] pub fn count_bits(&self) -> u32 { self.0.count_ones() }
    #[inline(always)] pub fn get_lsb_index(&self) -> u32 { self.0.trailing_zeros() }

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

impl std::ops::BitOr for Bitboard { type Output = Self; fn bitor(self, rhs: Self) -> Self { Bitboard(self.0 | rhs.0) } }
impl std::ops::BitAnd for Bitboard { type Output = Self; fn bitand(self, rhs: Self) -> Self { Bitboard(self.0 & rhs.0) } }
impl std::ops::BitXor for Bitboard { type Output = Self; fn bitxor(self, rhs: Self) -> Self { Bitboard(self.0 ^ rhs.0) } }
impl std::ops::Not for Bitboard { type Output = Self; fn not(self) -> Self { Bitboard(!self.0) } }

// --- INITIALIZATION ---
pub fn init_magic_tables() {
    get_magics(); 
    EVAL_DATA.get_or_init(init_eval_data);
    println!("All Bitboard Tables Initialized (Thread-Safe).");
}

struct MagicData {
    r_magics: [u64; 64],
    r_shifts: [u8; 64],
    r_table: Vec<Vec<Bitboard>>,
    b_magics: [u64; 64],
    b_shifts: [u8; 64],
    b_table: Vec<Vec<Bitboard>>,
}

fn get_magics() {
    ROOK_MAGICS.get_or_init(|| {
        let data = compute_magics();
        let _ = ROOK_SHIFTS.set(data.r_shifts);
        let _ = ROOK_TABLE.set(data.r_table);
        let _ = BISHOP_MAGICS.set(data.b_magics);
        let _ = BISHOP_SHIFTS.set(data.b_shifts);
        let _ = BISHOP_TABLE.set(data.b_table);
        data.r_magics
    });
}

fn compute_magics() -> MagicData {
    let mut r_magics = [0u64; 64];
    let mut b_magics = [0u64; 64];
    
    let loaded = if let Ok(file) = File::open("magics.dat") {
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut success = true;
        for i in 0..64 {
            if let Some(Ok(line)) = lines.next() { r_magics[i] = line.parse().unwrap_or(0); } else { success = false; }
        }
        for i in 0..64 {
            if let Some(Ok(line)) = lines.next() { b_magics[i] = line.parse().unwrap_or(0); } else { success = false; }
        }
        if success { println!("Magic Bitboards loaded from file."); }
        success
    } else { false };

    if !loaded {
        println!("Calculating Magics...");
        for square in 0..64 {
            let r_mask = mask_rook_attacks(square);
            r_magics[square as usize] = find_magic_number(square, r_mask, r_mask.count_bits(), false);
            let b_mask = mask_bishop_attacks(square);
            b_magics[square as usize] = find_magic_number(square, b_mask, b_mask.count_bits(), true);
        }
        if let Ok(file) = File::create("magics.dat") {
            let mut writer = BufWriter::new(file);
            for m in r_magics { writeln!(writer, "{}", m).unwrap(); }
            for m in b_magics { writeln!(writer, "{}", m).unwrap(); }
        }
    }

    let mut r_shifts = [0u8; 64];
    let mut r_table = Vec::with_capacity(64);
    let mut b_shifts = [0u8; 64];
    let mut b_table = Vec::with_capacity(64);

    for square in 0..64 {
        let r_mask = mask_rook_attacks(square);
        let r_shift = 64 - r_mask.count_bits();
        r_shifts[square as usize] = r_shift as u8;
        let r_vars = 1 << r_mask.count_bits();
        let mut r_entries = vec![Bitboard(0); r_vars];
        for i in 0..r_vars {
            let occ = set_occupancy(i as u32, r_mask.count_bits(), r_mask);
            let att = generate_rook_attacks_on_the_fly(square, occ);
            let idx = (occ.0.wrapping_mul(r_magics[square as usize])) >> r_shift;
            r_entries[idx as usize] = att;
        }
        r_table.push(r_entries);

        let b_mask = mask_bishop_attacks(square);
        let b_shift = 64 - b_mask.count_bits();
        b_shifts[square as usize] = b_shift as u8;
        let b_vars = 1 << b_mask.count_bits();
        let mut b_entries = vec![Bitboard(0); b_vars];
        for i in 0..b_vars {
            let occ = set_occupancy(i as u32, b_mask.count_bits(), b_mask);
            let att = generate_bishop_attacks_on_the_fly(square, occ);
            let idx = (occ.0.wrapping_mul(b_magics[square as usize])) >> b_shift;
            b_entries[idx as usize] = att;
        }
        b_table.push(b_entries);
    }

    MagicData { r_magics, r_shifts, r_table, b_magics, b_shifts, b_table }
}

fn init_eval_data() -> EvalData {
    let mut file_masks = [Bitboard(0); 64];
    let mut rank_masks = [Bitboard(0); 64];
    let mut passed_pawn_masks = [[Bitboard(0); 64]; 2];
    let mut king_zone_masks = [Bitboard(0); 64];

    for r in 0..8 {
        for f in 0..8 {
            let sq = r * 8 + f;
            for r2 in 0..8 { file_masks[sq].set_bit((r2 * 8 + f) as u8); }
            for f2 in 0..8 { rank_masks[sq].set_bit((r * 8 + f2) as u8); }

            let mut w_passed = Bitboard(0);
            for r2 in (r + 1)..8 {
                w_passed.set_bit((r2 * 8 + f) as u8);
                if f > 0 { w_passed.set_bit((r2 * 8 + (f - 1)) as u8); }
                if f < 7 { w_passed.set_bit((r2 * 8 + (f + 1)) as u8); }
            }
            passed_pawn_masks[0][sq] = w_passed;

            let mut b_passed = Bitboard(0);
            for r2 in 0..r {
                b_passed.set_bit((r2 * 8 + f) as u8);
                if f > 0 { b_passed.set_bit((r2 * 8 + (f - 1)) as u8); }
                if f < 7 { b_passed.set_bit((r2 * 8 + (f + 1)) as u8); }
            }
            passed_pawn_masks[1][sq] = b_passed;

            let mut zone = Bitboard(0);
            let king_bb = 1u64 << sq;
            let attacks = mask_king_attacks(sq as u8);
            zone.0 |= attacks.0 | king_bb;
            if r < 6 { zone.0 |= (zone.0 << 8) | (zone.0 << 16); }
            king_zone_masks[sq] = zone;
        }
    }
    
    EvalData { file_masks, rank_masks, passed_pawn_masks, king_zone_masks }
}

#[inline(always)]
pub fn get_rook_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    let magics = ROOK_MAGICS.get().unwrap_or_else(|| { get_magics(); ROOK_MAGICS.get().unwrap() });
    let shifts = ROOK_SHIFTS.get().unwrap();
    let table = ROOK_TABLE.get().unwrap();
    
    let magic = magics[square as usize];
    let shift = shifts[square as usize];
    let mask = mask_rook_attacks(square); 
    let index = ((occupancy.0 & mask.0).wrapping_mul(magic)) >> shift;
    table[square as usize][index as usize]
}

#[inline(always)]
pub fn get_bishop_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    let magics = BISHOP_MAGICS.get().unwrap_or_else(|| { get_magics(); BISHOP_MAGICS.get().unwrap() });
    let shifts = BISHOP_SHIFTS.get().unwrap();
    let table = BISHOP_TABLE.get().unwrap();
    
    let magic = magics[square as usize];
    let shift = shifts[square as usize];
    let mask = mask_bishop_attacks(square);
    let index = ((occupancy.0 & mask.0).wrapping_mul(magic)) >> shift;
    table[square as usize][index as usize]
}

#[inline(always)]
pub fn get_queen_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    get_rook_attacks(square, occupancy) | get_bishop_attacks(square, occupancy)
}

pub fn mask_rook_attacks(square: u8) -> Bitboard {
    let mut attacks = Bitboard(0);
    let r = square / 8;
    let f = square % 8;
    for r_itr in 1..7 { if r_itr != r { attacks.set_bit(r_itr * 8 + f); } }
    for f_itr in 1..7 { if f_itr != f { attacks.set_bit(r * 8 + f_itr); } }
    attacks
}

pub fn mask_bishop_attacks(square: u8) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8;
    let file = (square % 8) as i8;
    let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for (r_step, f_step) in directions {
        let mut r = rank + r_step; let mut f = file + f_step;
        while r > 0 && r < 7 && f > 0 && f < 7 {
            attacks.set_bit((r * 8 + f) as u8);
            r += r_step; f += f_step;
        }
    }
    attacks
}

pub fn mask_knight_attacks(square: u8) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8; let file = (square % 8) as i8;
    let jumps = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)];
    for (r_off, f_off) in jumps {
        let tr = rank + r_off; let tf = file + f_off;
        if tr >= 0 && tr <= 7 && tf >= 0 && tf <= 7 { attacks.set_bit((tr * 8 + tf) as u8); }
    }
    attacks
}

pub fn mask_king_attacks(square: u8) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8; let file = (square % 8) as i8;
    let steps = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];
    for (r_off, f_off) in steps {
        let tr = rank + r_off; let tf = file + f_off;
        if tr >= 0 && tr <= 7 && tf >= 0 && tf <= 7 { attacks.set_bit((tr * 8 + tf) as u8); }
    }
    attacks
}

pub fn generate_rook_attacks_on_the_fly(square: u8, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8; let file = (square % 8) as i8;
    let directions = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for (r_step, f_step) in directions {
        let mut r = rank + r_step; let mut f = file + f_step;
        while r >= 0 && r <= 7 && f >= 0 && f <= 7 {
            let target = (r * 8 + f) as u8;
            attacks.set_bit(target);
            if blockers.get_bit(target) { break; }
            r += r_step; f += f_step;
        }
    }
    attacks
}

pub fn generate_bishop_attacks_on_the_fly(square: u8, blockers: Bitboard) -> Bitboard {
    let mut attacks = Bitboard(0);
    let rank = (square / 8) as i8; let file = (square % 8) as i8;
    let directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for (r_step, f_step) in directions {
        let mut r = rank + r_step; let mut f = file + f_step;
        while r >= 0 && r <= 7 && f >= 0 && f <= 7 {
            let target = (r * 8 + f) as u8;
            attacks.set_bit(target);
            if blockers.get_bit(target) { break; }
            r += r_step; f += f_step;
        }
    }
    attacks
}

pub struct Random { pub state: u32 }
impl Random {
    pub fn new(seed: u32) -> Self { Random { state: seed } }
    pub fn get_u32(&mut self) -> u32 {
        let mut x = self.state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; self.state = x; x
    }
    pub fn get_u64(&mut self) -> u64 {
        let n1 = self.get_u32() as u64; let n2 = self.get_u32() as u64;
        let n3 = self.get_u32() as u64; let n4 = self.get_u32() as u64;
        n1 | (n2 << 16) | (n3 << 32) | (n4 << 48)
    }
    pub fn get_magic_candidate(&mut self) -> u64 {
        self.get_u64() & self.get_u64() & self.get_u64()
    }
}

pub fn set_occupancy(index: u32, bits_in_mask: u32, mut attack_mask: Bitboard) -> Bitboard {
    let mut occupancy = Bitboard(0);
    for count in 0..bits_in_mask {
        let square = attack_mask.get_lsb_index();
        attack_mask.pop_bit(square as u8);
        if (index & (1 << count)) != 0 { occupancy.set_bit(square as u8); }
    }
    occupancy
}

pub fn find_magic_number(square: u8, mask: Bitboard, bits: u32, is_bishop: bool) -> u64 {
    let variations = 1 << bits;
    let mut occupancies = vec![Bitboard(0); variations];
    let mut attacks = vec![Bitboard(0); variations];
    
    for i in 0..variations {
        occupancies[i] = set_occupancy(i as u32, bits, mask);
        attacks[i] = if is_bishop { generate_bishop_attacks_on_the_fly(square, occupancies[i]) } 
                     else { generate_rook_attacks_on_the_fly(square, occupancies[i]) };
    }
    
    let mut seed_rotator = 0;
    loop {
        let mut rng = Random::new(1804289383 + (square as u32 * 1000) + seed_rotator);
        seed_rotator += 1;
        for _ in 0..1_000_000 {
            let magic = rng.get_magic_candidate();
            let mut used = vec![Bitboard(0); variations];
            let mut fail = false;
            for i in 0..variations {
                let idx = (occupancies[i].0.wrapping_mul(magic)) >> (64 - bits);
                if used[idx as usize].0 == 0 { used[idx as usize] = attacks[i]; }
                else if used[idx as usize] != attacks[i] { fail = true; break; }
            }
            if !fail { return magic; }
        }
    }
}