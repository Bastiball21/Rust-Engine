use std::sync::OnceLock;

// --- GLOBAL CONSTANTS ---
pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_H: u64 = 0x8080808080808080;
pub const RANK_1: u64 = 0x00000000000000FF;
pub const RANK_8: u64 = 0xFF00000000000000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bitboard(pub u64);

// --- THREAD-SAFE GLOBAL TABLES (Using OnceLock) ---
static ROOK_MAGICS: OnceLock<[u64; 64]> = OnceLock::new();
static ROOK_SHIFTS: OnceLock<[u8; 64]> = OnceLock::new();
static ROOK_ATTACKS: OnceLock<Vec<Vec<Bitboard>>> = OnceLock::new(); // Vector of vectors for safety

static BISHOP_MAGICS: OnceLock<[u64; 64]> = OnceLock::new();
static BISHOP_SHIFTS: OnceLock<[u8; 64]> = OnceLock::new();
static BISHOP_ATTACKS: OnceLock<Vec<Vec<Bitboard>>> = OnceLock::new();

// Evaluation Masks
static FILE_MASKS: OnceLock<[Bitboard; 64]> = OnceLock::new();
static RANK_MASKS: OnceLock<[Bitboard; 64]> = OnceLock::new();
static PASSED_PAWN_MASKS: OnceLock<[[Bitboard; 64]; 2]> = OnceLock::new();
static KING_ZONE_MASKS: OnceLock<[Bitboard; 64]> = OnceLock::new();

// --- SAFE ACCESSORS ---
// Used by eval.rs and others
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

// Operator Overloads
impl std::ops::BitOr for Bitboard { type Output = Self; fn bitor(self, rhs: Self) -> Self { Bitboard(self.0 | rhs.0) } }
impl std::ops::BitAnd for Bitboard { type Output = Self; fn bitand(self, rhs: Self) -> Self { Bitboard(self.0 & rhs.0) } }
impl std::ops::BitXor for Bitboard { type Output = Self; fn bitxor(self, rhs: Self) -> Self { Bitboard(self.0 ^ rhs.0) } }
impl std::ops::Not for Bitboard { type Output = Self; fn not(self) -> Self { Bitboard(!self.0) } }

// --- INITIALIZATION ---
pub fn init_magic_tables() {
    // 1. Initialize simple eval masks
    init_eval_masks();

    // 2. Initialize Magics (Runtime Generation)
    // This removes the need for magics.dat
    println!("Initializing Magics (Runtime)... please wait.");
    
    let mut rng = Random::new(1804289383);

    // Initialize Rook Tables
    let mut r_magics = [0u64; 64];
    let mut r_shifts = [0u8; 64];
    let mut r_attacks = Vec::with_capacity(64);

    for square in 0..64 {
        let mask = mask_rook_attacks(square);
        let bits = mask.count_bits();
        r_shifts[square as usize] = 64 - bits as u8;
        
        // Find magic
        loop {
            let magic = rng.get_magic_candidate();
            if  (mask.0.wrapping_mul(magic) & 0xFF00000000000000).count_ones() < 6 { continue; } // Optimization filter
            
            let (valid, table) = try_magic(square, mask, bits, magic, true);
            if valid {
                r_magics[square as usize] = magic;
                r_attacks.push(table);
                break;
            }
        }
    }
    
    // Store Rook Data
    ROOK_MAGICS.set(r_magics).expect("Failed to set Rook Magics");
    ROOK_SHIFTS.set(r_shifts).expect("Failed to set Rook Shifts");
    ROOK_ATTACKS.set(r_attacks).expect("Failed to set Rook Attacks");

    // Initialize Bishop Tables
    let mut b_magics = [0u64; 64];
    let mut b_shifts = [0u8; 64];
    let mut b_attacks = Vec::with_capacity(64);

    for square in 0..64 {
        let mask = mask_bishop_attacks(square);
        let bits = mask.count_bits();
        b_shifts[square as usize] = 64 - bits as u8;

        loop {
            let magic = rng.get_magic_candidate();
             if  (mask.0.wrapping_mul(magic) & 0xFF00000000000000).count_ones() < 6 { continue; } 
             
            let (valid, table) = try_magic(square, mask, bits, magic, false);
            if valid {
                b_magics[square as usize] = magic;
                b_attacks.push(table);
                break;
            }
        }
    }

    // Store Bishop Data
    BISHOP_MAGICS.set(b_magics).expect("Failed to set Bishop Magics");
    BISHOP_SHIFTS.set(b_shifts).expect("Failed to set Bishop Shifts");
    BISHOP_ATTACKS.set(b_attacks).expect("Failed to set Bishop Attacks");

    println!("All Tables Ready.");
}

// Helper to test a candidate magic number
fn try_magic(square: u8, mask: Bitboard, bits: u32, magic: u64, is_rook: bool) -> (bool, Vec<Bitboard>) {
    let size = 1 << bits;
    let mut table = vec![Bitboard(0); size];
    let mut used = vec![false; size];

    // Generate all occupancies
    // We iterate through all subsets of the mask
    let mut index = 0;
    
    // A trick to iterate all submasks of a mask
    let mut occupancy = Bitboard(0);
    loop {
        let attacks = if is_rook {
            generate_rook_attacks_on_the_fly(square, occupancy)
        } else {
            generate_bishop_attacks_on_the_fly(square, occupancy)
        };

        let magic_index = ((occupancy.0.wrapping_mul(magic)) >> (64 - bits)) as usize;
        
        if !used[magic_index] {
            used[magic_index] = true;
            table[magic_index] = attacks;
        } else if table[magic_index] != attacks {
            // Collision detected with different result -> Invalid magic
            return (false, Vec::new());
        }

        // Next occupancy
        occupancy.0 = occupancy.0.wrapping_sub(mask.0) & mask.0;
        if occupancy.0 == 0 { break; }
        index += 1;
    }
    
    (true, table)
}

fn init_eval_masks() {
    let mut f_masks = [Bitboard(0); 64];
    let mut r_masks = [Bitboard(0); 64];
    let mut passed = [[Bitboard(0); 64]; 2];
    let mut k_zones = [Bitboard(0); 64];

    for r in 0..8 {
        for f in 0..8 {
            let sq = r * 8 + f;
            for r2 in 0..8 { f_masks[sq].set_bit((r2 * 8 + f) as u8); }
            for f2 in 0..8 { r_masks[sq].set_bit((r * 8 + f2) as u8); }

            // Passed Pawn Masks
            let mut w_passed = Bitboard(0);
            for r2 in (r + 1)..8 {
                w_passed.set_bit((r2 * 8 + f) as u8);
                if f > 0 { w_passed.set_bit((r2 * 8 + (f - 1)) as u8); }
                if f < 7 { w_passed.set_bit((r2 * 8 + (f + 1)) as u8); }
            }
            passed[0][sq] = w_passed;

            let mut b_passed = Bitboard(0);
            for r2 in 0..r {
                b_passed.set_bit((r2 * 8 + f) as u8);
                if f > 0 { b_passed.set_bit((r2 * 8 + (f - 1)) as u8); }
                if f < 7 { b_passed.set_bit((r2 * 8 + (f + 1)) as u8); }
            }
            passed[1][sq] = b_passed;

            // King Zones
            let mut zone = Bitboard(0);
            let king_bb = 1u64 << sq;
            let attacks = mask_king_attacks(sq as u8);
            zone.0 |= attacks.0 | king_bb;
            if r < 6 { zone.0 |= (zone.0 << 8) | (zone.0 << 16); } // Extend forward
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
    let magics = ROOK_MAGICS.get().unwrap(); // Fast access, assumes init called
    let shifts = ROOK_SHIFTS.get().unwrap();
    let attacks = ROOK_ATTACKS.get().unwrap();
    
    let magic = magics[square as usize];
    let shift = shifts[square as usize];
    let mask = mask_rook_attacks(square);
    
    let idx = ((occupancy.0 & mask.0).wrapping_mul(magic)) >> shift;
    attacks[square as usize][idx as usize]
}

#[inline(always)]
pub fn get_bishop_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    let magics = BISHOP_MAGICS.get().unwrap();
    let shifts = BISHOP_SHIFTS.get().unwrap();
    let attacks = BISHOP_ATTACKS.get().unwrap();
    
    let magic = magics[square as usize];
    let shift = shifts[square as usize];
    let mask = mask_bishop_attacks(square);
    
    let idx = ((occupancy.0 & mask.0).wrapping_mul(magic)) >> shift;
    attacks[square as usize][idx as usize]
}

#[inline(always)]
pub fn get_queen_attacks(square: u8, occupancy: Bitboard) -> Bitboard {
    get_rook_attacks(square, occupancy) | get_bishop_attacks(square, occupancy)
}

// --- BASIC MASKS & GENERATORS ---

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

// PRNG for Runtime Magic Generation
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