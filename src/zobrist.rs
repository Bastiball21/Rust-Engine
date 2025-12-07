// src/zobrist.rs
use std::sync::OnceLock;

// Safe globals
pub static PIECE_KEYS: OnceLock<[[u64; 64]; 12]> = OnceLock::new();
pub static CASTLING_KEYS: OnceLock<[u64; 16]> = OnceLock::new();
pub static EN_PASSANT_KEYS: OnceLock<[u64; 8]> = OnceLock::new();
pub static SIDE_KEY: OnceLock<u64> = OnceLock::new();

// Simple PRNG struct local to this module
struct PRNG {
    state: u32,
}

impl PRNG {
    fn new(seed: u32) -> Self {
        PRNG { state: seed }
    }
    fn get_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.state = x;
        x
    }
    fn get_u64(&mut self) -> u64 {
        let n1 = self.get_u32() as u64;
        let n2 = self.get_u32() as u64;
        let n3 = self.get_u32() as u64;
        let n4 = self.get_u32() as u64;
        n1 | (n2 << 16) | (n3 << 32) | (n4 << 48)
    }
}

pub fn init_zobrist() {
    let mut rng = PRNG::new(1070372);

    let mut p_keys = [[0; 64]; 12];
    for piece in 0..12 {
        for square in 0..64 {
            p_keys[piece][square] = rng.get_u64();
        }
    }
    let _ = PIECE_KEYS.set(p_keys);

    let mut c_keys = [0; 16];
    for i in 0..16 { c_keys[i] = rng.get_u64(); }
    let _ = CASTLING_KEYS.set(c_keys);

    let mut ep_keys = [0; 8];
    for i in 0..8 { ep_keys[i] = rng.get_u64(); }
    let _ = EN_PASSANT_KEYS.set(ep_keys);

    let _ = SIDE_KEY.set(rng.get_u64());

    println!("Zobrist Keys Initialized.");
}

// Accessors
#[inline(always)]
pub fn piece_key(piece: usize, sq: usize) -> u64 {
    PIECE_KEYS.get().unwrap()[piece][sq]
}
#[inline(always)]
pub fn castling_key(rights: u8) -> u64 {
    CASTLING_KEYS.get().unwrap()[rights as usize]
}
#[inline(always)]
pub fn en_passant_key(file: u8) -> u64 {
    EN_PASSANT_KEYS.get().unwrap()[file as usize]
}
#[inline(always)]
pub fn side_key() -> u64 {
    *SIDE_KEY.get().unwrap()
}