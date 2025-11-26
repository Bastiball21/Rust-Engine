use crate::bitboard::Random;

// --- ZOBRIST KEYS ---
// We store them globally.
// [Piece Type][Square]
pub static mut PIECE_KEYS: [[u64; 64]; 12] = [[0; 64]; 12];
// [Castling Rights]
pub static mut CASTLING_KEYS: [u64; 16] = [0; 16];
// [File 0-7]
pub static mut EN_PASSANT_KEYS: [u64; 8] = [0; 8];
// Black to move
pub static mut SIDE_KEY: u64 = 0;

pub fn init_zobrist() {
    // Use a FIXED seed so the engine always plays the same way
    let mut rng = Random::new(1070372); 

    unsafe {
        for piece in 0..12 {
            for square in 0..64 {
                PIECE_KEYS[piece][square] = rng.get_u64();
            }
        }
        for i in 0..16 { CASTLING_KEYS[i] = rng.get_u64(); }
        for i in 0..8 { EN_PASSANT_KEYS[i] = rng.get_u64(); }
        SIDE_KEY = rng.get_u64();
    }
    println!("Zobrist Keys Initialized.");
}