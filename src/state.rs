#![allow(non_upper_case_globals)] // Fixes the warnings about p, n, b, etc.

use crate::bitboard::Bitboard;
use crate::zobrist::{PIECE_KEYS, SIDE_KEY, CASTLING_KEYS, EN_PASSANT_KEYS};

// --- CONSTANTS ---
pub const P: usize = 0; pub const N: usize = 1; pub const B: usize = 2; 
pub const R: usize = 3; pub const Q: usize = 4; pub const K: usize = 5;
pub const p: usize = 6; pub const n: usize = 7; pub const b: usize = 8; 
pub const r: usize = 9; pub const q: usize = 10; pub const k: usize = 11;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;
pub const BOTH:  usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Move {
    pub source: u8,
    pub target: u8,
    pub promotion: Option<usize>,
    pub is_capture: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct GameState {
    pub bitboards: [Bitboard; 12],
    pub occupancies: [Bitboard; 3], 
    pub side_to_move: usize,
    pub castling_rights: u8,
    pub en_passant: u8,
    pub hash: u64,
}

impl GameState {
    pub fn new() -> Self {
        GameState {
            bitboards: [Bitboard(0); 12],
            occupancies: [Bitboard(0); 3],
            side_to_move: 0,
            castling_rights: 0,
            en_passant: 64,
            hash: 0,
        }
    }

    pub fn compute_hash(&mut self) {
        unsafe {
            let mut h = 0;
            for piece in 0..12 {
                let mut bb = self.bitboards[piece];
                while bb.0 != 0 {
                    let sq = bb.get_lsb_index();
                    bb.pop_bit(sq as u8);
                    h ^= PIECE_KEYS[piece][sq as usize];
                }
            }
            h ^= CASTLING_KEYS[self.castling_rights as usize];
            if self.side_to_move == BLACK { h ^= SIDE_KEY; }
            if self.en_passant != 64 {
                let file = (self.en_passant % 8) as usize;
                h ^= EN_PASSANT_KEYS[file];
            }
            self.hash = h;
        }
    }

    pub fn parse_fen(fen: &str) -> GameState {
        let mut state = GameState::new();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let board_part = parts[0];
        let mut rank = 7;
        let mut file = 0;

        for char in board_part.chars() {
            if char == '/' { rank -= 1; file = 0; }
            else if char.is_digit(10) { file += char.to_digit(10).unwrap(); }
            else {
                let square = rank * 8 + file;
                let piece = match char {
                    'P'=>P, 'N'=>N, 'B'=>B, 'R'=>R, 'Q'=>Q, 'K'=>K,
                    'p'=>p, 'n'=>n, 'b'=>b, 'r'=>r, 'q'=>q, 'k'=>k, _=>0
                };
                state.bitboards[piece].set_bit(square as u8);
                file += 1;
            }
        }
        if parts[1] == "b" { state.side_to_move = BLACK; }
        
        if parts[2] != "-" {
            for c in parts[2].chars() {
                match c {
                    'K' => state.castling_rights |= 1,
                    'Q' => state.castling_rights |= 2,
                    'k' => state.castling_rights |= 4,
                    'q' => state.castling_rights |= 8,
                    _ => {}
                }
            }
        }

        for piece in P..=K { state.occupancies[WHITE] = state.occupancies[WHITE] | state.bitboards[piece]; }
        for piece in p..=k { state.occupancies[BLACK] = state.occupancies[BLACK] | state.bitboards[piece]; }
        state.occupancies[BOTH] = state.occupancies[WHITE] | state.occupancies[BLACK];
        
        state.compute_hash();
        state
    }

    // --- MAKE NULL MOVE (For Search Heuristics) ---
    pub fn make_null_move(&self) -> GameState {
        let mut new_state = *self;
        
        // 1. Swap Side
        new_state.side_to_move = 1 - self.side_to_move;
        unsafe { new_state.hash ^= SIDE_KEY; }

        // 2. Reset En Passant (It expires)
        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as usize;
            unsafe { new_state.hash ^= EN_PASSANT_KEYS[file]; }
            new_state.en_passant = 64;
        }

        new_state
    }

    // --- MAKE MOVE ---
    pub fn make_move(&self, mv: Move) -> GameState {
        let mut new_state = *self;
        let side = self.side_to_move;
        
        let mut piece_type = 12;
        let start_range = if side == WHITE { P } else { p };
        let end_range   = if side == WHITE { K } else { k };

        for pp in start_range..=end_range {
            if self.bitboards[pp].get_bit(mv.source) {
                piece_type = pp;
                break;
            }
        }

        unsafe {
            new_state.hash ^= PIECE_KEYS[piece_type][mv.source as usize];
            new_state.bitboards[piece_type].pop_bit(mv.source);
            
            new_state.hash ^= PIECE_KEYS[piece_type][mv.target as usize];
            new_state.bitboards[piece_type].set_bit(mv.target);

            if mv.is_capture {
                let enemy_start = if side == WHITE { p } else { P };
                let enemy_end   = if side == WHITE { k } else { K };
                
                if (piece_type == P || piece_type == p) && mv.target == self.en_passant {
                    let cap_sq = if side == WHITE { mv.target - 8 } else { mv.target + 8 };
                    let enemy_pawn = if side == WHITE { p } else { P };
                    new_state.bitboards[enemy_pawn].pop_bit(cap_sq);
                    new_state.hash ^= PIECE_KEYS[enemy_pawn][cap_sq as usize];
                } else {
                    for pp in enemy_start..=enemy_end {
                        if new_state.bitboards[pp].get_bit(mv.target) {
                            new_state.bitboards[pp].pop_bit(mv.target);
                            new_state.hash ^= PIECE_KEYS[pp][mv.target as usize];
                            break;
                        }
                    }
                }
            }

            if let Some(promo) = mv.promotion {
                new_state.bitboards[piece_type].pop_bit(mv.target);
                new_state.hash ^= PIECE_KEYS[piece_type][mv.target as usize]; 

                let actual_promo = if side == WHITE { promo } else { promo + 6 };
                new_state.bitboards[actual_promo].set_bit(mv.target);
                new_state.hash ^= PIECE_KEYS[actual_promo][mv.target as usize];
            }

            if (piece_type == K || piece_type == k) && (mv.target as i8 - mv.source as i8).abs() == 2 {
                if mv.target == 6 { 
                    new_state.bitboards[R].pop_bit(7); new_state.hash ^= PIECE_KEYS[R][7];
                    new_state.bitboards[R].set_bit(5); new_state.hash ^= PIECE_KEYS[R][5];
                }
                else if mv.target == 2 { 
                    new_state.bitboards[R].pop_bit(0); new_state.hash ^= PIECE_KEYS[R][0];
                    new_state.bitboards[R].set_bit(3); new_state.hash ^= PIECE_KEYS[R][3];
                }
                else if mv.target == 62 { 
                    new_state.bitboards[r].pop_bit(63); new_state.hash ^= PIECE_KEYS[r][63];
                    new_state.bitboards[r].set_bit(61); new_state.hash ^= PIECE_KEYS[r][61];
                }
                else if mv.target == 58 { 
                    new_state.bitboards[r].pop_bit(56); new_state.hash ^= PIECE_KEYS[r][56];
                    new_state.bitboards[r].set_bit(59); new_state.hash ^= PIECE_KEYS[r][59];
                }
            }

            if self.en_passant != 64 {
                let file = (self.en_passant % 8) as usize;
                new_state.hash ^= EN_PASSANT_KEYS[file]; 
            }
            
            new_state.en_passant = 64; 
            if piece_type == P || piece_type == p {
                let diff = (mv.target as i8 - mv.source as i8).abs();
                if diff == 16 {
                    let ep_sq = if side == WHITE { mv.target - 8 } else { mv.target + 8 };
                    new_state.en_passant = ep_sq;
                    new_state.hash ^= EN_PASSANT_KEYS[(ep_sq % 8) as usize]; 
                }
            }

            new_state.hash ^= CASTLING_KEYS[new_state.castling_rights as usize]; 
            if piece_type == K { new_state.castling_rights &= !3; }
            if piece_type == k { new_state.castling_rights &= !12; }
            new_state.hash ^= CASTLING_KEYS[new_state.castling_rights as usize]; 

            new_state.side_to_move = 1 - side;
            new_state.hash ^= SIDE_KEY;
        }

        new_state.occupancies = [Bitboard(0); 3];
        for pp in P..=K { new_state.occupancies[WHITE] = new_state.occupancies[WHITE] | new_state.bitboards[pp]; }
        for pp in p..=k { new_state.occupancies[BLACK] = new_state.occupancies[BLACK] | new_state.bitboards[pp]; }
        new_state.occupancies[BOTH] = new_state.occupancies[WHITE] | new_state.occupancies[BLACK];

        new_state
    }
}