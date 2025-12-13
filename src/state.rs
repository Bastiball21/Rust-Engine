// src/state.rs
#![allow(non_upper_case_globals)]

use crate::bitboard::Bitboard;
use crate::nnue::Accumulator;
use crate::zobrist;
use smallvec::SmallVec;

pub const P: usize = 0;
pub const N: usize = 1;
pub const B: usize = 2;
pub const R: usize = 3;
pub const Q: usize = 4;
pub const K: usize = 5;
pub const p: usize = 6;
pub const n: usize = 7;
pub const b: usize = 8;
pub const r: usize = 9;
pub const q: usize = 10;
pub const k: usize = 11;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;
pub const BOTH: usize = 2;

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
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
    pub accumulator: [Accumulator; 2],
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
            halfmove_clock: 0,
            fullmove_number: 1,
            accumulator: [Accumulator::default(); 2],
        }
    }

    pub fn compute_hash(&mut self) {
        let mut h = 0;
        for piece in 0..12 {
            let mut bb = self.bitboards[piece];
            while bb.0 != 0 {
                let sq = bb.get_lsb_index();
                bb.pop_bit(sq as u8);
                h ^= zobrist::piece_key(piece, sq as usize);
            }
        }
        h ^= zobrist::castling_key(self.castling_rights);
        if self.side_to_move == BLACK {
            h ^= zobrist::side_key();
        }
        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as u8;
            h ^= zobrist::en_passant_key(file);
        }
        self.hash = h;
    }

    pub fn refresh_accumulator(&mut self) {
        let bitboards = &self.bitboards;
        self.accumulator[0].refresh(bitboards, 0);
        self.accumulator[1].refresh(bitboards, 1);
    }

    pub fn parse_fen(fen: &str) -> GameState {
        let mut state = GameState::new();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let board_part = parts[0];
        let mut rank = 7;
        let mut file = 0;

        for char in board_part.chars() {
            if char == '/' {
                rank -= 1;
                file = 0;
            } else if char.is_digit(10) {
                file += char.to_digit(10).unwrap();
            } else {
                let square = rank * 8 + file;
                let piece = match char {
                    'P' => P,
                    'N' => N,
                    'B' => B,
                    'R' => R,
                    'Q' => Q,
                    'K' => K,
                    'p' => p,
                    'n' => n,
                    'b' => b,
                    'r' => r,
                    'q' => q,
                    'k' => k,
                    _ => 0,
                };
                state.bitboards[piece].set_bit(square as u8);
                file += 1;
            }
        }
        if parts[1] == "b" {
            state.side_to_move = BLACK;
        }

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

        if parts.len() > 3 && parts[3] != "-" {
            let chars: Vec<char> = parts[3].chars().collect();
            let f = chars[0] as u8 - b'a';
            let rank_idx = chars[1] as u8 - b'1';
            state.en_passant = rank_idx * 8 + f;
        }

        if parts.len() > 4 {
            state.halfmove_clock = parts[4].parse().unwrap_or(0);
        }

        if parts.len() > 5 {
            state.fullmove_number = parts[5].parse().unwrap_or(1);
        }

        for piece in P..=K {
            state.occupancies[WHITE] = state.occupancies[WHITE] | state.bitboards[piece];
        }
        for piece in p..=k {
            state.occupancies[BLACK] = state.occupancies[BLACK] | state.bitboards[piece];
        }
        state.occupancies[BOTH] = state.occupancies[WHITE] | state.occupancies[BLACK];

        state.compute_hash();
        state.refresh_accumulator();
        state
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::new();
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let sq = rank * 8 + file;
                let mut piece_char = ' ';

                if self.bitboards[P].get_bit(sq) {
                    piece_char = 'P';
                } else if self.bitboards[N].get_bit(sq) {
                    piece_char = 'N';
                } else if self.bitboards[B].get_bit(sq) {
                    piece_char = 'B';
                } else if self.bitboards[R].get_bit(sq) {
                    piece_char = 'R';
                } else if self.bitboards[Q].get_bit(sq) {
                    piece_char = 'Q';
                } else if self.bitboards[K].get_bit(sq) {
                    piece_char = 'K';
                } else if self.bitboards[p].get_bit(sq) {
                    piece_char = 'p';
                } else if self.bitboards[n].get_bit(sq) {
                    piece_char = 'n';
                } else if self.bitboards[b].get_bit(sq) {
                    piece_char = 'b';
                } else if self.bitboards[r].get_bit(sq) {
                    piece_char = 'r';
                } else if self.bitboards[q].get_bit(sq) {
                    piece_char = 'q';
                } else if self.bitboards[k].get_bit(sq) {
                    piece_char = 'k';
                }

                if piece_char == ' ' {
                    empty += 1;
                } else {
                    if empty > 0 {
                        fen.push_str(&empty.to_string());
                        empty = 0;
                    }
                    fen.push(piece_char);
                }
            }
            if empty > 0 {
                fen.push_str(&empty.to_string());
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        fen.push(' ');
        fen.push(if self.side_to_move == WHITE { 'w' } else { 'b' });
        fen.push(' ');

        let mut rights = String::new();
        if (self.castling_rights & 1) != 0 {
            rights.push('K');
        }
        if (self.castling_rights & 2) != 0 {
            rights.push('Q');
        }
        if (self.castling_rights & 4) != 0 {
            rights.push('k');
        }
        if (self.castling_rights & 8) != 0 {
            rights.push('q');
        }
        if rights.is_empty() {
            rights.push('-');
        }
        fen.push_str(&rights);

        fen.push(' ');
        if self.en_passant != 64 {
            let f = (b'a' + (self.en_passant % 8)) as char;
            let rank_char = (b'1' + (self.en_passant / 8)) as char;
            fen.push(f);
            fen.push(rank_char);
        } else {
            fen.push('-');
        }

        fen.push_str(&format!(
            " {} {}",
            self.halfmove_clock, self.fullmove_number
        ));
        fen
    }

    pub fn make_null_move(&self) -> GameState {
        let mut new_state = *self;
        new_state.side_to_move = 1 - self.side_to_move;
        new_state.hash ^= zobrist::side_key();
        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as u8;
            new_state.hash ^= zobrist::en_passant_key(file);
            new_state.en_passant = 64;
        }
        new_state.halfmove_clock = 0;

        if self.side_to_move == BLACK {
            new_state.fullmove_number += 1;
        }

        // Null move does not update accumulators (no pieces moved)
        new_state
    }

    pub fn make_move(&self, mv: Move) -> GameState {
        let mut new_state = *self;
        let side = self.side_to_move;

        // NNUE Incremental Update Tracking
        let mut added: SmallVec<[(usize, usize); 4]> = SmallVec::new();
        let mut removed: SmallVec<[(usize, usize); 4]> = SmallVec::new();

        if side == BLACK {
            new_state.fullmove_number += 1;
        }

        new_state.halfmove_clock += 1;

        let mut piece_type = 12;
        let start_range = if side == WHITE { P } else { p };
        let end_range = if side == WHITE { K } else { k };

        // Identify moving piece
        for pp in start_range..=end_range {
            if self.bitboards[pp].get_bit(mv.source) {
                piece_type = pp;
                break;
            }
        }

        // 1. Remove moving piece from source
        removed.push((piece_type, mv.source as usize));

        if piece_type == P || piece_type == p || mv.is_capture {
            new_state.halfmove_clock = 0;
        }

        new_state.hash ^= zobrist::piece_key(piece_type, mv.source as usize);
        new_state.bitboards[piece_type].pop_bit(mv.source);

        // 2. Add moving piece (or promo) to target
        let actual_piece = if let Some(promo) = mv.promotion {
            let p_idx = if side == WHITE { promo } else { promo + 6 };
            new_state.bitboards[piece_type].pop_bit(mv.target); // Should be empty unless capture? No, bitboard logic handles moves.
            // Wait, standard bitboard move logic:
            // Remove from source. Add to target.
            // If promo, remove source (type P), add target (type Promo).

            // Logic below handles target set_bit.
            p_idx
        } else {
            piece_type
        };
        added.push((actual_piece, mv.target as usize));


        new_state.hash ^= zobrist::piece_key(piece_type, mv.target as usize);
        new_state.bitboards[piece_type].set_bit(mv.target);

        if mv.is_capture {
            let enemy_start = if side == WHITE { p } else { P };
            let enemy_end = if side == WHITE { k } else { K };

            if (piece_type == P || piece_type == p) && mv.target == self.en_passant {
                // En Passant Capture
                let cap_sq = if side == WHITE {
                    mv.target - 8
                } else {
                    mv.target + 8
                };
                let enemy_pawn = if side == WHITE { p } else { P };
                new_state.bitboards[enemy_pawn].pop_bit(cap_sq);
                new_state.hash ^= zobrist::piece_key(enemy_pawn, cap_sq as usize);

                // NNUE: Remove captured pawn
                removed.push((enemy_pawn, cap_sq as usize));
            } else {
                // Normal Capture
                for pp in enemy_start..=enemy_end {
                    if new_state.bitboards[pp].get_bit(mv.target) {
                        // Note: We are checking new_state, but we just set the piece there?
                        // Actually, the logic in original make_move was:
                        // 1. pop source
                        // 2. set target
                        // 3. if capture, loop over enemies and pop target.
                        // So at this point, target has BOTH pieces?
                        // Yes, standard make_move implementation often overlaps bitboards briefly.

                        new_state.bitboards[pp].pop_bit(mv.target);
                        new_state.hash ^= zobrist::piece_key(pp, mv.target as usize);

                        // NNUE: Remove captured piece
                        removed.push((pp, mv.target as usize));
                        break;
                    }
                }
            }
        }

        if let Some(promo) = mv.promotion {
            // Fixup for promotion: we set P at target above (copying logic), now we swap it?
            // Original code:
            // new_state.bitboards[piece_type].set_bit(mv.target); (as P)
            // ...
            // if let Some(promo) ...
            //    pop P at target
            //    set Promo at target

            new_state.bitboards[piece_type].pop_bit(mv.target);
            new_state.hash ^= zobrist::piece_key(piece_type, mv.target as usize);

            let actual_promo = if side == WHITE { promo } else { promo + 6 };
            new_state.bitboards[actual_promo].set_bit(mv.target);
            new_state.hash ^= zobrist::piece_key(actual_promo, mv.target as usize);
        }

        if (piece_type == K || piece_type == k) && (mv.target as i8 - mv.source as i8).abs() == 2 {
            if mv.target == 6 {
                new_state.bitboards[R].pop_bit(7);
                new_state.hash ^= zobrist::piece_key(R, 7);
                new_state.bitboards[R].set_bit(5);
                new_state.hash ^= zobrist::piece_key(R, 5);

                removed.push((R, 7)); added.push((R, 5));
            } else if mv.target == 2 {
                new_state.bitboards[R].pop_bit(0);
                new_state.hash ^= zobrist::piece_key(R, 0);
                new_state.bitboards[R].set_bit(3);
                new_state.hash ^= zobrist::piece_key(R, 3);

                removed.push((R, 0)); added.push((R, 3));
            } else if mv.target == 62 {
                new_state.bitboards[r].pop_bit(63);
                new_state.hash ^= zobrist::piece_key(r, 63);
                new_state.bitboards[r].set_bit(61);
                new_state.hash ^= zobrist::piece_key(r, 61);

                removed.push((r, 63)); added.push((r, 61));
            } else if mv.target == 58 {
                new_state.bitboards[r].pop_bit(56);
                new_state.hash ^= zobrist::piece_key(r, 56);
                new_state.bitboards[r].set_bit(59);
                new_state.hash ^= zobrist::piece_key(r, 59);

                removed.push((r, 56)); added.push((r, 59));
            }
        }

        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as u8;
            new_state.hash ^= zobrist::en_passant_key(file);
        }

        new_state.en_passant = 64;
        if piece_type == P || piece_type == p {
            let diff = (mv.target as i8 - mv.source as i8).abs();
            if diff == 16 {
                let ep_sq = if side == WHITE {
                    mv.target - 8
                } else {
                    mv.target + 8
                };
                new_state.en_passant = ep_sq;
                new_state.hash ^= zobrist::en_passant_key((ep_sq % 8) as u8);
            }
        }

        new_state.hash ^= zobrist::castling_key(new_state.castling_rights);

        if piece_type == K {
            new_state.castling_rights &= !3;
        }
        if piece_type == k {
            new_state.castling_rights &= !12;
        }

        if mv.source == 7 || mv.target == 7 {
            new_state.castling_rights &= !1;
        }
        if mv.source == 0 || mv.target == 0 {
            new_state.castling_rights &= !2;
        }

        if mv.source == 63 || mv.target == 63 {
            new_state.castling_rights &= !4;
        }
        if mv.source == 56 || mv.target == 56 {
            new_state.castling_rights &= !8;
        }

        new_state.hash ^= zobrist::castling_key(new_state.castling_rights);

        new_state.side_to_move = 1 - side;
        new_state.hash ^= zobrist::side_key();

        new_state.occupancies = [Bitboard(0); 3];
        for pp in P..=K {
            new_state.occupancies[WHITE] = new_state.occupancies[WHITE] | new_state.bitboards[pp];
        }
        for pp in p..=k {
            new_state.occupancies[BLACK] = new_state.occupancies[BLACK] | new_state.bitboards[pp];
        }
        new_state.occupancies[BOTH] = new_state.occupancies[WHITE] | new_state.occupancies[BLACK];

        // Update Accumulators
        new_state.accumulator[0].update(&added, &removed, 0);
        new_state.accumulator[1].update(&added, &removed, 1);

        new_state
    }
}
