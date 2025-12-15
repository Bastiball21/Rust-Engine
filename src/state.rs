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
    // Chess960: Store the FILE (0-7) of the initial rook positions for castling.
    // Index: [Color][Side] where Side 0=KingSide, 1=QueenSide
    pub castling_rook_files: [[u8; 2]; 2],
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
            castling_rook_files: [[7, 0], [7, 0]],
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

        if (self.castling_rights & 1) != 0 { // White King-Side
            h ^= zobrist::castling_file_key(WHITE, 0, self.castling_rook_files[WHITE][0]);
        }
        if (self.castling_rights & 2) != 0 { // White Queen-Side
            h ^= zobrist::castling_file_key(WHITE, 1, self.castling_rook_files[WHITE][1]);
        }
        if (self.castling_rights & 4) != 0 { // Black King-Side
            h ^= zobrist::castling_file_key(BLACK, 0, self.castling_rook_files[BLACK][0]);
        }
        if (self.castling_rights & 8) != 0 { // Black Queen-Side
            h ^= zobrist::castling_file_key(BLACK, 1, self.castling_rook_files[BLACK][1]);
        }

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
        let white_king_sq = self.bitboards[K].get_lsb_index() as usize;
        let black_king_sq = self.bitboards[k].get_lsb_index() as usize;

        self.accumulator[WHITE].refresh(bitboards, WHITE, white_king_sq);
        self.accumulator[BLACK].refresh(bitboards, BLACK, black_king_sq);
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

        state.castling_rook_files = [[7, 0], [7, 0]];
        let wk_sq = state.bitboards[K].get_lsb_index() as u8;
        let bk_sq = state.bitboards[k].get_lsb_index() as u8;
        let wk_file = wk_sq % 8;
        let bk_file = bk_sq % 8;

        if parts[2] != "-" {
            for c in parts[2].chars() {
                match c {
                    'K' => { state.castling_rights |= 1; },
                    'Q' => { state.castling_rights |= 2; },
                    'k' => { state.castling_rights |= 4; },
                    'q' => { state.castling_rights |= 8; },
                    'A'..='H' => {
                        let file = c as u8 - b'A';
                        if file > wk_file {
                            state.castling_rights |= 1;
                            state.castling_rook_files[WHITE][0] = file;
                        } else {
                            state.castling_rights |= 2;
                            state.castling_rook_files[WHITE][1] = file;
                        }
                    },
                    'a'..='h' => {
                        let file = c as u8 - b'a';
                        if file > bk_file {
                            state.castling_rights |= 4;
                            state.castling_rook_files[BLACK][0] = file;
                        } else {
                            state.castling_rights |= 8;
                            state.castling_rook_files[BLACK][1] = file;
                        }
                    },
                    _ => {}
                }
            }
        }

        if (state.castling_rights & 1) != 0 {
            let default_f = state.castling_rook_files[WHITE][0];
            let r_sq = (rank_of(wk_sq) * 8) + default_f;
            if !state.bitboards[R].get_bit(r_sq) {
                if let Some(f) = find_outermost_rook(&state, WHITE, true, wk_file) {
                    state.castling_rook_files[WHITE][0] = f;
                } else {
                    state.castling_rights &= !1;
                }
            }
        }
        if (state.castling_rights & 2) != 0 {
            let default_f = state.castling_rook_files[WHITE][1];
            let r_sq = (rank_of(wk_sq) * 8) + default_f;
            if !state.bitboards[R].get_bit(r_sq) {
                if let Some(f) = find_outermost_rook(&state, WHITE, false, wk_file) {
                    state.castling_rook_files[WHITE][1] = f;
                } else {
                    state.castling_rights &= !2;
                }
            }
        }
        if (state.castling_rights & 4) != 0 {
            let default_f = state.castling_rook_files[BLACK][0];
            let r_sq = (rank_of(bk_sq) * 8) + default_f;
            if !state.bitboards[r].get_bit(r_sq) {
                if let Some(f) = find_outermost_rook(&state, BLACK, true, bk_file) {
                    state.castling_rook_files[BLACK][0] = f;
                } else {
                    state.castling_rights &= !4;
                }
            }
        }
        if (state.castling_rights & 8) != 0 {
            let default_f = state.castling_rook_files[BLACK][1];
            let r_sq = (rank_of(bk_sq) * 8) + default_f;
            if !state.bitboards[r].get_bit(r_sq) {
                if let Some(f) = find_outermost_rook(&state, BLACK, false, bk_file) {
                    state.castling_rook_files[BLACK][1] = f;
                } else {
                    state.castling_rights &= !8;
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
        let mut any_rights = false;

        if (self.castling_rights & 1) != 0 {
            any_rights = true;
            let file = self.castling_rook_files[WHITE][0];
            if file == 7 { rights.push('K'); } else { rights.push((b'A' + file) as char); }
        }
        if (self.castling_rights & 2) != 0 {
            any_rights = true;
            let file = self.castling_rook_files[WHITE][1];
            if file == 0 { rights.push('Q'); } else { rights.push((b'A' + file) as char); }
        }
        if (self.castling_rights & 4) != 0 {
            any_rights = true;
            let file = self.castling_rook_files[BLACK][0];
            if file == 7 { rights.push('k'); } else { rights.push((b'a' + file) as char); }
        }
        if (self.castling_rights & 8) != 0 {
            any_rights = true;
            let file = self.castling_rook_files[BLACK][1];
            if file == 0 { rights.push('q'); } else { rights.push((b'a' + file) as char); }
        }

        if !any_rights {
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

        let mut is_castling = false;

        // Detect Castling: King capturing friendly rook
        if piece_type == K || piece_type == k {
            let friendly_rooks = self.bitboards[if side == WHITE { R } else { r }];
            if friendly_rooks.get_bit(mv.target) {
                is_castling = true;
            }
        }

        // Update En Passant: Reset it for next move
        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as u8;
            new_state.hash ^= zobrist::en_passant_key(file);
        }
        new_state.en_passant = 64;

        if is_castling {
            // CASTLING LOGIC (Unified)
            // 1. Remove King from Source
            new_state.hash ^= zobrist::piece_key(piece_type, mv.source as usize);
            new_state.bitboards[piece_type].pop_bit(mv.source);
            removed.push((piece_type, mv.source as usize));

            // 2. Remove Rook from Target (Rook's source)
            let rook_type = if side == WHITE { R } else { r };
            new_state.hash ^= zobrist::piece_key(rook_type, mv.target as usize);
            new_state.bitboards[rook_type].pop_bit(mv.target);
            removed.push((rook_type, mv.target as usize));

            // 3. Determine Destinations
            let rank_base = if side == WHITE { 0 } else { 56 };
            let king_file_dst;
            let rook_file_dst;

            // Which side?
            // We can check if Rook file > King file (Kingside) or < (Queenside)
            // Or compare mv.target vs mv.source
            if mv.target > mv.source { // Kingside (usually, but be careful with indices 0-63 vs file)
                 // Careful: if King on B1, Rook on A1. Target < Source.
                 // Rook on H1, King on B1. Target > Source.
                 // Correct logic: Compare files.
                 if (mv.target % 8) > (mv.source % 8) {
                     // Kingside
                     king_file_dst = 6; // g-file
                     rook_file_dst = 5; // f-file
                 } else {
                     // Queenside
                     king_file_dst = 2; // c-file
                     rook_file_dst = 3; // d-file
                 }
            } else {
                 if (mv.target % 8) > (mv.source % 8) {
                     king_file_dst = 6;
                     rook_file_dst = 5;
                 } else {
                     king_file_dst = 2;
                     rook_file_dst = 3;
                 }
            }

            let k_dst = rank_base + king_file_dst;
            let r_dst = rank_base + rook_file_dst;

            // 4. Place King and Rook at Destinations
            new_state.bitboards[piece_type].set_bit(k_dst);
            new_state.hash ^= zobrist::piece_key(piece_type, k_dst as usize);
            added.push((piece_type, k_dst as usize));

            new_state.bitboards[rook_type].set_bit(r_dst);
            new_state.hash ^= zobrist::piece_key(rook_type, r_dst as usize);
            added.push((rook_type, r_dst as usize));

            // Castling resets halfmove clock
            new_state.halfmove_clock = 0;

        } else {
            // NORMAL MOVE LOGIC

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
                p_idx
            } else {
                piece_type
            };

            if mv.promotion.is_some() {
                 new_state.bitboards[actual_piece].set_bit(mv.target);
                 new_state.hash ^= zobrist::piece_key(actual_piece, mv.target as usize);
            } else {
                 new_state.bitboards[piece_type].set_bit(mv.target);
                 new_state.hash ^= zobrist::piece_key(piece_type, mv.target as usize);
            }

            added.push((actual_piece, mv.target as usize));

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

                    removed.push((enemy_pawn, cap_sq as usize));
                } else {
                    // Normal Capture
                    for pp in enemy_start..=enemy_end {
                        if new_state.bitboards[pp].get_bit(mv.target) {
                            new_state.bitboards[pp].pop_bit(mv.target);
                            new_state.hash ^= zobrist::piece_key(pp, mv.target as usize);
                            removed.push((pp, mv.target as usize));
                            break;
                        }
                    }
                }
            }

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
        }

        // COMMON UPDATES (Castling Rights, Occupancies, Hash, Accumulator)

        // Update Castling Rights
        // Remove rights if they were present
        if (new_state.castling_rights & 1) != 0 { new_state.hash ^= zobrist::castling_file_key(WHITE, 0, new_state.castling_rook_files[WHITE][0]); }
        if (new_state.castling_rights & 2) != 0 { new_state.hash ^= zobrist::castling_file_key(WHITE, 1, new_state.castling_rook_files[WHITE][1]); }
        if (new_state.castling_rights & 4) != 0 { new_state.hash ^= zobrist::castling_file_key(BLACK, 0, new_state.castling_rook_files[BLACK][0]); }
        if (new_state.castling_rights & 8) != 0 { new_state.hash ^= zobrist::castling_file_key(BLACK, 1, new_state.castling_rook_files[BLACK][1]); }
        new_state.hash ^= zobrist::castling_key(new_state.castling_rights);

        // Update rights logic
        // 1. King moves/captured? (Captured handled by source check if we want, but King never captured)
        // 2. Rook moves/captured?

        // If King moves, lose both rights
        if piece_type == K { new_state.castling_rights &= !3; }
        if piece_type == k { new_state.castling_rights &= !12; }

        // If Rook moves or is captured, lose specific right
        // Check Source (Piece moved) and Target (Piece captured) against Rook Files
        // Careful: We need to check against SPECIFIC rook files now, not just 0/7/56/63.

        // Helper to check against file
        let check_rook_rights = |sq: u8, rights: &mut u8, files: [[u8; 2]; 2]| {
             let f = sq % 8;
             let rank_val = sq / 8;

             // White Rooks (Rank 0)
             if rank_val == 0 {
                 if f == files[WHITE][0] { *rights &= !1; } // Kingside
                 if f == files[WHITE][1] { *rights &= !2; } // Queenside
             }
             // Black Rooks (Rank 7)
             if rank_val == 7 {
                 if f == files[BLACK][0] { *rights &= !4; }
                 if f == files[BLACK][1] { *rights &= !8; }
             }
        };

        check_rook_rights(mv.source, &mut new_state.castling_rights, new_state.castling_rook_files);
        check_rook_rights(mv.target, &mut new_state.castling_rights, new_state.castling_rook_files);

        // Re-add rights keys for remaining rights
        new_state.hash ^= zobrist::castling_key(new_state.castling_rights);
        if (new_state.castling_rights & 1) != 0 { new_state.hash ^= zobrist::castling_file_key(WHITE, 0, new_state.castling_rook_files[WHITE][0]); }
        if (new_state.castling_rights & 2) != 0 { new_state.hash ^= zobrist::castling_file_key(WHITE, 1, new_state.castling_rook_files[WHITE][1]); }
        if (new_state.castling_rights & 4) != 0 { new_state.hash ^= zobrist::castling_file_key(BLACK, 0, new_state.castling_rook_files[BLACK][0]); }
        if (new_state.castling_rights & 8) != 0 { new_state.hash ^= zobrist::castling_file_key(BLACK, 1, new_state.castling_rook_files[BLACK][1]); }


        new_state.side_to_move = 1 - side;
        new_state.hash ^= zobrist::side_key();

        // Update Occupancies
        new_state.occupancies = [Bitboard(0); 3];
        for pp in P..=K {
            new_state.occupancies[WHITE] = new_state.occupancies[WHITE] | new_state.bitboards[pp];
        }
        for pp in p..=k {
            new_state.occupancies[BLACK] = new_state.occupancies[BLACK] | new_state.bitboards[pp];
        }
        new_state.occupancies[BOTH] = new_state.occupancies[WHITE] | new_state.occupancies[BLACK];

        // --------------------------------------------------------------------
        // Update Accumulators (Logic for King Buckets)
        // --------------------------------------------------------------------

        let old_k_sq_white = self.bitboards[K].get_lsb_index() as usize;
        let old_k_sq_black = self.bitboards[k].get_lsb_index() as usize;

        let new_k_sq_white = new_state.bitboards[K].get_lsb_index() as usize;
        let new_k_sq_black = new_state.bitboards[k].get_lsb_index() as usize;

        // Check White Accumulator
        let old_bucket_w = crate::nnue::get_king_bucket(WHITE, old_k_sq_white);
        let new_bucket_w = crate::nnue::get_king_bucket(WHITE, new_k_sq_white);

        if old_bucket_w != new_bucket_w {
            new_state.accumulator[WHITE].refresh(&new_state.bitboards, WHITE, new_k_sq_white);
        } else {
            new_state.accumulator[WHITE].update(&added, &removed, WHITE, new_k_sq_white);
        }

        // Check Black Accumulator
        let old_bucket_b = crate::nnue::get_king_bucket(BLACK, old_k_sq_black);
        let new_bucket_b = crate::nnue::get_king_bucket(BLACK, new_k_sq_black);

        if old_bucket_b != new_bucket_b {
            new_state.accumulator[BLACK].refresh(&new_state.bitboards, BLACK, new_k_sq_black);
        } else {
            new_state.accumulator[BLACK].update(&added, &removed, BLACK, new_k_sq_black);
        }

        new_state
    }
}

// Helpers
fn rank_of(sq: u8) -> u8 {
    sq / 8
}

fn find_outermost_rook(state: &GameState, side: usize, is_kingside: bool, k_file: u8) -> Option<u8> {
    let rook_type = if side == WHITE { R } else { r };
    let rank = if side == WHITE { 0 } else { 7 };
    let rooks = state.bitboards[rook_type];

    if is_kingside {
        // Look for rook to the right of King (file > k_file)
        // We want the *outermost* one, so iterate 7 down to k_file + 1
        for f in (k_file + 1..8).rev() {
            let sq = rank * 8 + f;
            if rooks.get_bit(sq) {
                return Some(f);
            }
        }
    } else {
        // Look for rook to the left of King (file < k_file)
        // We want the *outermost* one, so iterate 0 up to k_file - 1
        for f in 0..k_file {
            let sq = rank * 8 + f;
            if rooks.get_bit(sq) {
                return Some(f);
            }
        }
    }
    None
}
