// src/state.rs
#![allow(non_upper_case_globals)]

use crate::bitboard::Bitboard;
use crate::nnue::Accumulator;
use crate::zobrist;

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
pub const NO_PIECE: usize = 12;

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;
pub const BOTH: usize = 2;

// Helper to replace SmallVec with stack array
pub struct UpdateList {
    pub items: [(usize, usize); 4],
    pub len: usize,
}

impl UpdateList {
    pub fn new() -> Self {
        UpdateList {
            items: [(0, 0); 4],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, item: (usize, usize)) {
        if self.len < 4 {
            self.items[self.len] = item;
            self.len += 1;
        }
    }

    pub fn as_slice(&self) -> &[(usize, usize)] {
        &self.items[0..self.len]
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Move(u16);

impl Move {
    pub fn new(source: u8, target: u8, promotion: Option<usize>, is_capture: bool) -> Self {
        let mut d: u16 = 0;
        d |= (source as u16) & 0x3F;
        d |= ((target as u16) & 0x3F) << 6;
        let p_val = match promotion {
            None => 0,
            Some(1) | Some(7) => 1, // N/n
            Some(2) | Some(8) => 2, // B/b
            Some(3) | Some(9) => 3, // R/r
            Some(4) | Some(10) => 4, // Q/q
            _ => 0,
        };
        d |= (p_val << 12) & 0x7000;
        if is_capture {
            d |= 0x8000;
        }
        Move(d)
    }

    #[inline(always)]
    pub fn source(&self) -> u8 {
        (self.0 & 0x3F) as u8
    }

    #[inline(always)]
    pub fn target(&self) -> u8 {
        ((self.0 >> 6) & 0x3F) as u8
    }

    #[inline(always)]
    pub fn promotion(&self) -> Option<usize> {
        let val = (self.0 >> 12) & 0x7;
        match val {
            1 => Some(1),
            2 => Some(2),
            3 => Some(3),
            4 => Some(4),
            _ => None,
        }
    }

    #[inline(always)]
    pub fn is_capture(&self) -> bool {
        (self.0 & 0x8000) != 0
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }
}

impl Default for Move {
    fn default() -> Self {
        Move(0)
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Move {{ source: {}, target: {}, promotion: {:?}, is_capture: {} }}",
            self.source(),
            self.target(),
            self.promotion(),
            self.is_capture()
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UnmakeInfo {
    pub captured: u8, // piece type or NO_PIECE
    pub en_passant: u8,
    pub castling_rights: u8,
    pub halfmove_clock: u8,
    pub old_hash: u64,
    pub is_castling: bool,
    // Accumulator backup for King moves (rare)
    pub acc_backup: Option<[Accumulator; 2]>,
}

#[derive(Debug, Clone, Copy)]
pub struct GameState {
    pub bitboards: [Bitboard; 12],
    pub occupancies: [Bitboard; 3],
    pub board: [u8; 64], // Mailbox representation (NO_PIECE if empty)
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
            board: [NO_PIECE as u8; 64],
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

        if (self.castling_rights & 1) != 0 {
            // White King-Side
            h ^= zobrist::castling_file_key(WHITE, 0, self.castling_rook_files[WHITE][0]);
        }
        if (self.castling_rights & 2) != 0 {
            // White Queen-Side
            h ^= zobrist::castling_file_key(WHITE, 1, self.castling_rook_files[WHITE][1]);
        }
        if (self.castling_rights & 4) != 0 {
            // Black King-Side
            h ^= zobrist::castling_file_key(BLACK, 0, self.castling_rook_files[BLACK][0]);
        }
        if (self.castling_rights & 8) != 0 {
            // Black Queen-Side
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
                    _ => NO_PIECE,
                };
                if piece != NO_PIECE {
                    state.bitboards[piece].set_bit(square as u8);
                    state.board[square as usize] = piece as u8;
                }
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
                    'K' => {
                        state.castling_rights |= 1;
                    }
                    'Q' => {
                        state.castling_rights |= 2;
                    }
                    'k' => {
                        state.castling_rights |= 4;
                    }
                    'q' => {
                        state.castling_rights |= 8;
                    }
                    'A'..='H' => {
                        let file = c as u8 - b'A';
                        if file > wk_file {
                            state.castling_rights |= 1;
                            state.castling_rook_files[WHITE][0] = file;
                        } else {
                            state.castling_rights |= 2;
                            state.castling_rook_files[WHITE][1] = file;
                        }
                    }
                    'a'..='h' => {
                        let file = c as u8 - b'a';
                        if file > bk_file {
                            state.castling_rights |= 4;
                            state.castling_rook_files[BLACK][0] = file;
                        } else {
                            state.castling_rights |= 8;
                            state.castling_rook_files[BLACK][1] = file;
                        }
                    }
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
            if file == 7 {
                rights.push('K');
            } else {
                rights.push((b'A' + file) as char);
            }
        }
        if (self.castling_rights & 2) != 0 {
            any_rights = true;
            let file = self.castling_rook_files[WHITE][1];
            if file == 0 {
                rights.push('Q');
            } else {
                rights.push((b'A' + file) as char);
            }
        }
        if (self.castling_rights & 4) != 0 {
            any_rights = true;
            let file = self.castling_rook_files[BLACK][0];
            if file == 7 {
                rights.push('k');
            } else {
                rights.push((b'a' + file) as char);
            }
        }
        if (self.castling_rights & 8) != 0 {
            any_rights = true;
            let file = self.castling_rook_files[BLACK][1];
            if file == 0 {
                rights.push('q');
            } else {
                rights.push((b'a' + file) as char);
            }
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

    pub fn make_null_move_inplace(&mut self) -> UnmakeInfo {
        let old_hash = self.hash;
        let old_en_passant = self.en_passant;
        let old_castling_rights = self.castling_rights;
        let old_halfmove_clock = self.halfmove_clock;

        self.side_to_move = 1 - self.side_to_move;
        self.hash ^= zobrist::side_key();

        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as u8;
            self.hash ^= zobrist::en_passant_key(file);
            self.en_passant = 64;
        }

        self.halfmove_clock = 0;

        // Note: Fullmove number usually increments after Black moves.
        // We handle this in make_move. We should do it here too?
        // Logic: if Side was BLACK, we moved to WHITE, so increment.
        // But self.side_to_move is now flipped.
        // So if self.side_to_move == WHITE, it means BLACK just moved.
        if self.side_to_move == WHITE {
            self.fullmove_number += 1;
        }

        UnmakeInfo {
            captured: NO_PIECE as u8,
            en_passant: old_en_passant,
            castling_rights: old_castling_rights,
            halfmove_clock: old_halfmove_clock,
            old_hash,
            is_castling: false,
            acc_backup: None,
        }
    }

    pub fn unmake_null_move(&mut self, info: UnmakeInfo) {
        if self.side_to_move == WHITE {
             self.fullmove_number -= 1;
        }
        self.side_to_move = 1 - self.side_to_move;

        self.en_passant = info.en_passant;
        self.halfmove_clock = info.halfmove_clock;
        self.hash = info.old_hash;
        // Castling rights don't change in null move
    }

    // Legacy support for Copy-Make (can be removed later or kept for tests)
    pub fn make_move(&self, mv: Move) -> GameState {
        let mut new_state = *self;
        new_state.make_move_inplace(mv);
        new_state
    }

    pub fn make_move_inplace(&mut self, mv: Move) -> UnmakeInfo {
        let side = self.side_to_move;
        let mut captured_piece;
        let old_hash = self.hash;
        let old_en_passant = self.en_passant;
        let old_castling_rights = self.castling_rights;
        let old_halfmove_clock = self.halfmove_clock;

        let mut acc_backup = None;

        // NNUE Incremental Update Tracking (Using Stack Array)
        let mut added = UpdateList::new();
        let mut removed = UpdateList::new();

        if side == BLACK {
            self.fullmove_number += 1;
        }

        self.halfmove_clock += 1;

        // MAILBOX OPTIMIZATION: Get piece from board array instead of loop
        let source = mv.source();
        let target = mv.target();
        let promotion = mv.promotion();
        let is_capture = mv.is_capture();

        let mut piece_type = self.board[source as usize] as usize;

        if piece_type == NO_PIECE {
            // Should not happen in legal search
             let start_range = if side == WHITE { P } else { p };
             let end_range = if side == WHITE { K } else { k };
             for pp in start_range..=end_range {
                 if self.bitboards[pp].get_bit(source) {
                     piece_type = pp;
                     break;
                 }
             }
        }

        if piece_type == NO_PIECE {
            panic!(
                "CRITICAL: Attempted to move from empty square! Move: {:?}, FEN: {}",
                mv,
                self.to_fen()
            );
        }

        let mut is_castling = false;

        // Detect Castling
        if piece_type == K || piece_type == k {
            let target_piece = self.board[target as usize] as usize;
            let rook_type = if side == WHITE { R } else { r };
            if target_piece == rook_type {
                 is_castling = true;
            }
        }

        // Update En Passant: Reset it for next move
        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as u8;
            self.hash ^= zobrist::en_passant_key(file);
        }
        self.en_passant = 64;

        if is_castling {
            captured_piece = NO_PIECE as u8; // Castling "captures" own rook but we handle restore manually

            // 1. Remove King from Source
            self.hash ^= zobrist::piece_key(piece_type, source as usize);
            self.bitboards[piece_type].pop_bit(source);
            self.board[source as usize] = NO_PIECE as u8;
            removed.push((piece_type, source as usize));

            // Occupancy Update
            self.occupancies[side].pop_bit(source);
            self.occupancies[BOTH].pop_bit(source);

            // 2. Remove Rook from Target (Rook's source)
            let rook_type = if side == WHITE { R } else { r };
            self.hash ^= zobrist::piece_key(rook_type, target as usize);
            self.bitboards[rook_type].pop_bit(target);
            self.board[target as usize] = NO_PIECE as u8;
            removed.push((rook_type, target as usize));

            self.occupancies[side].pop_bit(target);
            self.occupancies[BOTH].pop_bit(target);

            // 3. Determine Destinations
            let rank_base = if side == WHITE { 0 } else { 56 };
            let king_file_dst;
            let rook_file_dst;

            if target > source {
                // Kingside
                if (target % 8) > (source % 8) {
                    king_file_dst = 6; // g-file
                    rook_file_dst = 5; // f-file
                } else {
                    king_file_dst = 2; // c-file
                    rook_file_dst = 3; // d-file
                }
            } else {
                if (target % 8) > (source % 8) {
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
            self.bitboards[piece_type].set_bit(k_dst);
            self.board[k_dst as usize] = piece_type as u8;
            self.hash ^= zobrist::piece_key(piece_type, k_dst as usize);
            added.push((piece_type, k_dst as usize));

            self.occupancies[side].set_bit(k_dst);
            self.occupancies[BOTH].set_bit(k_dst);

            self.bitboards[rook_type].set_bit(r_dst);
            self.board[r_dst as usize] = rook_type as u8;
            self.hash ^= zobrist::piece_key(rook_type, r_dst as usize);
            added.push((rook_type, r_dst as usize));

            self.occupancies[side].set_bit(r_dst);
            self.occupancies[BOTH].set_bit(r_dst);

            self.halfmove_clock = 0;
        } else {
            // NORMAL MOVE LOGIC

            // 1. Remove moving piece from source
            removed.push((piece_type, source as usize));

            if piece_type == P || piece_type == p || is_capture {
                self.halfmove_clock = 0;
            }

            self.hash ^= zobrist::piece_key(piece_type, source as usize);
            self.bitboards[piece_type].pop_bit(source);
            self.board[source as usize] = NO_PIECE as u8;

            self.occupancies[side].pop_bit(source);
            self.occupancies[BOTH].pop_bit(source);

            // 2. Add moving piece (or promo) to target
            let actual_piece = if let Some(promo) = promotion {
                let p_idx = if side == WHITE { promo } else { promo + 6 };
                p_idx
            } else {
                piece_type
            };

            if promotion.is_some() {
                self.bitboards[actual_piece].set_bit(target);
                self.hash ^= zobrist::piece_key(actual_piece, target as usize);
            } else {
                self.bitboards[piece_type].set_bit(target);
                self.hash ^= zobrist::piece_key(piece_type, target as usize);
            }

            // Capture Logic
            if is_capture {
                let enemy_side = 1 - side;
                if (piece_type == P || piece_type == p) && target == old_en_passant {
                    // En Passant
                    let cap_sq = if side == WHITE {
                        target - 8
                    } else {
                        target + 8
                    };
                    let enemy_pawn = if side == WHITE { p } else { P };
                    captured_piece = enemy_pawn as u8;

                    self.bitboards[enemy_pawn].pop_bit(cap_sq);
                    self.board[cap_sq as usize] = NO_PIECE as u8;
                    self.hash ^= zobrist::piece_key(enemy_pawn, cap_sq as usize);
                    removed.push((enemy_pawn, cap_sq as usize));

                    self.occupancies[enemy_side].pop_bit(cap_sq);
                    self.occupancies[BOTH].pop_bit(cap_sq);

                    // Target was empty for EP
                    self.board[target as usize] = actual_piece as u8;
                } else {
                    // Normal Capture
                    captured_piece = self.board[target as usize];

                    if captured_piece == NO_PIECE as u8 {
                        // Fallback to bitboards if board[] is desynchronized (empty)
                        let enemy_start = if side == WHITE { p } else { P };
                        let enemy_end = if side == WHITE { k } else { K };
                        for pp in enemy_start..=enemy_end {
                            if self.bitboards[pp].get_bit(target) {
                                captured_piece = pp as u8;
                                eprintln!(
                                    "WARNING: Board desync detected at {}. Recovered via bitboards. Move: {:?}",
                                    target, mv
                                );
                                break;
                            }
                        }
                    }

                    if captured_piece == NO_PIECE as u8 {
                        panic!(
                            "CRITICAL: Capture move on empty square! Move: {:?}, FEN: {}",
                            mv,
                            self.to_fen()
                        );
                    }

                    if captured_piece != NO_PIECE as u8 {
                         let cap_p = captured_piece as usize;
                         self.bitboards[cap_p].pop_bit(target);
                         self.hash ^= zobrist::piece_key(cap_p, target as usize);
                         removed.push((cap_p, target as usize));
                         // Note: board update for captured piece is implicit (overwritten below)
                    }
                    self.occupancies[enemy_side].pop_bit(target);
                    self.board[target as usize] = actual_piece as u8;
                }
            } else {
                captured_piece = NO_PIECE as u8;
                self.board[target as usize] = actual_piece as u8;
            }

            added.push((actual_piece, target as usize));

            // Occupancy Add
            self.occupancies[side].set_bit(target);
            self.occupancies[BOTH].set_bit(target);

            // Set new EP
             if piece_type == P || piece_type == p {
                let diff = (target as i8 - source as i8).abs();
                if diff == 16 {
                    let ep_sq = if side == WHITE {
                        target - 8
                    } else {
                        target + 8
                    };
                    self.en_passant = ep_sq;
                    self.hash ^= zobrist::en_passant_key((ep_sq % 8) as u8);
                }
            }
        }

        // CASTLING RIGHTS UPDATE
        // Remove rights if they were present
        if (self.castling_rights & 1) != 0 {
            self.hash ^=
                zobrist::castling_file_key(WHITE, 0, self.castling_rook_files[WHITE][0]);
        }
        if (self.castling_rights & 2) != 0 {
            self.hash ^=
                zobrist::castling_file_key(WHITE, 1, self.castling_rook_files[WHITE][1]);
        }
        if (self.castling_rights & 4) != 0 {
            self.hash ^=
                zobrist::castling_file_key(BLACK, 0, self.castling_rook_files[BLACK][0]);
        }
        if (self.castling_rights & 8) != 0 {
            self.hash ^=
                zobrist::castling_file_key(BLACK, 1, self.castling_rook_files[BLACK][1]);
        }
        self.hash ^= zobrist::castling_key(self.castling_rights);

        // Logic
        if piece_type == K {
            self.castling_rights &= !3;
        }
        if piece_type == k {
            self.castling_rights &= !12;
        }

        // Remove 'mut' from 'check_rook_rights' callback to avoid unused warning
        let check_rook_rights = |sq: u8, rights: &mut u8, files: [[u8; 2]; 2]| {
            let f = sq % 8;
            let rank_val = sq / 8;
            if rank_val == 0 {
                if f == files[WHITE][0] { *rights &= !1; }
                if f == files[WHITE][1] { *rights &= !2; }
            }
            if rank_val == 7 {
                if f == files[BLACK][0] { *rights &= !4; }
                if f == files[BLACK][1] { *rights &= !8; }
            }
        };

        check_rook_rights(source, &mut self.castling_rights, self.castling_rook_files);
        check_rook_rights(target, &mut self.castling_rights, self.castling_rook_files);

        // Re-add rights keys
        self.hash ^= zobrist::castling_key(self.castling_rights);
        if (self.castling_rights & 1) != 0 {
            self.hash ^=
                zobrist::castling_file_key(WHITE, 0, self.castling_rook_files[WHITE][0]);
        }
        if (self.castling_rights & 2) != 0 {
            self.hash ^=
                zobrist::castling_file_key(WHITE, 1, self.castling_rook_files[WHITE][1]);
        }
        if (self.castling_rights & 4) != 0 {
            self.hash ^=
                zobrist::castling_file_key(BLACK, 0, self.castling_rook_files[BLACK][0]);
        }
        if (self.castling_rights & 8) != 0 {
            self.hash ^=
                zobrist::castling_file_key(BLACK, 1, self.castling_rook_files[BLACK][1]);
        }

        self.side_to_move = 1 - side;
        self.hash ^= zobrist::side_key();

        // --------------------------------------------------------------------
        // Update Accumulators (Logic for King Buckets)
        // --------------------------------------------------------------------

        // We need to check if King bucket changed.
        // If King moved, potential bucket change.
        // We check if either King's bucket changed.

        let new_k_sq_white = self.bitboards[K].get_lsb_index() as usize;
        let new_k_sq_black = self.bitboards[k].get_lsb_index() as usize;

        // We know old positions? We can infer or check if piece_type == K/k.
        let (old_k_sq_w, old_k_sq_b) = if piece_type == K {
             (source as usize, new_k_sq_black)
        } else if piece_type == k {
             (new_k_sq_white, source as usize)
        } else {
             (new_k_sq_white, new_k_sq_black)
        };

        let old_bucket_w = crate::nnue::get_king_bucket(WHITE, old_k_sq_w);
        let new_bucket_w = crate::nnue::get_king_bucket(WHITE, new_k_sq_white);

        let old_bucket_b = crate::nnue::get_king_bucket(BLACK, old_k_sq_b);
        let new_bucket_b = crate::nnue::get_king_bucket(BLACK, new_k_sq_black);

        let mut needs_backup = false;
        if old_bucket_w != new_bucket_w || old_bucket_b != new_bucket_b {
             needs_backup = true;
        }

        if needs_backup {
             acc_backup = Some(self.accumulator); // Clone full accumulators (2KB)
             // Refresh
             if old_bucket_w != new_bucket_w {
                  self.accumulator[WHITE].refresh(&self.bitboards, WHITE, new_k_sq_white);
             } else {
                  self.accumulator[WHITE].update(added.as_slice(), removed.as_slice(), WHITE, new_k_sq_white);
             }
             if old_bucket_b != new_bucket_b {
                  self.accumulator[BLACK].refresh(&self.bitboards, BLACK, new_k_sq_black);
             } else {
                  self.accumulator[BLACK].update(added.as_slice(), removed.as_slice(), BLACK, new_k_sq_black);
             }
        } else {
             self.accumulator[WHITE].update(added.as_slice(), removed.as_slice(), WHITE, new_k_sq_white);
             self.accumulator[BLACK].update(added.as_slice(), removed.as_slice(), BLACK, new_k_sq_black);
        }

        UnmakeInfo {
            captured: captured_piece,
            en_passant: old_en_passant,
            castling_rights: old_castling_rights,
            halfmove_clock: old_halfmove_clock,
            old_hash,
            is_castling,
            acc_backup,
        }
    }

    pub fn unmake_move(&mut self, mv: Move, info: UnmakeInfo) {
        // Reverse Move Logic
        // 1. Restore Side
        self.side_to_move = 1 - self.side_to_move;
        let side = self.side_to_move; // Now back to original side
        if side == BLACK {
            self.fullmove_number -= 1;
        }

        // 2. Restore Scalar Fields
        self.en_passant = info.en_passant;
        self.castling_rights = info.castling_rights;
        self.halfmove_clock = info.halfmove_clock;
        self.hash = info.old_hash;

        let mut added = UpdateList::new();
        let mut removed = UpdateList::new();

        let source = mv.source();
        let target = mv.target();
        let promotion = mv.promotion();
        let is_capture = mv.is_capture();

        if info.is_castling {
             // Castling Unmake
             // Make Logic: K from src->k_dst, R from tgt->r_dst.
             // We need to move K from k_dst->src, R from r_dst->tgt.

             let rank_base = if side == WHITE { 0 } else { 56 };
             let king_file_dst;
             let rook_file_dst;

             if target > source {
                 if (target % 8) > (source % 8) {
                     king_file_dst = 6;
                     rook_file_dst = 5;
                 } else {
                     king_file_dst = 2;
                     rook_file_dst = 3;
                 }
             } else {
                 if (target % 8) > (source % 8) {
                     king_file_dst = 6;
                     rook_file_dst = 5;
                 } else {
                     king_file_dst = 2;
                     rook_file_dst = 3;
                 }
             }

             let k_dst = rank_base + king_file_dst;
             let r_dst = rank_base + rook_file_dst;

             let k_piece = if side == WHITE { K } else { k };
             let r_piece = if side == WHITE { R } else { r };

             // Remove from destinations
             self.bitboards[k_piece].pop_bit(k_dst);
             self.board[k_dst as usize] = NO_PIECE as u8;
             self.occupancies[side].pop_bit(k_dst);
             self.occupancies[BOTH].pop_bit(k_dst);
             removed.push((k_piece, k_dst as usize));

             self.bitboards[r_piece].pop_bit(r_dst);
             self.board[r_dst as usize] = NO_PIECE as u8;
             self.occupancies[side].pop_bit(r_dst);
             self.occupancies[BOTH].pop_bit(r_dst);
             removed.push((r_piece, r_dst as usize));

             // Add back to sources
             self.bitboards[k_piece].set_bit(source);
             self.board[source as usize] = k_piece as u8;
             self.occupancies[side].set_bit(source);
             self.occupancies[BOTH].set_bit(source);
             added.push((k_piece, source as usize));

             self.bitboards[r_piece].set_bit(target);
             self.board[target as usize] = r_piece as u8;
             self.occupancies[side].set_bit(target);
             self.occupancies[BOTH].set_bit(target);
             added.push((r_piece, target as usize));

        } else {
             // Normal Unmake

             // What is at target?
             // If promotion, it's the promoted piece.
             // If normal, it's the mover.
             let moved_piece = self.board[target as usize] as usize; // This is what is currently at target

             // Remove from target
             self.bitboards[moved_piece].pop_bit(target);
             self.board[target as usize] = NO_PIECE as u8;
             self.occupancies[side].pop_bit(target);
             self.occupancies[BOTH].pop_bit(target);
             removed.push((moved_piece, target as usize));

             // Place back at source
             // If promotion, we placed PromotedPiece. We remove it (done above).
             // We need to place Pawn at source.
             let original_piece = if promotion.is_some() {
                  if side == WHITE { P } else { p }
             } else {
                  moved_piece
             };

             self.bitboards[original_piece].set_bit(source);
             self.board[source as usize] = original_piece as u8;
             self.occupancies[side].set_bit(source);
             self.occupancies[BOTH].set_bit(source);
             added.push((original_piece, source as usize));

             // Restore captured piece
             if is_capture {
                  let captured = info.captured as usize;
                  let enemy_side = 1 - side;
                  let cap_sq = if (original_piece == P || original_piece == p) && target == info.en_passant {
                       // EP Capture
                       if side == WHITE { target - 8 } else { target + 8 }
                  } else {
                       target
                  };

                  self.bitboards[captured].set_bit(cap_sq);
                  self.board[cap_sq as usize] = captured as u8;
                  self.occupancies[enemy_side].set_bit(cap_sq);
                  self.occupancies[BOTH].set_bit(cap_sq);
                  added.push((captured, cap_sq as usize));
             }
        }

        // NNUE Restore
        if let Some(backup) = info.acc_backup {
             self.accumulator = backup;
        } else {
             // Inverse update
             let k_sq_white = self.bitboards[K].get_lsb_index() as usize;
             let k_sq_black = self.bitboards[k].get_lsb_index() as usize;

             self.accumulator[WHITE].update(added.as_slice(), removed.as_slice(), WHITE, k_sq_white);
             self.accumulator[BLACK].update(added.as_slice(), removed.as_slice(), BLACK, k_sq_black);
        }
    }
}

// Helpers
fn rank_of(sq: u8) -> u8 {
    sq / 8
}

fn find_outermost_rook(
    state: &GameState,
    side: usize,
    is_kingside: bool,
    k_file: u8,
) -> Option<u8> {
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
