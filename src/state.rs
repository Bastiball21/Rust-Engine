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
pub struct Move(pub u16);

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
    pub pawn_key: u64, // Pawn structure hash (Zobrist of pawns only)
    pub halfmove_clock: u8,
    pub fullmove_number: u16,
    // REMOVED: pub accumulator: [Accumulator; 2],
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
            pawn_key: 0,
            halfmove_clock: 0,
            fullmove_number: 1,
            // accumulator removed
            castling_rook_files: [[7, 0], [7, 0]],
        }
    }

    pub fn compute_hash(&mut self) {
        let mut h: u64 = 0;
        let mut ph: u64 = 0;
        for piece in 0..12 {
            let mut bb = self.bitboards[piece];
            while bb.0 != 0 {
                let sq = bb.get_lsb_index();
                bb.pop_bit(sq as u8);
                let key_val = zobrist::piece_key(piece, sq as usize);
                h ^= key_val;
                if piece == P || piece == p {
                    ph ^= key_val;
                }
            }
        }
        self.pawn_key = ph;
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

    pub fn refresh_accumulator(&self, acc: &mut [Accumulator; 2]) {
        let bitboards = &self.bitboards;
        let white_king_sq = self.bitboards[K].get_lsb_index() as usize;
        let black_king_sq = self.bitboards[k].get_lsb_index() as usize;

        acc[WHITE].refresh(bitboards, WHITE, white_king_sq);
        acc[BLACK].refresh(bitboards, BLACK, black_king_sq);
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
        let wk_sq = if state.bitboards[K].0 != 0 { state.bitboards[K].get_lsb_index() as u8 } else { 64 };
        let bk_sq = if state.bitboards[k].0 != 0 { state.bitboards[k].get_lsb_index() as u8 } else { 64 };
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

        if wk_sq != 64 {
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
        } else {
            state.castling_rights &= !3;
        }

        if bk_sq != 64 {
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
        } else {
            state.castling_rights &= !12;
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
        // Accumulator refresh is now explicit caller responsibility
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
        // Pawn key doesn't change on null move

        self.side_to_move = 1 - self.side_to_move;
        self.hash ^= zobrist::side_key();

        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as u8;
            self.hash ^= zobrist::en_passant_key(file);
            self.en_passant = 64;
        }

        self.halfmove_clock = 0;

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
    }

    // Legacy Support: make_move needs to operate on CLONED accumulators if we want full state.
    // BUT we removed accumulators from GameState.
    // So this function returns GameState WITHOUT accumulators updated.
    // This is fine for Perft. For Search, we use inplace.
    pub fn make_move(&self, mv: Move) -> GameState {
        let mut new_state = *self;
        new_state.make_move_inplace(mv, &mut None); // Pass None for accumulator
        new_state
    }

    // New Signature: Accepts Option<&mut [Accumulator; 2]>
    pub fn make_move_inplace(&mut self, mv: Move, accumulators: &mut Option<&mut [Accumulator; 2]>) -> UnmakeInfo {
        #[cfg(debug_assertions)]
        {
            if let Err(e) = self.validate_consistency() {
                self.dump_diagnostics(mv, &format!("Pre-Make Consistency Failure: {}", e));
                panic!("State corrupted before move {:?}: {}", mv, e);
            }

            debug_assert!(
                self.board[mv.source() as usize] != NO_PIECE as u8,
                "No piece on source: {:?}, FEN: {}",
                mv,
                self.to_fen()
            );

            // Capture logic check
            if mv.is_capture() {
                // If it's a capture, the target must be occupied OR it must be en-passant
                // We check if target is empty first
                if self.board[mv.target() as usize] == NO_PIECE as u8 {
                     // Check for EP
                     // Must be Pawn + Target == En Passant Square
                     let piece = self.board[mv.source() as usize] as usize;
                     let is_pawn = piece == P || piece == p;
                     let is_ep_sq = mv.target() == self.en_passant;
                     debug_assert!(
                        is_pawn && is_ep_sq,
                        "Capture on empty square (not EP): {:?}, FEN: {}",
                        mv,
                        self.to_fen()
                     );
                }
            }
        }

        let side = self.side_to_move;
        let mut captured_piece;
        let old_hash = self.hash;
        let old_en_passant = self.en_passant;
        let old_castling_rights = self.castling_rights;
        let old_halfmove_clock = self.halfmove_clock;
        // No need to backup pawn_key, we can re-derive it on unmake if strictly necessary,
        // but since we modify it incrementally, we just reverse ops in unmake.

        let mut acc_backup = None;

        // NNUE Incremental Update Tracking (Using Stack Array)
        let mut added = UpdateList::new();
        let mut removed = UpdateList::new();

        if side == BLACK {
            self.fullmove_number += 1;
        }

        self.halfmove_clock += 1;

        let source = mv.source();
        let target = mv.target();
        let promotion = mv.promotion();
        let is_capture = mv.is_capture();

        let mut piece_type = self.board[source as usize] as usize;

        // Strict Check: Source must be occupied in bitboards too
        if piece_type != NO_PIECE {
            if !self.bitboards[piece_type].get_bit(source) {
                 panic!("CRITICAL: Source square {} has piece {} in mailbox but not in bitboards! Move: {:?}", source, piece_type, mv);
            }
        }

        if piece_type == NO_PIECE {
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

        // Strict Check: Quiet moves must not target occupied squares (in Bitboards)
        if !is_capture {
            let target_occ = self.occupancies[BOTH].get_bit(target);
            // Allow Castling (King -> Rook)
            let mut is_castling_attempt = false;
            if piece_type == K || piece_type == k {
                let rook_type = if side == WHITE { R } else { r };
                // If target has Rook bit, it's potentially castling.
                // Note: We check `target_piece` later for real castling logic,
                // but here we just want to avoid panic on valid castling.
                if self.bitboards[rook_type].get_bit(target) {
                    is_castling_attempt = true;
                }
            }

            if target_occ && !is_castling_attempt {
                 self.dump_diagnostics(mv, "Quiet Move to Occupied Square");
                 panic!("CRITICAL: Quiet move to occupied square! Target: {}, Move: {:?}", target, mv);
            }
        }

        let mut is_castling = false;

        if piece_type == K || piece_type == k {
            let target_piece = self.board[target as usize] as usize;
            let rook_type = if side == WHITE { R } else { r };
            if target_piece == rook_type {
                 is_castling = true;
            }
        }

        if self.en_passant != 64 {
            let file = (self.en_passant % 8) as u8;
            self.hash ^= zobrist::en_passant_key(file);
        }
        self.en_passant = 64;

        if is_castling {
            captured_piece = NO_PIECE as u8;
            // King
            self.hash ^= zobrist::piece_key(piece_type, source as usize);
            self.bitboards[piece_type].pop_bit(source);
            self.board[source as usize] = NO_PIECE as u8;
            removed.push((piece_type, source as usize));

            self.occupancies[side].pop_bit(source);
            self.occupancies[BOTH].pop_bit(source);

            // Rook
            let rook_type = if side == WHITE { R } else { r };
            self.hash ^= zobrist::piece_key(rook_type, target as usize);
            self.bitboards[rook_type].pop_bit(target);
            self.board[target as usize] = NO_PIECE as u8;
            removed.push((rook_type, target as usize));

            self.occupancies[side].pop_bit(target);
            self.occupancies[BOTH].pop_bit(target);

            let rank_base = if side == WHITE { 0 } else { 56 };
            let king_file_dst;
            let rook_file_dst;

            // Chess960 Castling Side Detection
            let target_file = target % 8;
            if target_file == self.castling_rook_files[side][0] {
                // King-Side Castling
                king_file_dst = 6; // g-file
                rook_file_dst = 5; // f-file
            } else if target_file == self.castling_rook_files[side][1] {
                // Queen-Side Castling
                king_file_dst = 2; // c-file
                rook_file_dst = 3; // d-file
            } else {
                panic!(
                    "CRITICAL: Castling target {} matches neither King-side ({}) nor Queen-side ({}) rook file. FEN: {}",
                    target_file,
                    self.castling_rook_files[side][0],
                    self.castling_rook_files[side][1],
                    self.to_fen()
                );
            }

            let k_dst = rank_base + king_file_dst;
            let r_dst = rank_base + rook_file_dst;

            // Ensure destinations are clean (Ghost Busting)
            self.clear_square(k_dst);
            self.clear_square(r_dst);

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
            removed.push((piece_type, source as usize));

            if piece_type == P || piece_type == p || is_capture {
                self.halfmove_clock = 0;
            }

            self.hash ^= zobrist::piece_key(piece_type, source as usize);
            if piece_type == P || piece_type == p {
                self.pawn_key ^= zobrist::piece_key(piece_type, source as usize);
            }

            self.bitboards[piece_type].pop_bit(source);
            self.board[source as usize] = NO_PIECE as u8;

            self.occupancies[side].pop_bit(source);
            self.occupancies[BOTH].pop_bit(source);

            let actual_piece = if let Some(promo) = promotion {
                let p_idx = if side == WHITE { promo } else { promo + 6 };
                p_idx
            } else {
                piece_type
            };

            if promotion.is_some() {
                self.bitboards[actual_piece].set_bit(target);
                self.hash ^= zobrist::piece_key(actual_piece, target as usize);
                // Promotion: Old pawn removed (above), new piece added. New piece not pawn.
            } else {
                // Ensure target is clean before setting (Ghost Busting for Quiet Moves)
                if !is_capture {
                     self.clear_square(target);
                }
                self.bitboards[piece_type].set_bit(target);
                self.hash ^= zobrist::piece_key(piece_type, target as usize);
                if piece_type == P || piece_type == p {
                    self.pawn_key ^= zobrist::piece_key(piece_type, target as usize);
                }
            }

            if is_capture {
                let enemy_side = 1 - side;
                if (piece_type == P || piece_type == p) && target == old_en_passant {
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
                    self.pawn_key ^= zobrist::piece_key(enemy_pawn, cap_sq as usize);

                    removed.push((enemy_pawn, cap_sq as usize));

                    self.occupancies[enemy_side].pop_bit(cap_sq);
                    self.occupancies[BOTH].pop_bit(cap_sq);

                    self.board[target as usize] = actual_piece as u8;
                } else {
                    captured_piece = self.board[target as usize];

                    if captured_piece == NO_PIECE as u8 {
                        // Scan all pieces, not just enemies, to detect friendly fire
                        for pp in 0..12 {
                            // Skip the piece we just moved to the target square!
                            if pp == actual_piece { continue; }

                            if self.bitboards[pp].get_bit(target) {
                                let is_friendly = if side == WHITE { pp <= K } else { pp >= p };
                                if is_friendly {
                                    self.dump_diagnostics(mv, "Friendly Fire Detected");
                                    panic!(
                                        "CRITICAL: Friendly fire detected (Self-Capture) - Possible TT Corruption. Target: {}, Piece: {}, Move: {:?}, FEN: {}",
                                        target, pp, mv, self.to_fen()
                                    );
                                }

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
                        self.dump_diagnostics(mv, "Capture on empty square");
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
                         if cap_p == P || cap_p == p {
                             self.pawn_key ^= zobrist::piece_key(cap_p, target as usize);
                         }
                         removed.push((cap_p, target as usize));
                    }
                    self.occupancies[enemy_side].pop_bit(target);
                    self.board[target as usize] = actual_piece as u8;
                }
            } else {
                captured_piece = NO_PIECE as u8;
                self.board[target as usize] = actual_piece as u8;
            }

            added.push((actual_piece, target as usize));

            self.occupancies[side].set_bit(target);
            self.occupancies[BOTH].set_bit(target);

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

        if piece_type == K {
            self.castling_rights &= !3;
        }
        if piece_type == k {
            self.castling_rights &= !12;
        }

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

        if let Some(acc) = accumulators {
            let new_k_sq_white = self.bitboards[K].get_lsb_index() as usize;
            let new_k_sq_black = self.bitboards[k].get_lsb_index() as usize;

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
                 acc_backup = Some(**acc); // Clone current
                 if old_bucket_w != new_bucket_w {
                      acc[WHITE].refresh(&self.bitboards, WHITE, new_k_sq_white);
                 } else {
                      acc[WHITE].update(added.as_slice(), removed.as_slice(), WHITE, new_k_sq_white);
                 }
                 if old_bucket_b != new_bucket_b {
                      acc[BLACK].refresh(&self.bitboards, BLACK, new_k_sq_black);
                 } else {
                      acc[BLACK].update(added.as_slice(), removed.as_slice(), BLACK, new_k_sq_black);
                 }
            } else {
                 acc[WHITE].update(added.as_slice(), removed.as_slice(), WHITE, new_k_sq_white);
                 acc[BLACK].update(added.as_slice(), removed.as_slice(), BLACK, new_k_sq_black);
            }
        }

        #[cfg(debug_assertions)]
        {
            if let Err(e) = self.validate_consistency() {
                self.dump_diagnostics(mv, &format!("Post-Make Consistency Failure: {}", e));
                panic!("State corrupted after move {:?}: {}", mv, e);
            }
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

    pub fn unmake_move(&mut self, mv: Move, info: UnmakeInfo, accumulators: &mut Option<&mut [Accumulator; 2]>) {
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
        let old_pawn_key = self.pawn_key; // Save current (new) pawn key to reverse changes
        self.hash = info.old_hash;

        let mut added = UpdateList::new();
        let mut removed = UpdateList::new();

        let source = mv.source();
        let target = mv.target();
        let promotion = mv.promotion();
        let is_capture = mv.is_capture();

        if info.is_castling {
             let rank_base = if side == WHITE { 0 } else { 56 };
             let king_file_dst;
             let rook_file_dst;

             // Chess960 Castling Side Detection
             let target_file = target % 8;
             if target_file == self.castling_rook_files[side][0] {
                 // King-Side Castling
                 king_file_dst = 6;
                 rook_file_dst = 5;
             } else if target_file == self.castling_rook_files[side][1] {
                 // Queen-Side Castling
                 king_file_dst = 2;
                 rook_file_dst = 3;
             } else {
                 panic!(
                     "CRITICAL: Unmake Castling target {} matches neither King-side ({}) nor Queen-side ({}) rook file. FEN: {}",
                     target_file,
                     self.castling_rook_files[side][0],
                     self.castling_rook_files[side][1],
                     self.to_fen()
                 );
             }

             let k_dst = rank_base + king_file_dst;
             let r_dst = rank_base + rook_file_dst;

             let k_piece = if side == WHITE { K } else { k };
             let r_piece = if side == WHITE { R } else { r };

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
            // 1. Identify what is currently at the target (the piece that moved there)
            let mut moved_piece = self.board[target as usize] as usize;

            // Robustness: If board says NO_PIECE (Desync), try to find it in bitboards
            if moved_piece == NO_PIECE {
                for p_idx in 0..12 {
                    if self.bitboards[p_idx].get_bit(target) {
                        moved_piece = p_idx;
                        break;
                    }
                }
            }

            // 2. Remove it from target (NNUE update)
            if moved_piece != NO_PIECE {
                removed.push((moved_piece, target as usize));
                self.bitboards[moved_piece].pop_bit(target);
                if moved_piece == P || moved_piece == p {
                     self.pawn_key ^= zobrist::piece_key(moved_piece, target as usize);
                }
            }

            self.board[target as usize] = NO_PIECE as u8;
            self.occupancies[side].pop_bit(target);
            self.occupancies[BOTH].pop_bit(target);

            // 3. Put the original piece back at the source
            let original_piece = if promotion.is_some() {
                if side == WHITE {
                    P
                } else {
                    p
                }
            } else {
                moved_piece
            };

            // Critical: If moved_piece was NO_PIECE (desync), we might panic here if we use it.
            // But if we are here, we hope moved_piece is valid or original_piece is valid.
            // We'll trust the robustness check above found it.
            if original_piece != NO_PIECE {
                self.bitboards[original_piece].set_bit(source);
                self.board[source as usize] = original_piece as u8;
                self.occupancies[side].set_bit(source);
                self.occupancies[BOTH].set_bit(source);
                added.push((original_piece, source as usize));
                if original_piece == P || original_piece == p {
                    self.pawn_key ^= zobrist::piece_key(original_piece, source as usize);
                }
            }

            // 4. Restore captured piece
            if is_capture {
                let captured = info.captured as usize;
                let cap_sq =
                    if (original_piece == P || original_piece == p) && target == info.en_passant {
                        if side == WHITE {
                            target - 8
                        } else {
                            target + 8
                        }
                    } else {
                        target
                    };

                self.bitboards[captured].set_bit(cap_sq);
                self.board[cap_sq as usize] = captured as u8;
                let enemy_side = 1 - side;
                self.occupancies[enemy_side].set_bit(cap_sq);
                self.occupancies[BOTH].set_bit(cap_sq);
                added.push((captured, cap_sq as usize));
                if captured == P || captured == p {
                    self.pawn_key ^= zobrist::piece_key(captured, cap_sq as usize);
                }
            }
        }

        if let Some(acc) = accumulators {
            if let Some(backup) = info.acc_backup {
                 **acc = backup;
            } else {
                 let k_sq_white = self.bitboards[K].get_lsb_index() as usize;
                 let k_sq_black = self.bitboards[k].get_lsb_index() as usize;

                 acc[WHITE].update(added.as_slice(), removed.as_slice(), WHITE, k_sq_white);
                 acc[BLACK].update(added.as_slice(), removed.as_slice(), BLACK, k_sq_black);
            }
        }

        #[cfg(debug_assertions)]
        {
            if let Err(e) = self.validate_consistency() {
                self.dump_diagnostics(mv, &format!("Post-Unmake Consistency Failure: {}", e));
                panic!("State corrupted after UNMAKE move {:?}: {}", mv, e);
            }
        }
    }

    pub fn validate_consistency(&self) -> Result<(), String> {
        // 1. Check King Counts
        if self.bitboards[K].count_bits() != 1 {
            return Err(format!(
                "White King count is {}",
                self.bitboards[K].count_bits()
            ));
        }
        if self.bitboards[k].count_bits() != 1 {
            return Err(format!(
                "Black King count is {}",
                self.bitboards[k].count_bits()
            ));
        }

        // 2. Validate Board vs Bitboards & Reconstruct Occupancies
        let mut expected_occ_white = Bitboard(0);
        let mut expected_occ_black = Bitboard(0);

        for sq in 0..64 {
            let piece_on_board = self.board[sq] as usize;

            // Check if piece in mailbox exists in exactly one bitboard
            let mut found_in_bitboards = NO_PIECE;

            // Check all bitboards to see if this square is set
            for p_idx in 0..12 {
                if self.bitboards[p_idx].get_bit(sq as u8) {
                    if found_in_bitboards != NO_PIECE {
                        return Err(format!(
                            "Square {}: Multiple pieces in bitboards ({} and {})",
                            sq, found_in_bitboards, p_idx
                        ));
                    }
                    found_in_bitboards = p_idx;
                }
            }

            if piece_on_board != found_in_bitboards {
                return Err(format!(
                    "DESYNC at square {}: Mailbox says piece {}, but Bitboards say {}",
                    sq, piece_on_board, found_in_bitboards
                ));
            }

            // Accumulate expected occupancies
            if piece_on_board != NO_PIECE {
                let mask = Bitboard(1u64 << sq);
                if piece_on_board <= K {
                    expected_occ_white = expected_occ_white | mask;
                } else {
                    expected_occ_black = expected_occ_black | mask;
                }
            }
        }

        // 3. Check Occupancies
        if self.occupancies[WHITE] != expected_occ_white {
            return Err(format!(
                "White occupancy mismatch. stored: {:x}, calc: {:x}",
                self.occupancies[WHITE].0, expected_occ_white.0
            ));
        }
        if self.occupancies[BLACK] != expected_occ_black {
            return Err(format!(
                "Black occupancy mismatch. stored: {:x}, calc: {:x}",
                self.occupancies[BLACK].0, expected_occ_black.0
            ));
        }
        if self.occupancies[BOTH] != (expected_occ_white | expected_occ_black) {
            return Err("Both occupancy mismatch".to_string());
        }

        // 4. Check En Passant Validity
        if self.en_passant != 64 {
            let ep_rank = self.en_passant / 8;
            let ep_file = self.en_passant % 8;

            // Valid ranks for EP square are 2 (index 2, rank 3) and 5 (index 5, rank 6)
            // If side to move is White, EP square must be rank 5 (behind black pawn)
            // If side to move is Black, EP square must be rank 2 (behind white pawn)

            let (valid_rank, pawn_rank, enemy_pawn) = if self.side_to_move == WHITE {
                (
                    5, 4, p,
                ) // White to move, capture Black pawn. EP sq on rank 6 (index 5). Pawn on rank 5 (index 4).
            } else {
                (
                    2, 3, P,
                ) // Black to move, capture White pawn. EP sq on rank 3 (index 2). Pawn on rank 4 (index 3).
            };

            if ep_rank != valid_rank {
                return Err(format!(
                    "Invalid En Passant rank: {}. Side to move: {}",
                    ep_rank, self.side_to_move
                ));
            }

            let pawn_sq = pawn_rank * 8 + ep_file;
            if self.board[pawn_sq as usize] as usize != enemy_pawn {
                return Err(format!("No enemy pawn for En Passant at square {}", pawn_sq));
            }
        }

        Ok(())
    }

    pub fn is_consistent(&self) -> bool {
        self.validate_consistency().is_ok()
    }

    // Helper to clear a square completely (Ghost busting)
    fn clear_square(&mut self, sq: u8) {
        for p_idx in 0..12 {
            if self.bitboards[p_idx].get_bit(sq) {
                self.bitboards[p_idx].pop_bit(sq);
            }
        }
        self.occupancies[WHITE].pop_bit(sq);
        self.occupancies[BLACK].pop_bit(sq);
        self.occupancies[BOTH].pop_bit(sq);
    }

    pub fn is_move_consistent(&self, mv: Move) -> bool {
        let source = mv.source() as usize;
        let target = mv.target() as usize;

        // 1. Source Consistency
        let piece = self.board[source];
        if piece == NO_PIECE as u8 {
            return false;
        }

        // 2. Capture Consistency
        if mv.is_capture() {
            // Target occupied?
            if self.board[target] != NO_PIECE as u8 {
                return true;
            }
            // En Passant?
            let is_pawn = piece == P as u8 || piece == p as u8;
            if is_pawn && target as u8 == self.en_passant {
                return true;
            }
            // Otherwise inconsistent capture
            return false;
        } else {
            // Quiet Move
            // Target empty?
            if self.board[target] == NO_PIECE as u8 {
                return true;
            }
            // Castling? (King moves to own Rook)
            let piece_type = piece as usize;
            if piece_type == K || piece_type == k {
                let target_piece = self.board[target] as usize;
                let rook_type = if self.side_to_move == WHITE { R } else { r };
                if target_piece == rook_type {
                    return true;
                }
            }
            // Otherwise inconsistent quiet move (target occupied)
            return false;
        }
    }

    pub fn dump_diagnostics(&self, mv: Move, reason: &str) {
        eprintln!("=== DIAGNOSTIC DUMP: {} ===", reason);
        eprintln!("Move: {:?}, FEN: {}", mv, self.to_fen());
        eprintln!("Source: {}, Target: {}", mv.source(), mv.target());
        eprintln!("Mailbox (sq: piece):");
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                let piece = self.board[sq as usize];
                if piece != NO_PIECE as u8 {
                    eprintln!("  sq {}: {}", sq, piece);
                }
            }
        }
        eprintln!("Bitboards:");
        for piece_idx in 0..12 {
            eprintln!("  BB[{}]: {:016x}", piece_idx, self.bitboards[piece_idx].0);
        }
        eprintln!("Occupancies:");
        eprintln!("  White: {:016x}", self.occupancies[WHITE].0);
        eprintln!("  Black: {:016x}", self.occupancies[BLACK].0);
        eprintln!("  Both:  {:016x}", self.occupancies[BOTH].0);
        eprintln!("==============================");
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
        for f in (k_file + 1..8).rev() {
            let sq = rank * 8 + f;
            if rooks.get_bit(sq) {
                return Some(f);
            }
        }
    } else {
        for f in 0..k_file {
            let sq = rank * 8 + f;
            if rooks.get_bit(sq) {
                return Some(f);
            }
        }
    }
    None
}
