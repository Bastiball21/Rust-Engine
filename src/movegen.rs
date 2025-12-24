#![allow(non_upper_case_globals)]
use crate::bitboard::{self, Bitboard};
use crate::state::{b, k, n, p, q, r, GameState, Move, B, BLACK, BOTH, K, N, P, Q, R, WHITE, NO_PIECE};
use std::cmp::{max, min};

pub fn init_move_tables() {
    // Moved to bitboard.rs init_magic_tables
}

pub const MAX_MOVES: usize = 512;

#[derive(Clone, Copy)]
pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub count: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            moves: [Move::default(); MAX_MOVES],
            count: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        if self.count < MAX_MOVES {
            self.moves[self.count] = m;
            self.count += 1;
        } else {
            #[cfg(debug_assertions)]
            panic!(
                "MoveList overflow! Attempted to push move {} (limit {})",
                self.count + 1,
                MAX_MOVES
            );
        }
    }
}

#[derive(PartialEq)]
pub enum GenType {
    All,
    Captures,
    Quiets,
}

pub struct MoveGenerator {
    pub list: MoveList,
}

impl MoveGenerator {
    pub fn new() -> Self {
        Self {
            list: MoveList::new(),
        }
    }

    #[inline(always)]
    fn add_move(&mut self, source: u8, target: u8, promotion: Option<usize>, is_capture: bool) {
        self.list.push(Move::new(
            source,
            target,
            promotion,
            is_capture,
        ));
    }

    #[inline(always)]
    fn add_promotion_moves(&mut self, source: u8, target: u8, is_capture: bool, gen_type: &GenType) {
        if *gen_type == GenType::Quiets {
            return;
        }

        self.add_move(source, target, Some(Q), is_capture);
        self.add_move(source, target, Some(R), is_capture);
        self.add_move(source, target, Some(B), is_capture);
        self.add_move(source, target, Some(N), is_capture);
    }

    pub fn generate_moves(&mut self, state: &GameState) {
        self.generate_moves_type(state, GenType::All);
    }

    pub fn generate_moves_type(&mut self, state: &GameState, gen_type: GenType) {
        let side = state.side_to_move;
        let enemy = 1 - side;
        let occupancy_all = state.occupancies[BOTH];
        let occupancy_friendly = state.occupancies[side];

        // Prevent generating moves that capture the King (Pseudo-Legal safety)
        let enemy_king_bb = state.bitboards[if enemy == WHITE { K } else { k }];
        let occupancy_enemy = state.occupancies[enemy] & !enemy_king_bb;

        // TARGET MASKS
        let target_mask = match gen_type {
            GenType::All => !occupancy_friendly & !enemy_king_bb,
            GenType::Captures => occupancy_enemy,
            GenType::Quiets => !occupancy_all,
        };

        // PAWNS
        let (pawn_type, start_rank, promo_rank, direction) = if side == WHITE {
            (P, 1, 7, 1)
        } else {
            (p, 6, 0, -1)
        };

        let mut pawns = state.bitboards[pawn_type];
        while pawns.0 != 0 {
            let src = pawns.get_lsb_index() as u8;
            pawns.pop_bit(src);
            let rank = src / 8;

            let target = (src as i8 + (8 * direction)) as u8;
            // Quiet pushes
            if !occupancy_all.get_bit(target) {
                if (target / 8) == promo_rank {
                    if gen_type != GenType::Quiets {
                        self.add_promotion_moves(src, target, false, &gen_type);
                    }
                } else {
                    if gen_type != GenType::Captures {
                        self.add_move(src, target, None, false);
                    }
                    if rank == start_rank {
                        let double = (src as i8 + (16 * direction)) as u8;
                        if !occupancy_all.get_bit(double) {
                            if gen_type != GenType::Captures {
                                self.add_move(src, double, None, false);
                            }
                        }
                    }
                }
            }

            // Captures
            let file = src % 8;
            if file > 0 {
                let t = (src as i8 + (8 * direction) - 1) as u8;
                if occupancy_enemy.get_bit(t) {
                    if (t / 8) == promo_rank {
                        if gen_type != GenType::Quiets {
                            self.add_promotion_moves(src, t, true, &gen_type);
                        }
                    } else {
                        if gen_type != GenType::Quiets {
                            self.add_move(src, t, None, true);
                        }
                    }
                } else if t == state.en_passant {
                    if gen_type != GenType::Quiets {
                        self.add_move(src, t, None, true);
                    }
                }
            }
            if file < 7 {
                let t = (src as i8 + (8 * direction) + 1) as u8;
                if occupancy_enemy.get_bit(t) {
                    if (t / 8) == promo_rank {
                        if gen_type != GenType::Quiets {
                            self.add_promotion_moves(src, t, true, &gen_type);
                        }
                    } else {
                        if gen_type != GenType::Quiets {
                            self.add_move(src, t, None, true);
                        }
                    }
                } else if t == state.en_passant {
                    if gen_type != GenType::Quiets {
                        self.add_move(src, t, None, true);
                    }
                }
            }
        }

        // KNIGHTS
        let knight_type = if side == WHITE { N } else { n };
        let mut knights = state.bitboards[knight_type];
        while knights.0 != 0 {
            let src = knights.get_lsb_index() as u8;
            knights.pop_bit(src);
            let mut attacks = bitboard::get_knight_attacks(src) & target_mask;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                let is_capture = occupancy_enemy.get_bit(t);
                self.add_move(src, t, None, is_capture);
            }
        }

        // BISHOPS
        let bishop_type = if side == WHITE { B } else { b };
        let mut bishops = state.bitboards[bishop_type];
        while bishops.0 != 0 {
            let src = bishops.get_lsb_index() as u8;
            bishops.pop_bit(src);
            let mut attacks =
                bitboard::get_bishop_attacks(src, occupancy_all) & target_mask;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                let is_capture = occupancy_enemy.get_bit(t);
                if is_capture && state.board[t as usize] as usize == K { continue; } // Safety
                self.add_move(src, t, None, is_capture);
            }
        }

        // ROOKS
        let rook_type = if side == WHITE { R } else { r };
        let mut rooks = state.bitboards[rook_type];
        while rooks.0 != 0 {
            let src = rooks.get_lsb_index() as u8;
            rooks.pop_bit(src);
            let mut attacks = bitboard::get_rook_attacks(src, occupancy_all) & target_mask;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                let is_capture = occupancy_enemy.get_bit(t);
                if is_capture && state.board[t as usize] as usize == K { continue; } // Safety
                self.add_move(src, t, None, is_capture);
            }
        }

        // QUEENS
        let queen_type = if side == WHITE { Q } else { q };
        let mut queens = state.bitboards[queen_type];
        while queens.0 != 0 {
            let src = queens.get_lsb_index() as u8;
            queens.pop_bit(src);
            let mut attacks = bitboard::get_queen_attacks(src, occupancy_all) & target_mask;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                let is_capture = occupancy_enemy.get_bit(t);
                if is_capture && state.board[t as usize] as usize == K { continue; } // Safety
                self.add_move(src, t, None, is_capture);
            }
        }

        // KING
        let king_type = if side == WHITE { K } else { k };
        let king = state.bitboards[king_type];
        if king.0 != 0 {
            let src = king.get_lsb_index() as u8;
            let mut attacks = bitboard::get_king_attacks(src) & target_mask;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                let is_capture = occupancy_enemy.get_bit(t);
                if is_capture && state.board[t as usize] as usize == K { continue; } // Safety
                self.add_move(src, t, None, is_capture);
            }

            // CASTLING (Chess960 Unified)
            if gen_type != GenType::Captures {
                self.generate_castling_moves(state, src, side);
            }
        }
    }

    #[inline(always)]
    fn generate_castling_moves(&mut self, state: &GameState, king_sq: u8, side: usize) {
        let enemy = 1 - side;
        let rank_base = if side == WHITE { 0 } else { 56 };

        for side_idx in 0..2 {
            let mask = if side == WHITE {
                if side_idx == 0 { 1 } else { 2 }
            } else {
                if side_idx == 0 { 4 } else { 8 }
            };

            if (state.castling_rights & mask) != 0 {
                let file = state.castling_rook_files[side][side_idx];
                let rook_sq = rank_base + file;

                let rook_type = if side == WHITE { R } else { r };
                if !state.bitboards[rook_type].get_bit(rook_sq) {
                    continue;
                }

                let (k_dst_file, r_dst_file) = if side_idx == 0 {
                    (6, 5) // g, f
                } else {
                    (2, 3) // c, d
                };

                let k_dst = rank_base + k_dst_file;
                let r_dst = rank_base + r_dst_file;

                if !is_path_clear(state, king_sq, k_dst, king_sq, rook_sq) {
                    continue;
                }
                if !is_path_clear(state, rook_sq, r_dst, king_sq, rook_sq) {
                    continue;
                }

                if is_square_attacked(state, king_sq, enemy) {
                    continue;
                }

                let mut safe = true;
                let start = min(king_sq, k_dst);
                let end = max(king_sq, k_dst);

                for sq in start..=end {
                    if sq == king_sq {
                        continue;
                    }
                    if is_square_attacked(state, sq, enemy) {
                        safe = false;
                        break;
                    }
                }

                if safe {
                    self.add_move(king_sq, rook_sq, None, false);
                }
            }
        }
    }
}

pub fn is_path_clear(
    state: &GameState,
    from: u8,
    to: u8,
    ignore_k: u8,
    ignore_r: u8,
) -> bool {
    let start = min(from, to);
    let end = max(from, to);

    for sq in start..=end {
        if sq == ignore_k || sq == ignore_r {
            continue;
        }
        if state.occupancies[BOTH].get_bit(sq) {
            return false;
        }
    }
    true
}

pub fn is_square_attacked(state: &GameState, square: u8, attacker_side: usize) -> bool {
    if square >= 64 {
        return false;
    }

    // 1. Pawns (Cheapest: Bit Shifts)
    if attacker_side == WHITE {
        if square > 8 {
            if (square % 8) > 0 && state.bitboards[P].get_bit(square - 9) {
                return true;
            }
            if (square % 8) < 7 && state.bitboards[P].get_bit(square - 7) {
                return true;
            }
        }
    } else {
        if square < 56 {
            if (square % 8) > 0 && state.bitboards[p].get_bit(square + 7) {
                return true;
            }
            if (square % 8) < 7 && state.bitboards[p].get_bit(square + 9) {
                return true;
            }
        }
    }

    // 2. Knights (Cheap: Array Lookup)
    let knights = if attacker_side == WHITE {
        state.bitboards[N]
    } else {
        state.bitboards[n]
    };
    if (bitboard::get_knight_attacks(square) & knights).0 != 0 {
        return true;
    }

    let occupancy = state.occupancies[BOTH];

    // 3. Bishops/Queens (Diagonal)
    let bishops = if attacker_side == WHITE {
        state.bitboards[B] | state.bitboards[Q]
    } else {
        state.bitboards[b] | state.bitboards[q]
    };
    if (bitboard::get_bishop_attacks(square, occupancy) & bishops).0 != 0 {
        return true;
    }

    // 4. Rooks/Queens (Orthogonal)
    let rooks = if attacker_side == WHITE {
        state.bitboards[R] | state.bitboards[Q]
    } else {
        state.bitboards[r] | state.bitboards[q]
    };
    if (bitboard::get_rook_attacks(square, occupancy) & rooks).0 != 0 {
        return true;
    }

    // 5. King (Rare)
    let king = if attacker_side == WHITE {
        state.bitboards[K]
    } else {
        state.bitboards[k]
    };
    if (bitboard::get_king_attacks(square) & king).0 != 0 {
        return true;
    }

    false
}

pub fn gives_check(state: &GameState, mv: Move) -> bool {
    let side = state.side_to_move;
    let enemy = 1 - side;
    let enemy_king = state.bitboards[if enemy == WHITE { K } else { k }].get_lsb_index() as u8;

    let from = mv.source();
    let to = mv.target();

    let piece = state.board[from as usize] as usize;

    if piece == K || piece == k {
        let friendly_rooks = state.bitboards[if side == WHITE { R } else { r }];
        if friendly_rooks.get_bit(to) {
            return slow_gives_check(state, mv);
        }
        if (from as i8 - to as i8).abs() == 2 {
            return slow_gives_check(state, mv);
        }
    }

    if mv.promotion().is_some() {
        return slow_gives_check(state, mv);
    }
    if (piece == P || piece == p) && to == state.en_passant {
        return slow_gives_check(state, mv);
    }

    let occ = (state.occupancies[BOTH].0 & !(1u64 << from)) | (1u64 << to);
    let occ_bb = Bitboard(occ);

    let attacks = match piece {
        N | n => bitboard::get_knight_attacks(to),
        B | b => bitboard::get_bishop_attacks(to, occ_bb),
        R | r => bitboard::get_rook_attacks(to, occ_bb),
        Q | q => bitboard::get_queen_attacks(to, occ_bb),
        P => bitboard::pawn_attacks(Bitboard(1u64 << to), WHITE),
        p => bitboard::pawn_attacks(Bitboard(1u64 << to), BLACK),
        K | k => bitboard::get_king_attacks(to),
        _ => Bitboard(0),
    };

    if attacks.get_bit(enemy_king) {
        return true;
    }

    if (bitboard::get_queen_attacks(from, Bitboard(0)) & Bitboard(1u64 << enemy_king)).0 == 0 {
        return false;
    }

    let friendly_rooks = if side == WHITE {
        state.bitboards[R] | state.bitboards[Q]
    } else {
        state.bitboards[r] | state.bitboards[q]
    };
    let friendly_bishops = if side == WHITE {
        state.bitboards[B] | state.bitboards[Q]
    } else {
        state.bitboards[b] | state.bitboards[q]
    };

    if (bitboard::get_rook_attacks(enemy_king, occ_bb) & friendly_rooks).0 != 0 {
        return true;
    }
    if (bitboard::get_bishop_attacks(enemy_king, occ_bb) & friendly_bishops).0 != 0 {
        return true;
    }

    false
}

fn slow_gives_check(state: &GameState, mv: Move) -> bool {
    let next_state = state.make_move(mv);
    let king_sq =
        next_state.bitboards[if state.side_to_move == WHITE { k } else { K }].get_lsb_index() as u8;
    is_square_attacked(&next_state, king_sq, next_state.side_to_move)
}

/// Strictly validates if a move is pseudo-legal according to the rules of chess.
/// This checks geometric validity, piece ownership, capture rules, and castling prerequisites (rights/occupancy).
/// It does NOT check if the King is left in check (full legality).
pub fn is_move_pseudo_legal(state: &GameState, mv: Move) -> bool {
    let from = mv.source();
    let to = mv.target();
    let side = state.side_to_move;

    // 1. Basic Validity
    if from >= 64 || to >= 64 || from == to {
        return false;
    }

    // 2. Source Piece Existence & Ownership
    // Use mailbox for speed, but rely on bitboards for strictness if desired.
    // Here we trust mailbox consistent with bitboards as checked by `validate_consistency` in debug.
    let piece_type = state.board[from as usize] as usize;
    if piece_type == NO_PIECE {
        return false;
    }

    if side == WHITE {
        if piece_type > K { return false; }
    } else {
        if piece_type < p || piece_type > k { return false; }
    }

    // 3. Target Validity
    let target_piece = state.board[to as usize] as usize;
    let is_capture = mv.is_capture();
    let is_castling = (piece_type == K || piece_type == k)
        && (target_piece == if side == WHITE { R } else { r });

    if target_piece != NO_PIECE {
        let is_friendly = if side == WHITE { target_piece <= K } else { target_piece >= p && target_piece <= k };
        if is_friendly && !is_castling {
            return false;
        }
    } else {
        // Target is empty.
        // If move flag says capture, it MUST be En Passant.
        // If it's not EP, and flag is capture -> Invalid.
        if is_capture {
            let is_ep = (piece_type == P || piece_type == p) && to == state.en_passant;
            if !is_ep {
                return false;
            }
        }
    }

    // 4. Geometric & Rule Validation
    let occupancy = state.occupancies[BOTH];

    match piece_type {
        P | p => {
             let (direction, start_rank, promo_rank) = if side == WHITE { (1, 1, 7) } else { (-1, 6, 0) };
             let rank = from / 8;
             let is_promo = (to / 8) == promo_rank;

             // Check Promotion Flag matches geometry
             if is_promo != mv.promotion().is_some() {
                 return false;
             }

             if is_capture {
                 // Capture Logic: Diagonals
                 let diff = (to as i8) - (from as i8);
                 if diff == 8 * direction - 1 || diff == 8 * direction + 1 {
                     // Must be occupied by enemy OR En Passant square
                     if target_piece == NO_PIECE && to != state.en_passant {
                         return false;
                     }
                     return true;
                 }
                 return false;
             } else {
                 // Push Logic
                 if target_piece != NO_PIECE { return false; } // Must be empty

                 let diff = (to as i8) - (from as i8);

                 // Single Push
                 if diff == 8 * direction {
                     return true;
                 }

                 // Double Push
                 if diff == 16 * direction {
                     if rank != start_rank { return false; }
                     // Path must be clear
                     let mid = (from as i8 + 8 * direction) as u8;
                     if state.board[mid as usize] != NO_PIECE as u8 { return false; }
                     return true;
                 }

                 return false;
             }
        },
        N | n => {
            let attacks = bitboard::get_knight_attacks(from);
            attacks.get_bit(to)
        },
        B | b => {
            let attacks = bitboard::get_bishop_attacks(from, occupancy);
            attacks.get_bit(to)
        },
        R | r => {
            let attacks = bitboard::get_rook_attacks(from, occupancy);
            attacks.get_bit(to)
        },
        Q | q => {
            let attacks = bitboard::get_queen_attacks(from, occupancy);
            attacks.get_bit(to)
        },
        K | k => {
            if is_castling {
                 // Castling Logic
                 // King moves to own Rook
                 // 1. Rights
                 // 2. Path Clear
                 // 3. Not in Check (Start, Path, End) - This is mostly "Legal" not "Pseudo",
                 //    but strict pseudo-legal usually includes path clearance and rights.

                 // Find which side (King or Queen side) based on Rook position
                 let rook_file = to % 8;
                 let rank_base = if side == WHITE { 0 } else { 56 };

                 let mut side_idx = 2; // Invalid
                 if state.castling_rook_files[side][0] == rook_file { side_idx = 0; }
                 else if state.castling_rook_files[side][1] == rook_file { side_idx = 1; }

                 if side_idx == 2 { return false; } // Rook not tracked for castling

                 // Check Rights bitmask
                 let mask = if side == WHITE {
                     if side_idx == 0 { 1 } else { 2 }
                 } else {
                     if side_idx == 0 { 4 } else { 8 }
                 };

                 if (state.castling_rights & mask) == 0 {
                     return false;
                 }

                 // Map to standard King/Rook dests
                 let (k_dst_file, r_dst_file) = if side_idx == 0 { (6, 5) } else { (2, 3) };
                 let k_dst = rank_base + k_dst_file;
                 let r_dst = rank_base + r_dst_file;

                 // Path Clear
                 if !is_path_clear(state, from, k_dst, from, to) { return false; }
                 if !is_path_clear(state, to, r_dst, from, to) { return false; }

                 // Note: We skip "Safe from Check" here as that's full legality.
                 // We only check structural possibility.
                 return true;
            } else {
                let attacks = bitboard::get_king_attacks(from);
                attacks.get_bit(to)
            }
        },
        _ => false
    }
}
