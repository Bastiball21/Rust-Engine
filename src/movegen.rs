#![allow(non_upper_case_globals)]
use crate::bitboard::{self, Bitboard};
use crate::state::{b, k, n, p, q, r, GameState, Move, B, BLACK, BOTH, K, N, P, Q, R, WHITE};
use std::cmp::{max, min};
use std::sync::OnceLock;

// --- SAFE GLOBAL TABLES ---
static KNIGHT_TABLE: OnceLock<[Bitboard; 64]> = OnceLock::new();
static KING_TABLE: OnceLock<[Bitboard; 64]> = OnceLock::new();

pub fn init_move_tables() {
    KNIGHT_TABLE.get_or_init(|| {
        let mut table = [Bitboard(0); 64];
        for square in 0..64 {
            table[square] = bitboard::mask_knight_attacks(square as u8);
        }
        table
    });

    KING_TABLE.get_or_init(|| {
        let mut table = [Bitboard(0); 64];
        for square in 0..64 {
            table[square] = bitboard::mask_king_attacks(square as u8);
        }
        table
    });
}

// --- ACCESSORS ---

#[inline(always)]
pub fn get_knight_attacks(sq: u8) -> Bitboard {
    if sq >= 64 {
        return Bitboard(0);
    } // Safety check
    KNIGHT_TABLE.get().expect("Move tables not initialized")[sq as usize]
}

#[inline(always)]
pub fn get_king_attacks(sq: u8) -> Bitboard {
    if sq >= 64 {
        return Bitboard(0);
    } // Safety check
    KING_TABLE.get().expect("Move tables not initialized")[sq as usize]
}

#[derive(Clone, Copy)]
pub struct MoveList {
    pub moves: [Move; 256],
    pub count: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self {
            moves: [Move {
                source: 0,
                target: 0,
                promotion: None,
                is_capture: false,
            }; 256],
            count: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        if self.count < 256 {
            self.moves[self.count] = m;
            self.count += 1;
        }
    }
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
        self.list.push(Move {
            source,
            target,
            promotion,
            is_capture,
        });
    }

    #[inline(always)]
    fn add_promotion_moves(&mut self, source: u8, target: u8, is_capture: bool) {
        self.add_move(source, target, Some(Q), is_capture);
        self.add_move(source, target, Some(R), is_capture);
        self.add_move(source, target, Some(B), is_capture);
        self.add_move(source, target, Some(N), is_capture);
    }

    pub fn generate_moves(&mut self, state: &GameState) {
        let side = state.side_to_move;
        let enemy = 1 - side;
        let occupancy_all = state.occupancies[BOTH];
        let occupancy_friendly = state.occupancies[side];

        // Prevent generating moves that capture the King (Pseudo-Legal safety)
        let enemy_king_bb = state.bitboards[if enemy == WHITE { K } else { k }];
        let occupancy_enemy = state.occupancies[enemy] & !enemy_king_bb;

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
                    self.add_promotion_moves(src, target, false);
                } else {
                    self.add_move(src, target, None, false);
                    if rank == start_rank {
                        let double = (src as i8 + (16 * direction)) as u8;
                        if !occupancy_all.get_bit(double) {
                            self.add_move(src, double, None, false);
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
                        self.add_promotion_moves(src, t, true);
                    } else {
                        self.add_move(src, t, None, true);
                    }
                } else if t == state.en_passant {
                    self.add_move(src, t, None, true);
                }
            }
            if file < 7 {
                let t = (src as i8 + (8 * direction) + 1) as u8;
                if occupancy_enemy.get_bit(t) {
                    if (t / 8) == promo_rank {
                        self.add_promotion_moves(src, t, true);
                    } else {
                        self.add_move(src, t, None, true);
                    }
                } else if t == state.en_passant {
                    self.add_move(src, t, None, true);
                }
            }
        }

        // KNIGHTS
        let knight_type = if side == WHITE { N } else { n };
        let mut knights = state.bitboards[knight_type];
        while knights.0 != 0 {
            let src = knights.get_lsb_index() as u8;
            knights.pop_bit(src);
            let mut attacks = get_knight_attacks(src) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
        }

        // BISHOPS
        let bishop_type = if side == WHITE { B } else { b };
        let mut bishops = state.bitboards[bishop_type];
        while bishops.0 != 0 {
            let src = bishops.get_lsb_index() as u8;
            bishops.pop_bit(src);
            let mut attacks =
                bitboard::get_bishop_attacks(src, occupancy_all) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                // Explicitly check for King Capture to avoid illegal generation
                if state.occupancies[enemy].get_bit(t) && !occupancy_enemy.get_bit(t) {
                    continue;
                }
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
        }

        // ROOKS
        let rook_type = if side == WHITE { R } else { r };
        let mut rooks = state.bitboards[rook_type];
        while rooks.0 != 0 {
            let src = rooks.get_lsb_index() as u8;
            rooks.pop_bit(src);
            let mut attacks = bitboard::get_rook_attacks(src, occupancy_all) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                if state.occupancies[enemy].get_bit(t) && !occupancy_enemy.get_bit(t) {
                    continue;
                }
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
        }

        // QUEENS
        let queen_type = if side == WHITE { Q } else { q };
        let mut queens = state.bitboards[queen_type];
        while queens.0 != 0 {
            let src = queens.get_lsb_index() as u8;
            queens.pop_bit(src);
            let mut attacks = bitboard::get_queen_attacks(src, occupancy_all) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                if state.occupancies[enemy].get_bit(t) && !occupancy_enemy.get_bit(t) {
                    continue;
                }
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
        }

        // KING
        let king_type = if side == WHITE { K } else { k };
        let king = state.bitboards[king_type];
        if king.0 != 0 {
            let src = king.get_lsb_index() as u8;
            let mut attacks = get_king_attacks(src) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                if state.occupancies[enemy].get_bit(t) && !occupancy_enemy.get_bit(t) {
                    continue;
                }
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }

            // CASTLING (Chess960 Unified)
            self.generate_castling_moves(state, src, side);
        }
    }

    // New helper method for castling
    #[inline(always)]
    fn generate_castling_moves(&mut self, state: &GameState, king_sq: u8, side: usize) {
        let enemy = 1 - side;
        let rank_base = if side == WHITE { 0 } else { 56 };

        // Check both sides: 0 = Kingside, 1 = Queenside
        for side_idx in 0..2 {
            // Check rights mask
            let mask = if side == WHITE {
                if side_idx == 0 {
                    1
                } else {
                    2
                }
            } else {
                if side_idx == 0 {
                    4
                } else {
                    8
                }
            };

            if (state.castling_rights & mask) != 0 {
                let file = state.castling_rook_files[side][side_idx];
                let rook_sq = rank_base + file;

                // Verify rook is actually there (it should be if rights are set, but let's be safe)
                // Note: bitboards[R/r] check
                let rook_type = if side == WHITE { R } else { r };
                if !state.bitboards[rook_type].get_bit(rook_sq) {
                    continue;
                }

                // Determine destinations
                // Standard convention:
                // Kingside (side 0): King -> g-file (6/62), Rook -> f-file (5/61)
                // Queenside (side 1): King -> c-file (2/58), Rook -> d-file (3/59)

                let (k_dst_file, r_dst_file) = if side_idx == 0 {
                    (6, 5) // g, f
                } else {
                    (2, 3) // c, d
                };

                let k_dst = rank_base + k_dst_file;
                let r_dst = rank_base + r_dst_file;

                // 1. Path Obstruction Check
                // "All squares between the king's initial and final squares (including the final square),
                // and all squares between the rook's initial and final squares (including the final square),
                // must be vacant except for the king and rook."

                if !self.is_path_clear(state, king_sq, k_dst, king_sq, rook_sq) {
                    continue;
                }
                if !self.is_path_clear(state, rook_sq, r_dst, king_sq, rook_sq) {
                    continue;
                }

                // 2. King Safety Check
                // King must not be in check at start.
                // King must not pass through check.
                // King must not end in check.

                if is_square_attacked(state, king_sq, enemy) {
                    continue;
                }

                // Check path for King
                // Logic: Iterate from adjacent square to dest.
                // However, King moves one by one. In 960, the "leap" is conceptual.
                // But safety rules apply to the squares the king crosses *if it were moving normally*.
                // Standard Chess: King moves e1-f1-g1. Checks e1, f1, g1.
                // 960: King path from Start to End. All squares on interval must be safe.
                // Note: If start == end, no squares to check besides start (already checked).

                let mut safe = true;
                let start = min(king_sq, k_dst);
                let end = max(king_sq, k_dst);

                for sq in start..=end {
                    if sq == king_sq {
                        continue;
                    } // Already checked
                    if is_square_attacked(state, sq, enemy) {
                        safe = false;
                        break;
                    }
                }

                if safe {
                    // Valid Castling Move
                    // Target is the Rook's square.
                    self.add_move(king_sq, rook_sq, None, false);
                }
            }
        }
    }

    fn is_path_clear(
        &self,
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
}

pub fn is_square_attacked(state: &GameState, square: u8, attacker_side: usize) -> bool {
    if square >= 64 {
        return false;
    }

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

    let knights = if attacker_side == WHITE {
        state.bitboards[N]
    } else {
        state.bitboards[n]
    };
    if (get_knight_attacks(square) & knights).0 != 0 {
        return true;
    }

    let king = if attacker_side == WHITE {
        state.bitboards[K]
    } else {
        state.bitboards[k]
    };
    if (get_king_attacks(square) & king).0 != 0 {
        return true;
    }

    let occupancy = state.occupancies[BOTH];
    let rooks = if attacker_side == WHITE {
        state.bitboards[R] | state.bitboards[Q]
    } else {
        state.bitboards[r] | state.bitboards[q]
    };
    if (bitboard::get_rook_attacks(square, occupancy) & rooks).0 != 0 {
        return true;
    }

    let bishops = if attacker_side == WHITE {
        state.bitboards[B] | state.bitboards[Q]
    } else {
        state.bitboards[b] | state.bitboards[q]
    };
    if (bitboard::get_bishop_attacks(square, occupancy) & bishops).0 != 0 {
        return true;
    }

    false
}

// PATCH #2 & #3: Fixed Variable Name and Fast Check Detection
pub fn gives_check(state: &GameState, mv: Move) -> bool {
    let side = state.side_to_move;
    let enemy = 1 - side;
    let enemy_king = state.bitboards[if enemy == WHITE { K } else { k }].get_lsb_index() as u8;

    let from = mv.source;
    let to = mv.target;

    // 0. Locate Piece Type (Expensive lookups minimized)
    let mut piece = 12;
    let start = if side == WHITE { P } else { p };
    let end = if side == WHITE { K } else { k };

    // FIX: Renamed loop variable 'p' to 'current_p' to avoid conflict with imported constant 'p'
    for current_p in start..=end {
        if state.bitboards[current_p].get_bit(from) {
            piece = current_p;
            break;
        }
    }

    // Castling and promotions are rare; fallback to safe slow check to prevent bugs
    // Chess960 Castling: Check if target is own rook
    if piece == K || piece == k {
        // If target contains friendly rook, it's castling
        let friendly_rooks = state.bitboards[if side == WHITE { R } else { r }];
        if friendly_rooks.get_bit(to) {
            return slow_gives_check(state, mv);
        }
        // Legacy distance check for safety
        if (from as i8 - to as i8).abs() == 2 {
            return slow_gives_check(state, mv);
        }
    }

    if mv.promotion.is_some() {
        return slow_gives_check(state, mv);
    }
    if (piece == P || piece == p) && to == state.en_passant {
        return slow_gives_check(state, mv);
    }

    // 1. Direct Check
    // "Imagine" the board: 'from' is empty, 'to' has 'piece'.
    // We only need to check if 'piece' at 'to' attacks 'enemy_king'.
    let occ = (state.occupancies[BOTH].0 & !(1u64 << from)) | (1u64 << to);
    let occ_bb = Bitboard(occ);

    // Check attacks from 'to'
    let attacks = match piece {
        N | n => get_knight_attacks(to),
        B | b => bitboard::get_bishop_attacks(to, occ_bb),
        R | r => bitboard::get_rook_attacks(to, occ_bb),
        Q | q => bitboard::get_queen_attacks(to, occ_bb),
        P => bitboard::pawn_attacks(Bitboard(1u64 << to), WHITE),
        p => bitboard::pawn_attacks(Bitboard(1u64 << to), BLACK),
        K | k => get_king_attacks(to),
        _ => Bitboard(0),
    };

    if attacks.get_bit(enemy_king) {
        return true;
    }

    // 2. Discovered Check
    // Only possible if 'from' was blocking a line from a friendly slider to the king.
    // Check if 'from' is on a ray between friendly slider and enemy king.

    // Optimization: If 'from' is not aligned with king, no discovery.
    // FIX: Added 'bitboard::' prefix here
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

    // See if a slider attacks the king through the hole left by 'from'
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
