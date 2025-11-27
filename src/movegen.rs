use crate::bitboard::{self, Bitboard};
use crate::state::{GameState, Move, WHITE, BLACK, P, N, B, R, Q, K, p, n, b, r, q, k, BOTH};
use std::sync::OnceLock;

// --- SAFE GLOBAL TABLES ---
// Replaced static mut with thread-safe OnceLock
static KNIGHT_TABLE: OnceLock<[Bitboard; 64]> = OnceLock::new();
static KING_TABLE: OnceLock<[Bitboard; 64]> = OnceLock::new();

pub fn init_move_tables() {
    // Initialize Knight Table
    KNIGHT_TABLE.get_or_init(|| {
        let mut table = [Bitboard(0); 64];
        for square in 0..64 {
            table[square] = bitboard::mask_knight_attacks(square as u8);
        }
        table
    });

    // Initialize King Table
    KING_TABLE.get_or_init(|| {
        let mut table = [Bitboard(0); 64];
        for square in 0..64 {
            table[square] = bitboard::mask_king_attacks(square as u8);
        }
        table
    });
    
    // Ensure magics are ready (safe to call multiple times)
    bitboard::init_magic_tables();
    println!("Move Tables Initialized.");
}

// Helper accessors to avoid raw array access
#[inline(always)]
fn get_knight_attacks(sq: u8) -> Bitboard {
    KNIGHT_TABLE.get().expect("Move tables not initialized")[sq as usize]
}

#[inline(always)]
fn get_king_attacks(sq: u8) -> Bitboard {
    KING_TABLE.get().expect("Move tables not initialized")[sq as usize]
}

// A lightweight list to hold moves. 
#[derive(Clone, Copy)]
pub struct MoveList {
    pub moves: [Move; 256],
    pub count: usize,
}

impl MoveList {
    pub fn new() -> Self {
        Self { 
            moves: [Move { source: 0, target: 0, promotion: None, is_capture: false }; 256], 
            count: 0 
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) { // Added 'mut' here
        if self.count < 256 {
            self.moves[self.count] = m;
            self.count += 1;
        }
    }
}

// Helper for iteration
impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.moves[0..self.count].iter()
    }
}

pub struct MoveGenerator {
    pub list: MoveList,
}

impl MoveGenerator {
    pub fn new() -> Self {
        Self { list: MoveList::new() }
    }

    #[inline(always)]
    fn add_move(&mut self, source: u8, target: u8, promotion: Option<usize>, is_capture: bool) {
        self.list.push(Move { source, target, promotion, is_capture });
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
        let occupancy_enemy = state.occupancies[enemy];
        let occupancy_friendly = state.occupancies[side];

        // PAWNS
        let (pawn_type, start_rank, promo_rank, direction) = if side == WHITE {
            (P, 1, 7, 1)
        } else {
            (p, 6, 0, -1)
        };

        // Added 'mut' to allow popping bits
        let mut pawns = state.bitboards[pawn_type]; 
        while pawns.0 != 0 {
            let src = pawns.get_lsb_index() as u8;
            pawns.pop_bit(src);
            let rank = src / 8;
            
            let target = (src as i8 + (8 * direction)) as u8;
            if !occupancy_all.get_bit(target) {
                if (target / 8) == promo_rank { self.add_promotion_moves(src, target, false); }
                else {
                    self.add_move(src, target, None, false);
                    if rank == start_rank {
                        let double = (src as i8 + (16 * direction)) as u8;
                        if !occupancy_all.get_bit(double) { self.add_move(src, double, None, false); }
                    }
                }
            }
            let file = src % 8;
            if file > 0 {
                let t = (src as i8 + (8 * direction) - 1) as u8;
                if occupancy_enemy.get_bit(t) {
                    if (t / 8) == promo_rank { self.add_promotion_moves(src, t, true); }
                    else { self.add_move(src, t, None, true); }
                } else if t == state.en_passant { self.add_move(src, t, None, true); }
            }
            if file < 7 {
                let t = (src as i8 + (8 * direction) + 1) as u8;
                if occupancy_enemy.get_bit(t) {
                    if (t / 8) == promo_rank { self.add_promotion_moves(src, t, true); }
                    else { self.add_move(src, t, None, true); }
                } else if t == state.en_passant { self.add_move(src, t, None, true); }
            }
        }

        // KNIGHTS
        let knight_type = if side == WHITE { N } else { n };
        let mut knights = state.bitboards[knight_type]; // Added mut
        while knights.0 != 0 {
            let src = knights.get_lsb_index() as u8;
            knights.pop_bit(src);
            // Safe access, added mut to attacks
            let mut attacks = get_knight_attacks(src) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
        }

        // BISHOPS
        let bishop_type = if side == WHITE { B } else { b };
        let mut bishops = state.bitboards[bishop_type]; // Added mut
        while bishops.0 != 0 {
            let src = bishops.get_lsb_index() as u8;
            bishops.pop_bit(src);
            let mut attacks = bitboard::get_bishop_attacks(src, occupancy_all) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
        }

        // ROOKS
        let rook_type = if side == WHITE { R } else { r };
        let mut rooks = state.bitboards[rook_type]; // Added mut
        while rooks.0 != 0 {
            let src = rooks.get_lsb_index() as u8;
            rooks.pop_bit(src);
            let mut attacks = bitboard::get_rook_attacks(src, occupancy_all) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
        }

        // QUEENS
        let queen_type = if side == WHITE { Q } else { q };
        let mut queens = state.bitboards[queen_type]; // Added mut
        while queens.0 != 0 {
            let src = queens.get_lsb_index() as u8;
            queens.pop_bit(src);
            let mut attacks = bitboard::get_queen_attacks(src, occupancy_all) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
        }

        // KING
        let king_type = if side == WHITE { K } else { k };
        let king = state.bitboards[king_type];
        if king.0 != 0 {
            let src = king.get_lsb_index() as u8;
            // Safe access, added mut
            let mut attacks = get_king_attacks(src) & !occupancy_friendly;
            while attacks.0 != 0 {
                let t = attacks.get_lsb_index() as u8;
                attacks.pop_bit(t);
                self.add_move(src, t, None, occupancy_enemy.get_bit(t));
            }
            

            // CASTLING
            // CRITICAL FIX: Must check !is_square_attacked(king_sq) to ensure we aren't castling OUT of check.
            if side == WHITE {
                // White King Side (e1 -> g1)
                if (state.castling_rights & 1) != 0 {
                    // Check f1 and g1 empty
                    if !occupancy_all.get_bit(5) && !occupancy_all.get_bit(6) {
                        // Check e1(4), f1(5), g1(6) not attacked
                        if !is_square_attacked(state, 4, BLACK) 
                           && !is_square_attacked(state, 5, BLACK)
                           && !is_square_attacked(state, 6, BLACK) { // Added check for target square (g1) just in case
                             self.add_move(4, 6, None, false);
                        }
                    }
                }
                // White Queen Side (e1 -> c1)
                if (state.castling_rights & 2) != 0 {
                    if !occupancy_all.get_bit(1) && !occupancy_all.get_bit(2) && !occupancy_all.get_bit(3) {
                        if !is_square_attacked(state, 4, BLACK) 
                           && !is_square_attacked(state, 3, BLACK) 
                           && !is_square_attacked(state, 2, BLACK) { // Check c1 not attacked
                            self.add_move(4, 2, None, false);
                        }
                    }
                }
            } else {
                // Black King Side (e8 -> g8)
                if (state.castling_rights & 4) != 0 {
                    if !occupancy_all.get_bit(61) && !occupancy_all.get_bit(62) {
                        if !is_square_attacked(state, 60, WHITE) 
                           && !is_square_attacked(state, 61, WHITE)
                           && !is_square_attacked(state, 62, WHITE) {
                            self.add_move(60, 62, None, false);
                        }
                    }
                }
                // Black Queen Side (e8 -> c8)
                if (state.castling_rights & 8) != 0 {
                    if !occupancy_all.get_bit(57) && !occupancy_all.get_bit(58) && !occupancy_all.get_bit(59) {
                        if !is_square_attacked(state, 60, WHITE) 
                           && !is_square_attacked(state, 59, WHITE) 
                           && !is_square_attacked(state, 58, WHITE) {
                            self.add_move(60, 58, None, false);
                        }
                    }
                }
            }
        }
    }
}

pub fn is_square_attacked(state: &GameState, square: u8, attacker_side: usize) -> bool {
    if square >= 64 { return false; } 

    if attacker_side == WHITE {
        if square > 8 {
            if (square % 8) > 0 && state.bitboards[P].get_bit(square - 9) { return true; }
            if (square % 8) < 7 && state.bitboards[P].get_bit(square - 7) { return true; }
        }
    } else {
        if square < 56 {
            if (square % 8) > 0 && state.bitboards[p].get_bit(square + 7) { return true; }
            if (square % 8) < 7 && state.bitboards[p].get_bit(square + 9) { return true; }
        }
    }

    let knights = if attacker_side == WHITE { state.bitboards[N] } else { state.bitboards[n] };
    let knight_attacks = get_knight_attacks(square); // Safe Access
    if (knight_attacks & knights).0 != 0 { return true; }

    let king = if attacker_side == WHITE { state.bitboards[K] } else { state.bitboards[k] };
    let king_attacks = get_king_attacks(square); // Safe Access
    if (king_attacks & king).0 != 0 { return true; }

    let occupancy = state.occupancies[BOTH];
    let rooks = if attacker_side == WHITE { state.bitboards[R] | state.bitboards[Q] } 
                else { state.bitboards[r] | state.bitboards[q] };
    let rook_attacks = bitboard::get_rook_attacks(square, occupancy);
    if (rook_attacks & rooks).0 != 0 { return true; }

    let bishops = if attacker_side == WHITE { state.bitboards[B] | state.bitboards[Q] } 
                  else { state.bitboards[b] | state.bitboards[q] };
    let bishop_attacks = bitboard::get_bishop_attacks(square, occupancy);
    if (bishop_attacks & bishops).0 != 0 { return true; }

    false

}

