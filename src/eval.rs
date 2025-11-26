use crate::state::{GameState, WHITE, BLACK, P, N, B, R, Q, K, p, n, b, r, q, k, BOTH};
use crate::bitboard::{self, Bitboard};

// --- TUNABLE PARAMETERS ---
#[derive(Clone, Copy)]
pub struct EvalParams {
    pub material: [i32; 6],         // P, N, B, R, Q, K
    pub pawn_table: [i32; 64],
    pub knight_table: [i32; 64],
    pub bishop_table: [i32; 64],
    pub rook_table: [i32; 64],
    pub queen_table: [i32; 64],
    pub king_table: [i32; 64],
    pub passed_pawn_bonus: [i32; 8],
    pub mobility_bonus: [i32; 4],   // N, B, R, Q
}

// Default values
pub static mut PARAMS: EvalParams = EvalParams {
    material: [100, 320, 330, 500, 900, 20000],
    pawn_table: [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5, 5, 10, 25, 25, 10, 5, 5,
        0, 0, 0, 20, 20, 0, 0, 0,
        5, -5, -10, 0, 0, -10, -5, 5,
        5, 10, 10, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    knight_table: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 20, 20, 15, 5, -30,
        -30, 0, 15, 20, 20, 15, 0, -30,
        -30, 5, 10, 15, 15, 10, 5, -30,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50
    ],
    bishop_table: [0; 64],
    rook_table: [0; 64],
    queen_table: [0; 64],
    king_table: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 20, 0, 0, 0, 0, 20, 20,
        20, 30, 10, 0, 0, 10, 30, 20
    ],
    passed_pawn_bonus: [0, 5, 10, 20, 35, 60, 100, 200],
    mobility_bonus: [4, 3, 2, 1],
};

pub const ISOLATED_PAWN_PENALTY: i32 = -10;
pub const DOUBLED_PAWN_PENALTY: i32 = -10;
pub const OPEN_FILE_PENALTY: i32 = -25; 
pub const ROOK_OPEN_FILE_BONUS: i32 = 25;
pub const ROOK_SEMI_OPEN_FILE_BONUS: i32 = 10;
pub const TRAPPED_PIECE_PENALTY: i32 = -50;

pub fn evaluate(state: &GameState) -> i32 {
    let mut score = 0;
    // SAFE ACCESS FIX: Use addr_of! to avoid creating a reference to static mut directly
    unsafe {
        let params = &*std::ptr::addr_of!(PARAMS);
        score += eval_material(state, params);
        score += eval_pawns(state, params);
        score += eval_king_safety(state, params);
        score += eval_mobility(state, params);
        score += eval_rooks(state);
    }
    if state.side_to_move == WHITE { score } else { -score }
}

fn mirror(sq: usize) -> usize {
    sq ^ 56
}

fn eval_material(state: &GameState, params: &EvalParams) -> i32 {
    let mut score = 0;
    // WHITE
    let mut bb = state.bitboards[P]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score += params.material[0] + params.pawn_table[sq]; }
    let mut bb = state.bitboards[N]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score += params.material[1] + params.knight_table[sq]; }
    let mut bb = state.bitboards[B]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score += params.material[2] + params.bishop_table[sq]; }
    let mut bb = state.bitboards[R]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score += params.material[3] + params.rook_table[sq]; }
    let mut bb = state.bitboards[Q]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score += params.material[4] + params.queen_table[sq]; }
    let w_king = state.bitboards[K]; if w_king.0 != 0 { score += params.material[5] + params.king_table[w_king.get_lsb_index() as usize]; }

    // BLACK
    let mut bb = state.bitboards[p]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score -= params.material[0] + params.pawn_table[mirror(sq)]; }
    let mut bb = state.bitboards[n]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score -= params.material[1] + params.knight_table[mirror(sq)]; }
    let mut bb = state.bitboards[b]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score -= params.material[2] + params.bishop_table[mirror(sq)]; }
    let mut bb = state.bitboards[r]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score -= params.material[3] + params.rook_table[mirror(sq)]; }
    let mut bb = state.bitboards[q]; while bb.0 != 0 { let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8); score -= params.material[4] + params.queen_table[mirror(sq)]; }
    let b_king = state.bitboards[k]; if b_king.0 != 0 { score -= params.material[5] + params.king_table[mirror(b_king.get_lsb_index() as usize)]; }
    
    score
}

fn eval_pawns(state: &GameState, params: &EvalParams) -> i32 {
    let mut score = 0;
    let w_pawns = state.bitboards[P];
    let b_pawns = state.bitboards[p];

    let mut bb = w_pawns;
    while bb.0 != 0 {
        let sq = bb.get_lsb_index(); bb.pop_bit(sq as u8);
        let rank = sq / 8;
        if (bitboard::passed_pawn_mask(0, sq as usize).0 & b_pawns.0) == 0 { score += params.passed_pawn_bonus[rank as usize]; }
        if (bitboard::file_mask(sq as usize).0 & w_pawns.0).count_ones() > 1 { score += DOUBLED_PAWN_PENALTY; }
    }
    let mut bb = b_pawns;
    while bb.0 != 0 {
        let sq = bb.get_lsb_index(); bb.pop_bit(sq as u8);
        let rel_rank = 7 - (sq / 8);
        if (bitboard::passed_pawn_mask(1, sq as usize).0 & w_pawns.0) == 0 { score -= params.passed_pawn_bonus[rel_rank as usize]; }
        if (bitboard::file_mask(sq as usize).0 & b_pawns.0).count_ones() > 1 { score -= DOUBLED_PAWN_PENALTY; }
    }
    score
}

fn eval_mobility(state: &GameState, params: &EvalParams) -> i32 {
    let mut score = 0;
    let occupancy = state.occupancies[BOTH];
    let w_occ = state.occupancies[WHITE];
    let b_occ = state.occupancies[BLACK];

    let mut knights = state.bitboards[N];
    while knights.0 != 0 {
        let sq = knights.get_lsb_index() as u8; knights.pop_bit(sq);
        let attacks = bitboard::mask_knight_attacks(sq) & !w_occ;
        let count = attacks.count_bits() as i32;
        score += count * params.mobility_bonus[0];
    }
    let mut bishops = state.bitboards[B];
    while bishops.0 != 0 {
        let sq = bishops.get_lsb_index() as u8; bishops.pop_bit(sq);
        let attacks = bitboard::get_bishop_attacks(sq, occupancy) & !w_occ;
        let count = attacks.count_bits() as i32;
        score += count * params.mobility_bonus[1];
    }
    let mut rooks = state.bitboards[R];
    while rooks.0 != 0 {
        let sq = rooks.get_lsb_index() as u8; rooks.pop_bit(sq);
        let attacks = bitboard::get_rook_attacks(sq, occupancy) & !w_occ;
        score += (attacks.count_bits() as i32) * params.mobility_bonus[2];
    }
    let mut queens = state.bitboards[Q];
    while queens.0 != 0 {
        let sq = queens.get_lsb_index() as u8; queens.pop_bit(sq);
        let attacks = bitboard::get_queen_attacks(sq, occupancy) & !w_occ;
        score += (attacks.count_bits() as i32) * params.mobility_bonus[3];
    }

    let mut knights = state.bitboards[n];
    while knights.0 != 0 {
        let sq = knights.get_lsb_index() as u8; knights.pop_bit(sq);
        let attacks = bitboard::mask_knight_attacks(sq) & !b_occ;
        score -= (attacks.count_bits() as i32) * params.mobility_bonus[0];
    }
    let mut bishops = state.bitboards[b];
    while bishops.0 != 0 {
        let sq = bishops.get_lsb_index() as u8; bishops.pop_bit(sq);
        let attacks = bitboard::get_bishop_attacks(sq, occupancy) & !b_occ;
        score -= (attacks.count_bits() as i32) * params.mobility_bonus[1];
    }
    let mut rooks = state.bitboards[r];
    while rooks.0 != 0 {
        let sq = rooks.get_lsb_index() as u8; rooks.pop_bit(sq);
        let attacks = bitboard::get_rook_attacks(sq, occupancy) & !b_occ;
        score -= (attacks.count_bits() as i32) * params.mobility_bonus[2];
    }
    let mut queens = state.bitboards[q];
    while queens.0 != 0 {
        let sq = queens.get_lsb_index() as u8; queens.pop_bit(sq);
        let attacks = bitboard::get_queen_attacks(sq, occupancy) & !b_occ;
        score -= (attacks.count_bits() as i32) * params.mobility_bonus[3];
    }
    score
}

fn eval_king_safety(state: &GameState, params: &EvalParams) -> i32 {
    let mut score = 0;
    let w_king = state.bitboards[K];
    if w_king.0 != 0 {
        let sq = w_king.get_lsb_index();
        let zone = bitboard::king_zone_mask(sq as usize);
        let w_pawns = state.bitboards[P];
        let shield_count = (w_pawns.0 & zone.0).count_ones();
        if shield_count < 2 { score -= 15; } 
        let file_mask = bitboard::file_mask(sq as usize);
        if (w_pawns.0 & file_mask.0) == 0 { score += OPEN_FILE_PENALTY; }
    }
    let b_king = state.bitboards[k];
    if b_king.0 != 0 {
        let sq = b_king.get_lsb_index();
        let zone = bitboard::king_zone_mask(sq as usize);
        let b_pawns = state.bitboards[p];
        let shield_count = (b_pawns.0 & zone.0).count_ones();
        if shield_count < 2 { score += 15; }
        let file_mask = bitboard::file_mask(sq as usize);
        if (b_pawns.0 & file_mask.0) == 0 { score -= OPEN_FILE_PENALTY; }
    }
    score
}

fn eval_rooks(state: &GameState) -> i32 {
    let mut score = 0;
    let w_pawns = state.bitboards[P];
    let b_pawns = state.bitboards[p];

    let mut w_rooks = state.bitboards[R];
    while w_rooks.0 != 0 {
        let sq = w_rooks.get_lsb_index(); w_rooks.pop_bit(sq as u8);
        let file_mask = bitboard::file_mask(sq as usize);
        if (w_pawns.0 & file_mask.0) == 0 {
            if (b_pawns.0 & file_mask.0) == 0 { score += ROOK_OPEN_FILE_BONUS; } 
            else { score += ROOK_SEMI_OPEN_FILE_BONUS; }
        }
    }
    let mut b_rooks = state.bitboards[r];
    while b_rooks.0 != 0 {
        let sq = b_rooks.get_lsb_index(); b_rooks.pop_bit(sq as u8);
        let file_mask = bitboard::file_mask(sq as usize);
        if (b_pawns.0 & file_mask.0) == 0 {
            if (w_pawns.0 & file_mask.0) == 0 { score -= ROOK_OPEN_FILE_BONUS; } 
            else { score -= ROOK_SEMI_OPEN_FILE_BONUS; }
        }
    }
    score
}