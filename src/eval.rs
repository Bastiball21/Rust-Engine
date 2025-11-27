use crate::state::{GameState, WHITE, BLACK, P, N, B, R, Q, K, p, n, b, r, q, k, BOTH};
use crate::bitboard::{self, Bitboard}; 

// --- CONFIGURATION ---
const MG_VALS: [i32; 6] = [ 82, 337, 365, 477, 1025, 0 ];
const EG_VALS: [i32; 6] = [ 94, 281, 297, 512,  936, 0 ];
const PHASE_WEIGHTS: [i32; 6] = [0, 1, 1, 2, 4, 0];

const MOBILITY_BONUS: [i32; 4] = [5, 4, 2, 1]; 
const PASSED_PAWN_BONUS: [i32; 8] = [0, 10, 20, 40, 80, 160, 240, 0]; 
const OUTPOST_BONUS: i32 = 30;
const ROOK_OPEN_FILE: i32 = 25;
const ROOK_SEMI_OPEN: i32 = 10;
const BISHOP_PAIR_BONUS: i32 = 30;
const TEMPO_BONUS: i32 = 10;

const ISOLATED_PAWN: i32 = -15;
const DOUBLED_PAWN: i32 = -10;
const CONNECTED_PAWN: i32 = 10; 

#[rustfmt::skip] const MG_PAWN_TABLE: [i32; 64] = [0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 65, 56, 25, -20, -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0];
#[rustfmt::skip] const EG_PAWN_TABLE: [i32; 64] = [0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0];
#[rustfmt::skip] const MG_KNIGHT_TABLE: [i32; 64] = [-167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84, 129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23];
#[rustfmt::skip] const EG_KNIGHT_TABLE: [i32; 64] = [-58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22, -8, -20, 6, 9, 9, 19, -17, -38, -66, -53, -28, -23, -20, -18, -23, -48];
#[rustfmt::skip] const MG_BISHOP_TABLE: [i32; 64] = [-29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50, 37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18, 10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21];
#[rustfmt::skip] const EG_BISHOP_TABLE: [i32; 64] = [-14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15, -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17];
#[rustfmt::skip] const MG_ROOK_TABLE: [i32; 64] = [32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16, -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26];
#[rustfmt::skip] const EG_ROOK_TABLE: [i32; 64] = [13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20];
#[rustfmt::skip] const MG_QUEEN_TABLE: [i32; 64] = [-28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50];
#[rustfmt::skip] const EG_QUEEN_TABLE: [i32; 64] = [-9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9, 3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 39, 23, -16, -27, 15, 6, 9, 17, 10, 5, -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41];
#[rustfmt::skip] const MG_KING_TABLE: [i32; 64] = [-65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14];
#[rustfmt::skip] const EG_KING_TABLE: [i32; 64] = [-74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43];

const SAFETY_TABLE: [i32; 100] = [
    0,  0,  1,  2,  3,  5,  7,  9, 12, 15,
    18, 22, 26, 30, 35, 39, 44, 50, 56, 62,
    68, 75, 82, 85, 89, 97,105,113,122,131,
    140,150,169,180,191,202,213,225,237,249,
    261,274,287,300,313,327,341,355,369,384,
    399,414,429,444,460,476,492,508,524,540,
    556,572,589,605,621,638,655,672,689,706,
    723,740,757,774,791,809,827,845,863,881,
    900,919,938,957,976,995,1014,1033,1052,1071,
    1090,1109,1128,1147,1166,1185,1200,1200,1200,1200
];

fn mirror(sq: usize) -> usize { sq ^ 56 }

fn interpolate(mg: i32, eg: i32, phase: i32) -> i32 {
    let phase_val = phase.clamp(0, 24);
    (mg * phase_val + eg * (24 - phase_val)) / 24
}

pub fn evaluate(state: &GameState) -> i32 {
    let mut mg = 0;
    let mut eg = 0;
    let mut phase = 0;

    let w_pawns = state.bitboards[P];
    let b_pawns = state.bitboards[p];
    let occ = state.occupancies[BOTH];
    let w_occ = state.occupancies[WHITE];
    let b_occ = state.occupancies[BLACK];
    
    let mut w_bishops_count = 0;
    let mut b_bishops_count = 0;

    // --- WHITE PIECES ---
    let mut bb = w_pawns;
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        mg += MG_VALS[0] + MG_PAWN_TABLE[sq]; eg += EG_VALS[0] + EG_PAWN_TABLE[sq];
        
        let file = sq % 8;
        let rank = sq / 8;
        let file_mask = bitboard::file_mask(sq);
        
        if (w_pawns.0 & file_mask.0).count_ones() > 1 { mg += DOUBLED_PAWN; eg += DOUBLED_PAWN; }
        
        let mut isolated = true;
        if file > 0 && (w_pawns.0 & bitboard::file_mask(sq - 1).0) != 0 { isolated = false; }
        if file < 7 && (w_pawns.0 & bitboard::file_mask(sq + 1).0) != 0 { isolated = false; }
        if isolated { mg += ISOLATED_PAWN; eg += ISOLATED_PAWN; }
        
        let mut connected = false;
        if file > 0 && (w_pawns.0 & bitboard::file_mask(sq - 1).0) != 0 { connected = true; } 
        if connected { mg += CONNECTED_PAWN; eg += CONNECTED_PAWN; }

        if (bitboard::passed_pawn_mask(0, sq).0 & b_pawns.0) == 0 {
            let bonus = PASSED_PAWN_BONUS[rank];
            mg += bonus; eg += bonus + (rank as i32 * 20);
        }
    }

    let mut bb = state.bitboards[N];
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        mg += MG_VALS[1] + MG_KNIGHT_TABLE[sq]; eg += EG_VALS[1] + EG_KNIGHT_TABLE[sq];
        phase += PHASE_WEIGHTS[1];
        let attacks = bitboard::mask_knight_attacks(sq as u8);
        let mob = (attacks & !w_occ).count_bits() as i32;
        mg += mob * MOBILITY_BONUS[0]; eg += mob * MOBILITY_BONUS[0];
        
        let rank = sq / 8;
        if rank >= 3 && rank <= 5 {
             let mut protected = false;
             if sq > 8 {
                 if (sq % 8) > 0 && w_pawns.get_bit((sq - 9) as u8) { protected = true; }
                 else if (sq % 8) < 7 && w_pawns.get_bit((sq - 7) as u8) { protected = true; }
             }
             if protected { mg += OUTPOST_BONUS; eg += OUTPOST_BONUS; }
        }
    }

    let mut bb = state.bitboards[B];
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        mg += MG_VALS[2] + MG_BISHOP_TABLE[sq]; eg += EG_VALS[2] + EG_BISHOP_TABLE[sq];
        phase += PHASE_WEIGHTS[2];
        w_bishops_count += 1;
        let attacks = bitboard::get_bishop_attacks(sq as u8, occ);
        let mob = (attacks & !w_occ).count_bits() as i32;
        mg += mob * MOBILITY_BONUS[1]; eg += mob * MOBILITY_BONUS[1];
    }

    let mut bb = state.bitboards[R];
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        mg += MG_VALS[3] + MG_ROOK_TABLE[sq]; eg += EG_VALS[3] + EG_ROOK_TABLE[sq];
        phase += PHASE_WEIGHTS[3];
        let attacks = bitboard::get_rook_attacks(sq as u8, occ);
        let mob = (attacks & !w_occ).count_bits() as i32;
        mg += mob * MOBILITY_BONUS[2]; eg += mob * MOBILITY_BONUS[2];
        let file_mask = bitboard::file_mask(sq);
        if (w_pawns.0 & file_mask.0) == 0 {
            if (b_pawns.0 & file_mask.0) == 0 { mg += ROOK_OPEN_FILE; eg += ROOK_OPEN_FILE; }
            else { mg += ROOK_SEMI_OPEN; eg += ROOK_SEMI_OPEN; }
        }
    }

    let mut bb = state.bitboards[Q];
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        mg += MG_VALS[4] + MG_QUEEN_TABLE[sq]; eg += EG_VALS[4] + EG_QUEEN_TABLE[sq];
        phase += PHASE_WEIGHTS[4];
        let attacks = bitboard::get_queen_attacks(sq as u8, occ);
        let mob = (attacks & !w_occ).count_bits() as i32;
        mg += mob * MOBILITY_BONUS[3]; eg += mob * MOBILITY_BONUS[3];
    }
    
    if w_bishops_count >= 2 { mg += BISHOP_PAIR_BONUS; eg += BISHOP_PAIR_BONUS; }

    let w_k_sq = state.bitboards[K].get_lsb_index() as usize;
    mg += MG_VALS[5] + MG_KING_TABLE[w_k_sq]; eg += EG_VALS[5] + EG_KING_TABLE[w_k_sq];

    // --- BLACK PIECES ---
    let mut bb = b_pawns;
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        let m = mirror(sq);
        mg -= MG_VALS[0] + MG_PAWN_TABLE[m]; eg -= EG_VALS[0] + EG_PAWN_TABLE[m];
        
        let file = sq % 8;
        let file_mask = bitboard::file_mask(sq);
        if (b_pawns.0 & file_mask.0).count_ones() > 1 { mg -= DOUBLED_PAWN; eg -= DOUBLED_PAWN; }
        
        let mut isolated = true;
        if file > 0 && (b_pawns.0 & bitboard::file_mask(sq - 1).0) != 0 { isolated = false; }
        if file < 7 && (b_pawns.0 & bitboard::file_mask(sq + 1).0) != 0 { isolated = false; }
        if isolated { mg -= ISOLATED_PAWN; eg -= ISOLATED_PAWN; }
        
        let mut connected = false;
        if file > 0 && (b_pawns.0 & bitboard::file_mask(sq - 1).0) != 0 { connected = true; }
        if connected { mg -= CONNECTED_PAWN; eg -= CONNECTED_PAWN; }

        let rel_rank = 7 - (sq / 8);
        if (bitboard::passed_pawn_mask(1, sq).0 & w_pawns.0) == 0 {
            let bonus = PASSED_PAWN_BONUS[rel_rank];
            mg -= bonus; eg -= bonus + (rel_rank as i32 * 20);
        }
    }

    let mut bb = state.bitboards[n];
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        let m = mirror(sq);
        mg -= MG_VALS[1] + MG_KNIGHT_TABLE[m]; eg -= EG_VALS[1] + EG_KNIGHT_TABLE[m];
        phase += PHASE_WEIGHTS[1];
        let attacks = bitboard::mask_knight_attacks(sq as u8);
        let mob = (attacks & !b_occ).count_bits() as i32;
        mg -= mob * MOBILITY_BONUS[0]; eg -= mob * MOBILITY_BONUS[0];
        let rank = sq / 8;
        if rank >= 2 && rank <= 4 {
             let mut protected = false;
             if sq < 56 {
                 if (sq % 8 > 0 && b_pawns.get_bit((sq + 7) as u8)) || (sq % 8 < 7 && b_pawns.get_bit((sq + 9) as u8)) { protected = true; }
             }
             if protected { mg -= OUTPOST_BONUS; eg -= OUTPOST_BONUS; }
        }
    }

    let mut bb = state.bitboards[b];
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        let m = mirror(sq);
        mg -= MG_VALS[2] + MG_BISHOP_TABLE[m]; eg -= EG_VALS[2] + EG_BISHOP_TABLE[m];
        phase += PHASE_WEIGHTS[2];
        b_bishops_count += 1;
        let attacks = bitboard::get_bishop_attacks(sq as u8, occ);
        let mob = (attacks & !b_occ).count_bits() as i32;
        mg -= mob * MOBILITY_BONUS[1]; eg -= mob * MOBILITY_BONUS[1];
    }

    let mut bb = state.bitboards[r];
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        let m = mirror(sq);
        mg -= MG_VALS[3] + MG_ROOK_TABLE[m]; eg -= EG_VALS[3] + EG_ROOK_TABLE[m];
        phase += PHASE_WEIGHTS[3];
        let attacks = bitboard::get_rook_attacks(sq as u8, occ);
        let mob = (attacks & !b_occ).count_bits() as i32;
        mg -= mob * MOBILITY_BONUS[2]; eg -= mob * MOBILITY_BONUS[2];
        let file_mask = bitboard::file_mask(sq);
        if (b_pawns.0 & file_mask.0) == 0 {
            if (w_pawns.0 & file_mask.0) == 0 { mg -= ROOK_OPEN_FILE; eg -= ROOK_OPEN_FILE; }
            else { mg -= ROOK_SEMI_OPEN; eg -= ROOK_SEMI_OPEN; }
        }
    }

    let mut bb = state.bitboards[q];
    while bb.0 != 0 {
        let sq = bb.get_lsb_index() as usize; bb.pop_bit(sq as u8);
        let m = mirror(sq);
        mg -= MG_VALS[4] + MG_QUEEN_TABLE[m]; eg -= EG_VALS[4] + EG_QUEEN_TABLE[m];
        phase += PHASE_WEIGHTS[4];
        let attacks = bitboard::get_queen_attacks(sq as u8, occ);
        let mob = (attacks & !b_occ).count_bits() as i32;
        mg -= mob * MOBILITY_BONUS[3]; eg -= mob * MOBILITY_BONUS[3];
    }
    
    if b_bishops_count >= 2 { mg -= BISHOP_PAIR_BONUS; eg -= BISHOP_PAIR_BONUS; }

    let b_k_sq = state.bitboards[k].get_lsb_index() as usize;
    let m_k_sq = mirror(b_k_sq);
    mg -= MG_VALS[5] + MG_KING_TABLE[m_k_sq]; eg -= EG_VALS[5] + EG_KING_TABLE[m_k_sq];

    // King Safety
    if phase > 10 {
        let zone = bitboard::king_zone_mask(w_k_sq);
        let shield = (w_pawns.0 & zone.0).count_ones();
        if shield < 2 { mg -= 15; } 
        if (w_pawns.0 & bitboard::file_mask(w_k_sq).0) == 0 { mg -= 20; } 

        let b_zone = bitboard::king_zone_mask(b_k_sq);
        let shield = (b_pawns.0 & b_zone.0).count_ones();
        if shield < 2 { mg += 15; }
        if (b_pawns.0 & bitboard::file_mask(b_k_sq).0) == 0 { mg += 20; }
    }

    let mut final_score = interpolate(mg, eg, phase);

    // Mop-Up
    if phase < 4 && final_score.abs() > 200 {
         let winning_side = if final_score > 0 { WHITE } else { BLACK };
         let l_k = if winning_side == WHITE { b_k_sq as i32 } else { w_k_sq as i32 };
         let w_k = if winning_side == WHITE { w_k_sq as i32 } else { b_k_sq as i32 };
         
         let l_f = l_k % 8; let l_r = l_k / 8;
         let center_dist = (3 - l_f).abs().max((4 - l_f).abs()) + (3 - l_r).abs().max((4 - l_r).abs());
         let mop = 10 * center_dist;
         
         let dist_kings = ((w_k % 8) - l_f).abs() + ((w_k / 8) - l_r).abs();
         let close = (14 - dist_kings) * 4;
         
         if winning_side == WHITE { final_score += mop + close; } else { final_score -= mop + close; }
    }

    final_score += if state.side_to_move == WHITE { TEMPO_BONUS } else { -TEMPO_BONUS };

    if state.side_to_move == WHITE { final_score } else { -final_score }
}