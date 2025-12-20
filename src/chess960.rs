// src/chess960.rs
use crate::bitboard::Bitboard;
use crate::state::{b, k, n, p, q, r, GameState, B, BLACK, BOTH, K, N, P, Q, R, WHITE};

pub fn generate_chess960_position(index: u16) -> GameState {
    let mut state = GameState::new();
    // Clear the board
    state.bitboards = [Bitboard(0); 12];
    state.occupancies = [Bitboard(0); 3];
    state.castling_rights = 0;

    let mut board = [12usize; 8]; // 12 = empty

    // 1. Place Bishops
    let mut val = index;
    let b1_pos = (val % 4) * 2 + 1;
    val /= 4;
    let b2_pos = (val % 4) * 2;
    val /= 4;

    board[b1_pos as usize] = B;
    board[b2_pos as usize] = B;

    // 2. Place Queen
    let q_pos = val % 6;
    val /= 6;
    let mut empty_count = 0;
    for i in 0..8 {
        if board[i] == 12 {
            if empty_count == q_pos {
                board[i] = Q;
                break;
            }
            empty_count += 1;
        }
    }

    // 3. Place Knights
    // positions of knights among remaining 5 squares (0..9)
    // 5C2 = 10 combinations
    let kn_code = val % 10;
    // table of knights positions
    let kn_table = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ];
    let (k1_idx, k2_idx) = kn_table[kn_code as usize];

    empty_count = 0;
    let mut k1_placed = false;
    for i in 0..8 {
        if board[i] == 12 {
            if empty_count == k1_idx {
                board[i] = N;
                k1_placed = true;
            }
            if k1_placed && empty_count == k2_idx {
                board[i] = N;
                // k2 placed
            }
            empty_count += 1;
        }
    }

    // 4. Place Rooks and King
    // Remaining 3 squares: R K R
    let mut rooks_placed = 0;
    for i in 0..8 {
        if board[i] == 12 {
            if rooks_placed == 0 {
                board[i] = R;
                state.castling_rook_files[WHITE][1] = i as u8; // Left rook -> Queenside (convention)
                state.castling_rook_files[BLACK][1] = i as u8;
                rooks_placed += 1;
            } else if rooks_placed == 1 {
                board[i] = K;
                rooks_placed += 1;
            } else {
                board[i] = R;
                state.castling_rook_files[WHITE][0] = i as u8; // Right rook -> Kingside
                state.castling_rook_files[BLACK][0] = i as u8;
            }
        }
    }

    // Set pieces
    for i in 0..8 {
        let piece = board[i];
        if piece != 12 {
            state.bitboards[piece].set_bit(i as u8); // White backrank
            state.bitboards[piece + 6].set_bit((i + 56) as u8); // Black backrank (mirrored file)
        }
        // Pawns
        state.bitboards[P].set_bit((i + 8) as u8);
        state.bitboards[p].set_bit((i + 48) as u8);
    }

    // Update Occupancies
    for p_idx in P..=K {
        state.occupancies[WHITE] = state.occupancies[WHITE] | state.bitboards[p_idx];
    }
    for p_idx in p..=k {
        state.occupancies[BLACK] = state.occupancies[BLACK] | state.bitboards[p_idx];
    }
    state.occupancies[BOTH] = state.occupancies[WHITE] | state.occupancies[BLACK];

    // Castling Rights
    state.castling_rights = 15; // All rights

    // Update State Info
    state.side_to_move = WHITE;
    state.fullmove_number = 1;
    state.halfmove_clock = 0;
    state.en_passant = 64;

    state.compute_hash();
    state.refresh_accumulator();

    state
}
