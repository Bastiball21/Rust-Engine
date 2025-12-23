#[cfg(debug_assertions)]
use crate::state::{GameState, NO_PIECE, WHITE, BLACK, BOTH};
#[cfg(debug_assertions)]
use crate::bitboard::Bitboard;

#[cfg(debug_assertions)]
pub fn validate_board_consistency(state: &GameState) {
    let mut recomputed_bitboards = [Bitboard(0); 12];
    let mut recomputed_occupancies = [Bitboard(0); 3];

    // Recompute bitboards from the mailbox array
    for square in 0..64 {
        let piece = state.board[square as usize] as usize;
        if piece != NO_PIECE {
            recomputed_bitboards[piece].set_bit(square);

            let color = if piece < 6 { WHITE } else { BLACK };
            recomputed_occupancies[color].set_bit(square);
            recomputed_occupancies[BOTH].set_bit(square);
        }
    }

    // Check 1: Mailbox vs Piece Bitboards
    let mut consistent = true;
    for i in 0..12 {
        if recomputed_bitboards[i] != state.bitboards[i] {
            eprintln!("CRITICAL: Bitboard[{}] mismatch!", i);
            eprintln!("  Stored:     {:016x}", state.bitboards[i].0);
            eprintln!("  Recomputed: {:016x}", recomputed_bitboards[i].0);
            consistent = false;
        }
    }

    // Check 2: Piece Bitboards vs Occupancies
    // Check White Occupancy
    let mut union_white = Bitboard(0);
    for i in 0..6 { union_white = union_white | state.bitboards[i]; }

    if union_white != state.occupancies[WHITE] {
         eprintln!("CRITICAL: Occupancies[WHITE] mismatch!");
         eprintln!("  Union of Pieces: {:016x}", union_white.0);
         eprintln!("  Stored Occ:      {:016x}", state.occupancies[WHITE].0);
         eprintln!("  Diff:            {:016x}", union_white.0 ^ state.occupancies[WHITE].0);
         consistent = false;
    }

    // Check Black Occupancy
    let mut union_black = Bitboard(0);
    for i in 6..12 { union_black = union_black | state.bitboards[i]; }

    if union_black != state.occupancies[BLACK] {
         eprintln!("CRITICAL: Occupancies[BLACK] mismatch!");
         eprintln!("  Union of Pieces: {:016x}", union_black.0);
         eprintln!("  Stored Occ:      {:016x}", state.occupancies[BLACK].0);
         eprintln!("  Diff:            {:016x}", union_black.0 ^ state.occupancies[BLACK].0);
         consistent = false;
    }

    // Check Both Occupancy
    let union_both = union_white | union_black;
    if union_both != state.occupancies[BOTH] {
         eprintln!("CRITICAL: Occupancies[BOTH] mismatch!");
         eprintln!("  Union of All:    {:016x}", union_both.0);
         eprintln!("  Stored Occ:      {:016x}", state.occupancies[BOTH].0);
         eprintln!("  Diff:            {:016x}", union_both.0 ^ state.occupancies[BOTH].0);
         consistent = false;
    }

    // Check 3: Mailbox vs Occupancies (Redundant but explicit)
    if recomputed_occupancies[WHITE] != state.occupancies[WHITE] {
        eprintln!("CRITICAL: Mailbox vs Occupancies[WHITE] mismatch!");
        consistent = false;
    }
    if recomputed_occupancies[BLACK] != state.occupancies[BLACK] {
        eprintln!("CRITICAL: Mailbox vs Occupancies[BLACK] mismatch!");
        consistent = false;
    }
    if recomputed_occupancies[BOTH] != state.occupancies[BOTH] {
        eprintln!("CRITICAL: Mailbox vs Occupancies[BOTH] mismatch!");
        consistent = false;
    }

    if !consistent {
        eprintln!("FEN: {}", state.to_fen());
        eprintln!("Side to move: {}", if state.side_to_move == WHITE { "White" } else { "Black" });
        eprintln!("En Passant: {}", state.en_passant);
        eprintln!("Castling Rights: {}", state.castling_rights);

        // Dump the mailbox for visualization
        eprintln!("Mailbox:");
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                let p = state.board[sq as usize];
                let c = if p == NO_PIECE as u8 {
                    "."
                } else {
                    match p as usize {
                        0 => "P", 1 => "N", 2 => "B", 3 => "R", 4 => "Q", 5 => "K",
                        6 => "p", 7 => "n", 8 => "b", 9 => "r", 10 => "q", 11 => "k",
                        _ => "?",
                    }
                };
                eprint!("{} ", c);
            }
            eprintln!();
        }

        panic!("Board inconsistency detected");
    }
}
