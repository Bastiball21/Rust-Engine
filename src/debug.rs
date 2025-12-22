#[cfg(debug_assertions)]
use crate::state::{GameState, NO_PIECE};
#[cfg(debug_assertions)]
use crate::bitboard::Bitboard;

#[cfg(debug_assertions)]
pub fn validate_board_consistency(state: &GameState) {
    let mut recomputed = [Bitboard(0); 12];

    // Recompute bitboards from the mailbox array
    for square in 0..64 {
        let piece = state.board[square as usize] as usize;
        if piece != NO_PIECE {
            recomputed[piece].set_bit(square);
        }
    }

    // Compare with stored bitboards
    let mut consistent = true;
    for i in 0..12 {
        if recomputed[i] != state.bitboards[i] {
            consistent = false;
        }
    }

    if !consistent {
        eprintln!("CRITICAL: Bitboard/Mailbox inconsistency detected!");
        eprintln!("FEN: {}", state.to_fen());
        eprintln!("Detailed mismatch:");
        for i in 0..12 {
            if recomputed[i] != state.bitboards[i] {
                eprintln!("Piece {}: stored {:x?}, recomputed {:x?}", i, state.bitboards[i], recomputed[i]);
            }
        }

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
