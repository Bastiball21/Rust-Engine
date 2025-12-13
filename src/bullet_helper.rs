
use bulletformat::ChessBoard;
use crate::state::GameState;
use crate::state::WHITE;

pub fn convert_to_bullet(state: &GameState, score: i16, result: f32) -> ChessBoard {
    // 1. Bitboards: [White, Black, Pawn, Knight, Bishop, Rook, Queen, King]
    let mut bbs = [0u64; 8];

    // We need to reconstruct these from the state.
    // State has bitboards[12] -> [P, N, B, R, Q, K, p, n, b, r, q, k]

    // Extract .0 from Bitboard wrappers
    let w = state.bitboards[0].0 | state.bitboards[1].0 | state.bitboards[2].0 | state.bitboards[3].0 | state.bitboards[4].0 | state.bitboards[5].0;
    let b = state.bitboards[6].0 | state.bitboards[7].0 | state.bitboards[8].0 | state.bitboards[9].0 | state.bitboards[10].0 | state.bitboards[11].0;

    bbs[0] = w; // White
    bbs[1] = b; // Black

    // Pieces (Both colors combined)
    bbs[2] = state.bitboards[0].0 | state.bitboards[6].0; // Pawn
    bbs[3] = state.bitboards[1].0 | state.bitboards[7].0; // Knight
    bbs[4] = state.bitboards[2].0 | state.bitboards[8].0; // Bishop
    bbs[5] = state.bitboards[3].0 | state.bitboards[9].0; // Rook
    bbs[6] = state.bitboards[4].0 | state.bitboards[10].0; // Queen
    bbs[7] = state.bitboards[5].0 | state.bitboards[11].0; // King

    let stm = if state.side_to_move == WHITE { 0 } else { 1 };

    // score and result passed to this function are expected to be White Relative/Global for from_raw to handle correctl.

    ChessBoard::from_raw(bbs, stm, score, result).expect("Failed to create ChessBoard")
}
