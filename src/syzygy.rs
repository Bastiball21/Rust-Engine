use crate::state::GameState;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use pyrrhic_rs::{TableBases, EngineAdapter};
use crate::bitboard;

#[derive(Clone)]
pub struct AetherAdapter;

impl EngineAdapter for AetherAdapter {
    fn pawn_attacks(side: pyrrhic_rs::Color, pawns: u64) -> u64 {
        let s = match side {
            pyrrhic_rs::Color::White => crate::state::WHITE,
            pyrrhic_rs::Color::Black => crate::state::BLACK,
        };
        bitboard::pawn_attacks(bitboard::Bitboard(pawns), s).0
    }
    fn knight_attacks(squares: u64) -> u64 {
        let mut attacks = 0;
        let mut bb = squares;
        while bb != 0 {
             let sq = bb.trailing_zeros();
             bb &= !(1u64 << sq);
             attacks |= bitboard::mask_knight_attacks(sq as u8).0;
        }
        attacks
    }
    fn bishop_attacks(squares: u64, occ: u64) -> u64 {
        let mut attacks = 0;
        let mut bb = squares;
        while bb != 0 {
             let sq = bb.trailing_zeros();
             bb &= !(1u64 << sq);
             attacks |= bitboard::get_bishop_attacks(sq as u8, bitboard::Bitboard(occ)).0;
        }
        attacks
    }
    fn rook_attacks(squares: u64, occ: u64) -> u64 {
        let mut attacks = 0;
        let mut bb = squares;
        while bb != 0 {
             let sq = bb.trailing_zeros();
             bb &= !(1u64 << sq);
             attacks |= bitboard::get_rook_attacks(sq as u8, bitboard::Bitboard(occ)).0;
        }
        attacks
    }
    fn queen_attacks(squares: u64, occ: u64) -> u64 {
        let mut attacks = 0;
        let mut bb = squares;
        while bb != 0 {
             let sq = bb.trailing_zeros();
             bb &= !(1u64 << sq);
             attacks |= bitboard::get_queen_attacks(sq as u8, bitboard::Bitboard(occ)).0;
        }
        attacks
    }
    fn king_attacks(squares: u64) -> u64 {
        let mut attacks = 0;
        let mut bb = squares;
        while bb != 0 {
             let sq = bb.trailing_zeros();
             bb &= !(1u64 << sq);
             attacks |= bitboard::mask_king_attacks(sq as u8).0;
        }
        attacks
    }
}

pub static TABLEBASE: Mutex<Option<TableBases<AetherAdapter>>> = Mutex::new(None);
pub static TB_ENABLED: AtomicBool = AtomicBool::new(false);

const MATE_SCORE: i32 = 30000;
const TB_WIN_SCORE: i32 = MATE_SCORE - 1000;

pub fn init_tablebase(path: &str) {
    if path.is_empty() { return; }
    match TableBases::<AetherAdapter>::new(path) {
        Ok(tb) => {
             let mut lock = TABLEBASE.lock().unwrap();
             *lock = Some(tb);
             TB_ENABLED.store(true, Ordering::SeqCst);
             println!("info string Syzygy Tablebases found.");
        }
        Err(e) => println!("info string Syzygy Init Error: {:?}", e),
    }
}

pub fn probe_wdl(state: &GameState) -> Option<i32> {
    if !TB_ENABLED.load(Ordering::Relaxed) { return None; }
    if state.castling_rights != 0 { return None; }

    let white = state.occupancies[crate::state::WHITE].0;
    let black = state.occupancies[crate::state::BLACK].0;

    use crate::state::{P, N, B, R, Q, K, p, n, b, r, q, k};
    let kings = state.bitboards[K].0 | state.bitboards[k].0;
    let queens = state.bitboards[Q].0 | state.bitboards[q].0;
    let rooks = state.bitboards[R].0 | state.bitboards[r].0;
    let bishops = state.bitboards[B].0 | state.bitboards[b].0;
    let knights = state.bitboards[N].0 | state.bitboards[n].0;
    let pawns = state.bitboards[P].0 | state.bitboards[p].0;

    let rule50 = state.halfmove_clock as u32;
    let turn = state.side_to_move == crate::state::WHITE;

    let lock = TABLEBASE.lock().unwrap();
    if let Some(tb) = lock.as_ref() {
        if let Ok(wdl) = tb.probe_wdl(white, black, kings, queens, rooks, bishops, knights, pawns, rule50, turn) {
             use pyrrhic_rs::WdlProbeResult;
             // Map WDL to engine scores
             return Some(match wdl {
                 WdlProbeResult::Win => TB_WIN_SCORE,
                 WdlProbeResult::Loss => -TB_WIN_SCORE,
                 WdlProbeResult::Draw | WdlProbeResult::BlessedLoss | WdlProbeResult::CursedWin => 0,
             });
        }
    }
    None
}

pub fn probe_root(state: &GameState) -> Option<(crate::state::Move, i32)> {
    if !TB_ENABLED.load(Ordering::Relaxed) { return None; }
    if state.castling_rights != 0 { return None; }

    let white = state.occupancies[crate::state::WHITE].0;
    let black = state.occupancies[crate::state::BLACK].0;

    use crate::state::{P, N, B, R, Q, K, p, n, b, r, q, k};
    let kings = state.bitboards[K].0 | state.bitboards[k].0;
    let queens = state.bitboards[Q].0 | state.bitboards[q].0;
    let rooks = state.bitboards[R].0 | state.bitboards[r].0;
    let bishops = state.bitboards[B].0 | state.bitboards[b].0;
    let knights = state.bitboards[N].0 | state.bitboards[n].0;
    let pawns = state.bitboards[P].0 | state.bitboards[p].0;

    let rule50 = state.halfmove_clock as u32;
    let ep = if state.en_passant != 64 { state.en_passant as u32 } else { 0 };
    let turn = state.side_to_move == crate::state::WHITE;

    let lock = TABLEBASE.lock().unwrap();
    if let Some(tb) = lock.as_ref() {
        if let Ok(res) = tb.probe_root(white, black, kings, queens, rooks, bishops, knights, pawns, rule50, ep, turn) {
             use pyrrhic_rs::{DtzProbeValue, Piece};
             match res.root {
                 DtzProbeValue::Checkmate => {
                     return Some((crate::state::Move::default(), MATE_SCORE));
                 },
                 DtzProbeValue::Stalemate => {
                     return Some((crate::state::Move::default(), 0));
                 },
                 DtzProbeValue::Failed => return None,
                 DtzProbeValue::DtzResult(root_dtz) => {
                      // Map dtz result to Move
                      let mut mv = crate::state::Move {
                          source: root_dtz.from_square,
                          target: root_dtz.to_square,
                          promotion: match root_dtz.promotion {
                              Piece::Queen => Some(4), // Aether Queen = 4
                              Piece::Rook => Some(3),  // Aether Rook = 3
                              Piece::Bishop => Some(2), // Aether Bishop = 2
                              Piece::Knight => Some(1), // Aether Knight = 1
                              _ => None,
                          },
                          is_capture: false // Caller needs to verify or we can deduce?
                          // Actually Move struct in Aether has capture flag.
                          // But for root move reporting, Search usually re-generates or verifies.
                          // However, we should try to be accurate.
                          // But we don't have easy access to capture info here without bitboards.
                          // Aether's make_move uses is_capture to handle piece removal.
                          // We can check if target square is occupied.
                      };

                      // Check capture
                      let target_bit = 1u64 << mv.target;
                      if (state.occupancies[crate::state::BOTH].0 & target_bit) != 0 {
                          mv.is_capture = true;
                      } else if state.en_passant == mv.target && (get_piece_type_safe(state, mv.source) == 0 || get_piece_type_safe(state, mv.source) == 6) {
                          // En Passant capture logic would be here, but simpler:
                          // If it's EP, pyrrhic sets ep flag in DtzResult?
                          if root_dtz.ep {
                              mv.is_capture = true;
                          }
                      }

                      // Calculate score from DTZ
                      // root_dtz.dtz is distance to zeroing (ply count).
                      // Need to map to Mate Score if winning.
                      // Syzygy WDL:
                      // Win: +ve score
                      // Loss: -ve score

                      // Using WDL from result:
                      use pyrrhic_rs::WdlProbeResult;
                      let score = match root_dtz.wdl {
                          WdlProbeResult::Win => TB_WIN_SCORE, // Distance not incorporated in score here?
                          WdlProbeResult::Loss => -TB_WIN_SCORE,
                          _ => 0,
                      };

                      return Some((mv, score));
                 }
             }
        }
    }
    None
}

// Helper needed for is_capture check if we want to be pedantic,
// but actually `probe_root` returns `Move`, and search prints it.
// `format_move_uci` only cares about source/target/promotion.
// `make_move` might need `is_capture` to be correct?
// Let's check `make_move` in `state.rs`.

fn get_piece_type_safe(state: &GameState, square: u8) -> usize {
    if state.occupancies[crate::state::WHITE].0 & (1u64 << square) != 0 {
        for piece in 0..6 {
            if state.bitboards[piece].0 & (1u64 << square) != 0 {
                return piece;
            }
        }
    } else if state.occupancies[crate::state::BLACK].0 & (1u64 << square) != 0 {
        for piece in 6..12 {
            if state.bitboards[piece].0 & (1u64 << square) != 0 {
                return piece;
            }
        }
    }
    12
}
