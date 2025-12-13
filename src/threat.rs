// src/threat.rs
use crate::bitboard::{self, Bitboard, FILE_A, FILE_H};
use crate::eval;
use crate::movegen;
use crate::state::{b, k, n, p, q, r, GameState, B, BLACK, BOTH, K, N, P, Q, R, WHITE};
use std::sync::OnceLock;

// --- CONSTANTS & THRESHOLDS ---
pub const THREAT_EXTENSION_THRESHOLD: i32 = 120;
pub const THREAT_KING_DANGER_THRESHOLD: i32 = 150;
pub const THREAT_INSTABILITY_THRESHOLD: i32 = 200;

#[derive(Clone, Copy, Default)]
pub struct ThreatInfo {
    pub attacks_by_side: [Bitboard; 2],
    pub king_ring_3x3: [Bitboard; 2],
    pub king_ring_5x5: [Bitboard; 2],

    // Scores
    pub static_threat_score: i32,
    pub king_danger_score: [i32; 2], // Index by side whose king is in danger
    pub forcing_eval_drop: i32,
    pub tactical_instability: bool,

    // Specific Features
    pub hanging_piece_value: i32,
    pub undefended_count: [i32; 2],
    pub pinned_pieces: [Bitboard; 2],
    pub check_proximity: i32,
}

// --- INITIALIZATION ---
static KING_RING_MASKS: OnceLock<[[Bitboard; 2]; 64]> = OnceLock::new();

pub fn init_threat() {
    KING_RING_MASKS.get_or_init(|| {
        let mut table = [[Bitboard(0); 2]; 64];
        for sq in 0..64 {
            let b_sq = Bitboard(1 << sq);
            // 3x3 Ring (King moves)
            table[sq][0] = movegen::get_king_attacks(sq as u8);

            // 5x5 Ring (approximate by expanding 3x3 or manual)
            // A simple way: King attacks of the King attacks
            let mut ring5 = table[sq][0];
            let mut iter = table[sq][0];
            while iter.0 != 0 {
                let s = iter.get_lsb_index() as u8;
                iter.pop_bit(s);
                ring5 = ring5 | movegen::get_king_attacks(s);
            }
            // Exclude the king square itself
            ring5 = ring5 & !b_sq;
            table[sq][1] = ring5;
        }
        table
    });
}

// --- MAIN API ---

pub fn analyze(state: &GameState) -> ThreatInfo {
    let mut info = ThreatInfo::default();

    // 1. Generate Attacks & Rings
    compute_attacks_and_rings(state, &mut info);

    // 2. Static Threat Analysis
    compute_static_threats(state, &mut info);

    // 3. Conditional Expensive Checks (1-ply Probe)
    // Only run if static threat indicates danger or instability
    if info.static_threat_score > 50 || info.king_danger_score[state.side_to_move] > 30 {
        compute_forcing_threats(state, &mut info);
    }

    info
}

fn compute_attacks_and_rings(state: &GameState, info: &mut ThreatInfo) {
    let occ = state.occupancies[BOTH];

    for side in [WHITE, BLACK] {
        let mut attacks = Bitboard(0);

        // Pawns
        let pawn_type = if side == WHITE { P } else { p };
        let pawns = state.bitboards[pawn_type];
        // Efficient pawn attacks
        if side == WHITE {
            attacks = attacks | bitboard::pawn_attacks(pawns, WHITE);
        } else {
            attacks = attacks | bitboard::pawn_attacks(pawns, BLACK);
        }

        // Knights
        let knight_type = if side == WHITE { N } else { n };
        let mut knights = state.bitboards[knight_type];
        while knights.0 != 0 {
            let sq = knights.get_lsb_index() as u8;
            knights.pop_bit(sq);
            attacks = attacks | movegen::get_knight_attacks(sq);
        }

        // King
        let king_type = if side == WHITE { K } else { k };
        let king_bb = state.bitboards[king_type];
        let king_sq = if king_bb.0 != 0 {
            king_bb.get_lsb_index() as u8
        } else {
            0 // Fallback if no king (shouldn't happen in valid FEN)
        };
        attacks = attacks | movegen::get_king_attacks(king_sq);

        // Sliders (B/R/Q)
        let bishop_type = if side == WHITE { B } else { b };
        let mut bishops = state.bitboards[bishop_type] | state.bitboards[if side == WHITE { Q } else { q }];
        while bishops.0 != 0 {
            let sq = bishops.get_lsb_index() as u8;
            bishops.pop_bit(sq);
            attacks = attacks | bitboard::get_bishop_attacks(sq, occ);
        }

        let rook_type = if side == WHITE { R } else { r };
        let mut rooks = state.bitboards[rook_type] | state.bitboards[if side == WHITE { Q } else { q }];
        while rooks.0 != 0 {
            let sq = rooks.get_lsb_index() as u8;
            rooks.pop_bit(sq);
            attacks = attacks | bitboard::get_rook_attacks(sq, occ);
        }

        info.attacks_by_side[side] = attacks;

        // Rings
        let masks = KING_RING_MASKS.get().expect("Threat masks not init")[king_sq as usize];
        info.king_ring_3x3[side] = masks[0];
        info.king_ring_5x5[side] = masks[1];
    }
}

fn compute_static_threats(state: &GameState, info: &mut ThreatInfo) {
    let us = state.side_to_move;
    let them = 1 - us;

    let us_occ = state.occupancies[us];
    let them_occ = state.occupancies[them];

    // --- Hanging Pieces / En Prise ---
    // A piece is hanging if:
    // 1. Attacked by enemy, not defended by us.
    // 2. Attacked by enemy pawn/knight, and we are > pawn/knight (trade loss).
    // 3. Attackers > Defenders.

    let mut hanging_val = 0;

    // We only care about OUR hanging pieces for defense urgency, or ENEMY hanging pieces for attack urgency.
    // The "Threat Score" usually represents "How dangerous is the position for the side to move?"
    // OR "How much threat is the side to move generating?".
    // Let's define:
    // threat_score = (Threats against Enemy) - (Threats against Us) ?
    // No, standard convention: Threat Score = How much danger WE are in, OR how unstable the position is.
    // User definition: "Threat delta: eval(after opponent best forcing move) – eval(current)"
    // And "Capture urgency (is something en prise next move?)"

    // Let's track threats AGAINST THE SIDE TO MOVE first (Urgency).

    let our_attacked_pieces = us_occ & info.attacks_by_side[them];
    let mut iter = our_attacked_pieces;
    while iter.0 != 0 {
        let sq = iter.get_lsb_index() as u8;
        iter.pop_bit(sq);

        let piece_val = get_piece_value(state, sq);
        let defended = info.attacks_by_side[us].get_bit(sq);

        if !defended {
            hanging_val += piece_val;
        } else {
            // Defended, but maybe attacked by lower value?
            // This requires checking WHAT is attacking. Expensive loop?
            // "Layer 1: Static Threat Delta... Max en-prise piece value"
            // Approximation: If attacked by Pawn and we are not Pawn -> Hanging.

            let pawn_attacks = bitboard::pawn_attacks(Bitboard(1<<sq), us); // Squares that attack 'sq' as pawn
            if (pawn_attacks & state.bitboards[if them == WHITE { P } else { p }]).0 != 0 {
                if piece_val > 100 { // > Pawn
                     hanging_val += piece_val - 100;
                }
            }
        }
    }

    info.hanging_piece_value = hanging_val;
    if hanging_val > 0 {
        info.static_threat_score += hanging_val; // Danger!
        info.tactical_instability = true;
    }

    // --- King Danger ---
    for side in [WHITE, BLACK] {
        let enemy = 1 - side;
        let ring_3 = info.king_ring_3x3[side];
        let ring_5 = info.king_ring_5x5[side];

        let mut score = 0;

        // 1. Count Attackers Attacking the Ring (Correctly!)
        // Iterate enemy pieces and check their specific attacks against ring_3

        let occ = state.occupancies[BOTH];
        let enemy_pawns = state.bitboards[if enemy == WHITE { P } else { p }];
        let enemy_knights = state.bitboards[if enemy == WHITE { N } else { n }];
        let enemy_bishops = state.bitboards[if enemy == WHITE { B } else { b }] | state.bitboards[if enemy == WHITE { Q } else { q }];
        let enemy_rooks = state.bitboards[if enemy == WHITE { R } else { r }] | state.bitboards[if enemy == WHITE { Q } else { q }];
        let enemy_queens = state.bitboards[if enemy == WHITE { Q } else { q }];

        let mut attacker_count = 0;
        let mut attacker_weight = 0;

        // Pawns
        // Optimization: Pawn attacks are static relative to square, can check if ring intersects pawn attacks
        // OR: just use precomputed attacks from compute_attacks_and_rings?
        // info.attacks_by_side includes pawns. But we want "count of pieces".
        // A single pawn attacking 2 ring squares counts as 1 attacker.
        // It's cheaper to iterate pawns if count is low, or reverse check:
        // iterate ring squares, see if attacked by pawn? No, that's "Squares Attacked".
        // Correct: Iterate pieces.

        let mut pawns = enemy_pawns;
        if (bitboard::pawn_attacks(pawns, enemy) & ring_3).0 != 0 {
             // If any pawn attacks the ring, we need to know HOW MANY.
             // This is slightly tricky with bitboards without iteration.
             // Approximation: Just use the fact that pawns attack.
             // Let's iterate pawns that are close (rank distance).
             while pawns.0 != 0 {
                 let sq = pawns.get_lsb_index() as u8;
                 pawns.pop_bit(sq);
                 if (bitboard::pawn_attacks(Bitboard(1<<sq), enemy) & ring_3).0 != 0 {
                     attacker_count += 1;
                     attacker_weight += 10; // Low weight for pawns usually, but persistent. User said 15 for "Shield hit".
                 }
             }
        }

        // Knights
        let mut knights = enemy_knights;
        while knights.0 != 0 {
            let sq = knights.get_lsb_index() as u8;
            knights.pop_bit(sq);
            if (movegen::get_knight_attacks(sq) & ring_3).0 != 0 {
                attacker_count += 1;
                attacker_weight += 20;
            }
        }

        // Bishops (and Queens diagonal)
        // Note: Queens are in both 'enemy_bishops' and 'enemy_rooks' variable above for convenience of slider generation?
        // No, `state.bitboards[B]` contains Bishops. `state.bitboards[Q]` contains Queens.
        // I combined them above in variables.
        // BUT: If I count Q as Bishop and Rook, I double count it?
        // Better: Iterate Bishops, Rooks, Queens separately.

        let mut bishops = state.bitboards[if enemy == WHITE { B } else { b }];
        while bishops.0 != 0 {
            let sq = bishops.get_lsb_index() as u8;
            bishops.pop_bit(sq);
            if (bitboard::get_bishop_attacks(sq, occ) & ring_3).0 != 0 {
                attacker_count += 1;
                attacker_weight += 20;
            }
        }

        let mut rooks = state.bitboards[if enemy == WHITE { R } else { r }];
        while rooks.0 != 0 {
            let sq = rooks.get_lsb_index() as u8;
            rooks.pop_bit(sq);
            if (bitboard::get_rook_attacks(sq, occ) & ring_3).0 != 0 {
                attacker_count += 1;
                attacker_weight += 40;
            }
        }

        let mut queens = enemy_queens;
        while queens.0 != 0 {
            let sq = queens.get_lsb_index() as u8;
            queens.pop_bit(sq);
            if (bitboard::get_queen_attacks(sq, occ) & ring_3).0 != 0 {
                attacker_count += 1;
                attacker_weight += 60; // Big threat
            }
        }

        score += attacker_weight;

        // Bonus for multiple attackers (swarm)
        if attacker_count >= 2 {
            score += attacker_count * 10;
        }

        // 3. Undefended squares in Ring (King has nowhere to go)
        let undefended_ring = ring_3 & !info.attacks_by_side[side];
        let attacked_ring = ring_3 & info.attacks_by_side[enemy];
        let danger_zone = undefended_ring & attacked_ring; // King can't step here

        // Weight squares in danger zone
        score += (danger_zone.count_bits() as i32) * 10;

        info.king_danger_score[side] = score;
    }

    // If OUR king is in danger, increase threat score
    if info.king_danger_score[us] > 50 {
        info.static_threat_score += info.king_danger_score[us];
        info.tactical_instability = true;
    }
}

fn compute_forcing_threats(state: &GameState, info: &mut ThreatInfo) {
    // 1-Ply Forcing Probe (Layer 2)
    // "What if opponent actually plays the scary move?"
    // We check opponent's forcing moves (Checks, Captures)

    // We need to generate moves for the ENEMY (who is threatening us)
    // But GameState doesn't support "generate moves for enemy" easily without flipping side.
    // So we flip side conceptually.

    // Actually, `ThreatInfo` is calculated for `state.side_to_move`.
    // We want to know if `state.side_to_move` is in danger from `opponent`.
    // So we need to generate `opponent` moves.
    // The engine's move generator generates moves for `state.side_to_move`.
    // So we must make a null move or just inspect the board?
    // MoveGenerator generates for `side_to_move`.
    // We can swap side_to_move in a clone?

    let mut enemy_state = state.clone();
    enemy_state.side_to_move = 1 - state.side_to_move;
    enemy_state.hash = crate::zobrist::side_key() ^ state.hash; // Hacky hash update? Better to ignore hash here.
    // We don't need hash for move generation.

    let mut generator = movegen::MoveGenerator::new();
    generator.generate_moves(&enemy_state);

    // We need a baseline eval for US.
    // Note: evaluate() returns score from side_to_move perspective.
    // state.side_to_move is US.
    // evaluate(state) -> Score for US.

    // We can't re-enter full `evaluate` recursively easily without infinite recursion if we are not careful.
    // But `evaluate` calls `analyze`. `analyze` calls `compute_forcing`.
    // We MUST use `evaluate_hce` directly or a lighter eval, AND pass a dummy ThreatInfo to avoid recursion.
    // Or just use static material/pst.

    let current_eval = eval::evaluate_hce(state, &ThreatInfo::default());

    let mut max_drop = 0;

    for i in 0..generator.list.count {
        let mv = generator.list.moves[i];

        // Filter for Forcing Moves
        let is_capture = mv.is_capture;
        let is_check = movegen::gives_check(&enemy_state, mv);

        if !is_capture && !is_check {
            continue;
        }

        // Make move
        let next_state = enemy_state.make_move(mv); // Now side to move is US again

        // Check if move was legal (king not captured/left in check)
        // enemy_state is opponent. next_state is US to move.
        // We need to check if opponent left THEIR king in check.
        // `make_move` doesn't check legality.
        let enemy_king = if enemy_state.side_to_move == WHITE { K } else { k };
        let k_sq = next_state.bitboards[enemy_king].get_lsb_index() as u8;
        if movegen::is_square_attacked(&next_state, k_sq, next_state.side_to_move) {
             continue; // Illegal move by enemy
        }

        // Now evaluate resulting position for US.
        // next_state.side_to_move is US.
        // evaluate_hce returns score for US.
        let next_eval = eval::evaluate_hce(&next_state, &ThreatInfo::default());

        // Drop = Current - Next
        // If opponent makes a good move, Next should be lower for US.
        let drop = current_eval - next_eval;

        if drop > max_drop {
            max_drop = drop;
        }

        // Optimization: If drop is huge (mate/queen loss), stop early?
        if max_drop > 300 {
            break;
        }
    }

    info.forcing_eval_drop = max_drop;

    if max_drop > 150 {
        info.static_threat_score += max_drop / 2; // Add to threat score
        info.tactical_instability = true;
    }
}

fn get_piece_value(state: &GameState, sq: u8) -> i32 {
    // Simple lookup
    for piece in 0..12 {
        if state.bitboards[piece].get_bit(sq) {
             return match piece % 6 {
                 0 => 100, // Pawn
                 1 => 320, // Knight
                 2 => 330, // Bishop
                 3 => 500, // Rook
                 4 => 900, // Queen
                 5 => 20000, // King
                 _ => 0
             };
        }
    }
    0
}
