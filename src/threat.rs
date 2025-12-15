// src/threat.rs
use crate::bitboard::{self, Bitboard, FILE_A, FILE_H};
use crate::eval;
use crate::movegen;
use crate::state::{b, k, n, p, q, r, GameState, Move, B, BLACK, BOTH, K, N, P, Q, R, WHITE};
use std::sync::OnceLock;

// --- CONSTANTS & THRESHOLDS ---
pub const THREAT_EXTENSION_THRESHOLD: i32 = 120;
pub const THREAT_KING_DANGER_THRESHOLD: i32 = 150;
pub const THREAT_INSTABILITY_THRESHOLD: i32 = 200;
pub const SACRIFICE_THREAT_THRESHOLD: i32 = 50;

#[derive(Clone, Copy, Default, Debug)]
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

    // New Features
    pub clustering_score: [i32; 2],   // Feature #5: Multiple attackers on key squares
    pub coordination_score: [i32; 2], // Feature #2: Batteries and connected pieces
}

#[derive(Clone, Copy, Default, Debug)]
pub struct ThreatDeltaScore {
    pub threat_score: i32,     // Raw "threat created" score
    pub dominance_bonus: i32,  // Dominant square bonus
    pub defensive_bonus: i32,  // Defensive value (evasion)
    pub is_tactical: bool,     // Whether this move is "Tactical" enough to affect search

    // New
    pub prophylaxis_score: i32, // Feature #1
    pub pawn_lever_score: i32,  // Feature #3
    pub coordination_bonus: i32,// Feature #2 (Dynamic)
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

    // 3. Coordination & Clustering (New Features)
    compute_coordination(state, &mut info);
    compute_clustering(state, &mut info);

    // 4. Conditional Expensive Checks (1-ply Probe)
    // Only run if static threat indicates danger or instability
    // DISABLED: Too expensive for per-node analysis (causes massive slowdown/hang).
    // if info.static_threat_score > 50 || info.king_danger_score[state.side_to_move] > 30 {
    //    compute_forcing_threats(state, &mut info);
    // }

    info
}

// --- LIGHTWEIGHT TACTICAL ANALYSIS ---

/// Checks if a move creates threats, is defensive, or improves dominance.
/// This is designed to be lightweight enough for Search Move Ordering and Sacrifice Logic.
/// Works for both quiet moves and captures (to estimate post-capture threat).
pub fn analyze_move_threat_impact(
    state: &GameState,
    mv: Move,
    current_threat: &ThreatInfo,
) -> ThreatDeltaScore {
    let mut delta = ThreatDeltaScore::default();

    // 1. Dominant Square
    let piece = get_piece_type_safe(state, mv.source);
    delta.dominance_bonus = is_dominant_square(state, mv.target, piece, state.side_to_move);

    // 2. Defensive Move
    // If we are currently under threat, does this move help?
    if current_threat.static_threat_score > 0 || current_threat.king_danger_score[state.side_to_move] > 50 {
        delta.defensive_bonus = evaluate_defensive_move(state, mv, current_threat);
    }

    // Feature #2: Dynamic Coordination (Battery Formation)
    // If quiet move, does it align with other pieces?
    if !mv.is_capture {
        delta.coordination_bonus = evaluate_coordination_impact(state, mv);

        // Feature #3: Pawn Levers
        if piece == P || piece == p {
            delta.pawn_lever_score = get_pawn_lever_score(state, mv);
        }
    }

    // 3. Threat Creation (The expensive part)
    // We only estimate direct threat creation here.
    delta.threat_score = estimate_threat_creation(state, mv);

    // Classification
    if delta.threat_score > SACRIFICE_THREAT_THRESHOLD || delta.defensive_bonus > 50 || delta.dominance_bonus > 30 {
        delta.is_tactical = true;
    }

    delta
}

// Feature #1: Prophylaxis Helper
// Only call this for promising quiet moves!
pub fn get_prophylaxis_score(state: &GameState, mv: Move, current_threat: &ThreatInfo) -> i32 {
    // 1. Make move
    let next_state = state.make_move(mv);

    // 2. Analyze opponent's threat (Static)
    // We want to know if 'next_state' has LESS threat for US than 'state' had.
    // In 'state', threat is calculated for 'state.side_to_move' (US).
    // In 'next_state', side to move is OPPONENT.
    // ThreatInfo.static_threat_score measures threats against the side to move?
    // analyze() computes threats for both, but aggregates into 'static_threat_score' based on hanging pieces etc.

    // Let's re-run analyze lightly.
    // We care about threats AGAINST US (the side that just moved).
    // In 'next_state', US is '1 - next_state.side_to_move'.

    // However, analyze() returns a struct.
    // static_threat_score adds hanging_val.
    // king_danger_score is indexed by side.

    let next_info = analyze(&next_state);

    let us = state.side_to_move;

    // Did King Danger decrease?
    let current_danger = current_threat.king_danger_score[us];
    let next_danger = next_info.king_danger_score[us];

    // Did hanging pieces decrease?
    // hanging_piece_value in ThreatInfo aggregates generally?
    // Actually `compute_static_threats` computes `hanging_val` for `state.side_to_move` (US in `state`).
    // In `next_state`, `side_to_move` is THEM. `hanging_val` would be THEIR hanging pieces.
    // We need to look at `next_info`'s assessment of OUR pieces.
    // `compute_static_threats` only computes for `side_to_move`.

    // This implies `analyze` might not fully give us what we want for the *opponent* (us in next state).
    // But `king_danger_score` IS computed for both sides.

    let mut score = 0;

    if next_danger < current_danger {
        score += (current_danger - next_danger) * 2;
    }

    // Check threats to our pieces in next_state manually?
    // Too expensive.

    // Use `next_info.attacks_by_side[them]` vs `next_info.attacks_by_side[us]`.
    // A simple metric: Total number of squares attacked near our king?

    score
}

fn evaluate_defensive_move(state: &GameState, mv: Move, threat: &ThreatInfo) -> i32 {
    let mut score = 0;
    let us = state.side_to_move;
    // let them = 1 - us;

    // A. Escaping Hanging Piece
    // Check if 'from' square was attacked and not sufficiently defended (hanging)
    // ThreatInfo logic for 'hanging_piece_value' aggregates this, but doesn't store per-square.
    // We can check if `from` is in `attacks_by_side[them]`.
    let is_attacked = threat.attacks_by_side[1 - us].get_bit(mv.source);
    if is_attacked {
        // Simple check: moving away from attack?
        // Ideally we check if 'to' is safe, but that requires re-checking attacks.
        // Assume moving an attacked piece is "defensive".
        let val = get_piece_value_simple(get_piece_type_safe(state, mv.source));
        score += val / 4; // Bonus for moving threatened piece
    }

    // B. King Safety Evasion
    // If king is in danger, and we move a piece near the king (shielding)
    if threat.king_danger_score[us] > 0 {
        let king_sq = state.bitboards[if us == WHITE { K } else { k }].get_lsb_index() as u8;
        let dist = (king_sq as i32 - mv.target as i32).abs(); // Crude distance
        if dist < 16 { // Nearby
             score += 10;
        }
    }

    score
}

// Feature #3: Pawn Levers
fn get_pawn_lever_score(state: &GameState, mv: Move) -> i32 {
    let mut score = 0;
    let us = state.side_to_move;
    let them = 1 - us;

    // 1. Tension: Does it attack an enemy pawn?
    // Pawn attacks from 'target'
    let pawn_attacks = bitboard::pawn_attacks(Bitboard(1 << mv.target), us);
    let enemy_pawns = state.bitboards[if them == WHITE { P } else { p }];
    let attacked_pawns = pawn_attacks & enemy_pawns;

    if attacked_pawns.0 != 0 {
        score += 30;
    }

    // 2. Line Opening Potential (Heuristic)
    // If we capture (handled by capture logic), but this is usually quiet push.
    // If we push to a square where we are attacked by a pawn, we invite a capture -> Open file.
    let enemy_pawn_attacks = bitboard::pawn_attacks(enemy_pawns, them);
    if enemy_pawn_attacks.get_bit(mv.target) {
        score += 20; // Invitation to exchange
    }

    // 3. Squares Controlled (Space)
    // Bonus for advanced ranks (4,5 for White, 3,2 for Black)
    let rank = mv.target / 8;
    if us == WHITE {
        if rank >= 3 { score += 10; }
        if rank >= 4 { score += 10; }
    } else {
        if rank <= 4 { score += 10; }
        if rank <= 3 { score += 10; }
    }

    score
}

// Feature #2: Dynamic Coordination Check
fn evaluate_coordination_impact(state: &GameState, mv: Move) -> i32 {
    let piece = get_piece_type_safe(state, mv.source);
    let us = state.side_to_move;
    // Only sliders
    if piece == N || piece == n || piece == K || piece == k || piece == P || piece == p {
        return 0;
    }

    let occ = state.occupancies[BOTH]; // Approx, ignore self-move change
    let mut score = 0;

    // Check if we align with another friendly piece of compatible type?
    // Rooks align with R/Q on file/rank.
    // Bishops align with B/Q on diagonal.

    let friendly_rooks = state.bitboards[if us == WHITE { R } else { r }];
    let friendly_bishops = state.bitboards[if us == WHITE { B } else { b }];
    let friendly_queens = state.bitboards[if us == WHITE { Q } else { q }];

    match piece % 6 {
        3 => { // R
            // Check file/rank for other R or Q
            let attacks = bitboard::get_rook_attacks(mv.target, occ);
            if (attacks & (friendly_rooks | friendly_queens)).0 != 0 {
                score += 15;
            }
        },
        2 => { // B
            let attacks = bitboard::get_bishop_attacks(mv.target, occ);
             if (attacks & (friendly_bishops | friendly_queens)).0 != 0 {
                score += 15;
            }
        },
        4 => { // Q
            let r_attacks = bitboard::get_rook_attacks(mv.target, occ);
            if (r_attacks & friendly_rooks).0 != 0 { score += 15; }

            let b_attacks = bitboard::get_bishop_attacks(mv.target, occ);
            if (b_attacks & friendly_bishops).0 != 0 { score += 15; }
        },
        _ => {}
    }

    score
}

fn estimate_threat_creation(state: &GameState, mv: Move) -> i32 {
    let mut score = 0;
    let us = state.side_to_move;
    let them = 1 - us;
    let piece = get_piece_type_safe(state, mv.source); // This is piece index (0-11)
    let piece_type = piece % 6; // 0-5

    // Get attacks from new square
    // We use the 'slow' generators or bitboard helpers for the specific square.
    // Optimization: Use `bitboard` helpers.
    let occ = state.occupancies[BOTH]; // Note: This is OLD occupancy. Target is empty (quiet move). Source is occupied.
    // For slider attacks from 'target', we need to pretend source is empty and target is occupied.
    // Update occupancy for ray cast
    let mut new_occ = occ;
    new_occ.pop_bit(mv.source);
    new_occ.set_bit(mv.target);

    let attacks = match piece_type {
        N => movegen::get_knight_attacks(mv.target),
        B => bitboard::get_bishop_attacks(mv.target, new_occ),
        R => bitboard::get_rook_attacks(mv.target, new_occ),
        Q => bitboard::get_queen_attacks(mv.target, new_occ),
        P => bitboard::pawn_attacks(Bitboard(1 << mv.target), us), // Pawn attacks from new square
        K => movegen::get_king_attacks(mv.target),
        _ => Bitboard(0),
    };

    // 1. Direct Attacks on High Value Targets
    let enemy_pieces = state.occupancies[them];
    let targets = attacks & enemy_pieces;

    let mut iter = targets;
    while iter.0 != 0 {
        let sq = iter.get_lsb_index() as u8;
        iter.pop_bit(sq);
        let victim = get_piece_type_safe(state, sq) % 6;

        // Bonus if we attack something valuable
        // And we are not easily captured? (That requires SEE).
        // For "Threat Creation", we just count the threat.
        if victim > piece_type {
            score += (get_piece_value_simple(victim) - get_piece_value_simple(piece_type)) / 2;
        } else if victim == piece_type {
             score += 20; // Attack equal value
        }
    }

    // 2. King Ring Pressure
    let king_sq = state.bitboards[if them == WHITE { K } else { k }].get_lsb_index() as u8;
    // We can't access `threat_info` here easily without recomputing.
    // Use precomputed mask if possible, or just distance.
    // Let's use `movegen::get_king_attacks` as 3x3 ring.
    let ring = movegen::get_king_attacks(king_sq);
    if (attacks & ring).0 != 0 {
        score += 60; // Direct hit on king ring
    }

    // 3. Discovered Attacks (Optional/Expensive)
    // Skipped for speed in this "Lightweight" version.

    score
}

pub fn is_dominant_square(state: &GameState, sq: u8, piece_idx: usize, side: usize) -> i32 {
    let mut score = 0;
    let piece = piece_idx % 6;
    let rank = sq / 8;
    let file = sq % 8;

    let relative_rank = if side == WHITE { rank } else { 7 - rank };

    match piece {
        N => {
            // Outpost: Rank 4, 5, 6
            if relative_rank >= 3 && relative_rank <= 5 {
                // Supported by pawn?
                let pawn = if side == WHITE { P } else { p };
                let pawn_attacks = bitboard::pawn_attacks(Bitboard(1 << sq), 1 - side); // Attack FROM sq AS opponent = squares that attack sq? No.
                // We want squares that attack `sq`.
                // Pawns that attack `sq` are `pawn_attacks(sq, them)` relative to us.
                // Correct: Pawn attacks from `sq` by US go forward.
                // We want to know if `sq` is attacked by OUR pawns.
                // `pawn_attacks(Bitboard(1<<sq), them)` gives squares that `sq` would attack if it was `them`.
                // Wait. `pawn_attacks(bb, side)` returns squares attacked BY `bb` of `side`.
                // Squares that attack `sq` with a pawn of `side`:
                // If white pawn on A2 attacks B3. `pawn_attacks(A2, White)` -> B3.
                // We want to know if `sq` is in `pawn_attacks(OurPawns, Us)`.
                let our_pawns = state.bitboards[pawn];
                let defended_by_pawn = (bitboard::pawn_attacks(our_pawns, side) & Bitboard(1<<sq)).0 != 0;

                if defended_by_pawn {
                    score += 25; // Supported knight
                    if relative_rank >= 4 { score += 15; } // Deep outpost
                }

                // Center Control
                if (file >= 2 && file <= 5) && (rank >= 2 && rank <= 5) {
                    score += 10;
                }
            }
        },
        B => {
             // Center / Long Diagonals
             if (file >= 2 && file <= 5) && (rank >= 2 && rank <= 5) {
                 score += 10;
             }
        },
        R => {
            // Open File (No pawns of ours)
            let our_pawns = state.bitboards[if side == WHITE { P } else { p }];
            let file_bb = bitboard::file_mask(file as usize);
            if (our_pawns & file_bb).0 == 0 {
                score += 20;
                // Semi-open (enemy pawns?)
                // If 7th rank
                if relative_rank == 6 {
                    score += 30;
                }
            }
        },
        _ => {}
    }

    score
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
    // let them_occ = state.occupancies[them];

    // --- Hanging Pieces / En Prise ---
    let mut hanging_val = 0;

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
        // let ring_5 = info.king_ring_5x5[side];

        let mut score = 0;
        let occ = state.occupancies[BOTH];
        let enemy_pawns = state.bitboards[if enemy == WHITE { P } else { p }];
        let enemy_knights = state.bitboards[if enemy == WHITE { N } else { n }];
        // let enemy_bishops = state.bitboards[if enemy == WHITE { B } else { b }] | state.bitboards[if enemy == WHITE { Q } else { q }];
        // let enemy_rooks = state.bitboards[if enemy == WHITE { R } else { r }] | state.bitboards[if enemy == WHITE { Q } else { q }];
        let enemy_queens = state.bitboards[if enemy == WHITE { Q } else { q }];

        let mut attacker_count = 0;
        let mut attacker_weight = 0;

        // Pawns
        let mut pawns = enemy_pawns;
        if (bitboard::pawn_attacks(pawns, enemy) & ring_3).0 != 0 {
             while pawns.0 != 0 {
                 let sq = pawns.get_lsb_index() as u8;
                 pawns.pop_bit(sq);
                 if (bitboard::pawn_attacks(Bitboard(1<<sq), enemy) & ring_3).0 != 0 {
                     attacker_count += 1;
                     attacker_weight += 10;
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
                attacker_weight += 25;
            }
        }

        // Bishops
        let mut bishops = state.bitboards[if enemy == WHITE { B } else { b }];
        while bishops.0 != 0 {
            let sq = bishops.get_lsb_index() as u8;
            bishops.pop_bit(sq);
            if (bitboard::get_bishop_attacks(sq, occ) & ring_3).0 != 0 {
                attacker_count += 1;
                attacker_weight += 25;
            }
        }

        let mut rooks = state.bitboards[if enemy == WHITE { R } else { r }];
        while rooks.0 != 0 {
            let sq = rooks.get_lsb_index() as u8;
            rooks.pop_bit(sq);
            if (bitboard::get_rook_attacks(sq, occ) & ring_3).0 != 0 {
                attacker_count += 1;
                attacker_weight += 50;
            }
        }

        let mut queens = enemy_queens;
        while queens.0 != 0 {
            let sq = queens.get_lsb_index() as u8;
            queens.pop_bit(sq);
            if (bitboard::get_queen_attacks(sq, occ) & ring_3).0 != 0 {
                attacker_count += 1;
                attacker_weight += 75; // Big threat
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

// Feature #5: Threat Clustering
fn compute_clustering(state: &GameState, info: &mut ThreatInfo) {
    // Idea: Detect squares in the king ring (3x3) that are attacked by MULTIPLE enemy pieces.
    // We already have 'attacks_by_side'. But that is a union. We don't know overlap count.
    // We need to iterate enemy pieces again?
    // Optimization: We already iterated in King Danger.
    // But `compute_static_threats` aggregated it into `attacker_count` (which is total attackers on the ring).
    // Clustering means: 2+ attackers on the SAME square.

    // Let's do a lightweight check for the side to move (us).
    // Is *our* king ring clustered by enemy?

    let us = state.side_to_move;
    let them = 1 - us;
    let ring = info.king_ring_3x3[us];

    let mut score = 0;

    // We need per-square attack counts on the ring.
    // This is expensive if we re-generate everything.
    // Let's approximate using bitwise logic?
    // No, we need counts.

    // Let's iterate the ring squares.
    let mut ring_iter = ring;
    while ring_iter.0 != 0 {
        let sq = ring_iter.get_lsb_index() as u8;
        ring_iter.pop_bit(sq);

        let count = count_attackers(state, sq, them);
        if count >= 2 {
            score += (count - 1) * 20; // Multiplier
        }
    }

    info.clustering_score[us] = score;
}

fn count_attackers(state: &GameState, sq: u8, side: usize) -> i32 {
    let mut count = 0;

    // Pawns
    let pawns = state.bitboards[if side == WHITE { P } else { p }];
    if (bitboard::pawn_attacks(Bitboard(1 << sq), 1 - side) & pawns).0 != 0 {
        count += 1; // Count as 1 even if 2 pawns?
        // bitboard::pawn_attacks(sq, us) gives squares that attack sq? No.
        // bitboard::pawn_attacks(sq, them) returns captures.
        // We use reverse logic: pawn_attacks(sq, US) -> squares that attack sq (if they are occupied by pawns)
        // Correct.
        // Ideally we count how many bits.
        // But Pawn attacks are max 2.
        // (bitboard::pawn_attacks(Bitboard(1 << sq), 1 - side) & pawns).count_bits();
        // Since `pawn_attacks` returns a bitboard, we can count bits.
        let attackers = bitboard::pawn_attacks(Bitboard(1 << sq), 1 - side) & pawns;
        count += attackers.count_bits() as i32;
    }

    let knights = state.bitboards[if side == WHITE { N } else { n }];
    let k_att = movegen::get_knight_attacks(sq) & knights;
    count += k_att.count_bits() as i32;

    let occ = state.occupancies[BOTH];

    let b_q = state.bitboards[if side == WHITE { B } else { b }] | state.bitboards[if side == WHITE { Q } else { q }];
    let b_att = bitboard::get_bishop_attacks(sq, occ) & b_q;
    count += b_att.count_bits() as i32;

    let r_q = state.bitboards[if side == WHITE { R } else { r }] | state.bitboards[if side == WHITE { Q } else { q }];
    let r_att = bitboard::get_rook_attacks(sq, occ) & r_q;
    count += r_att.count_bits() as i32;

    count
}

// Feature #2: Global Coordination (Static)
fn compute_coordination(state: &GameState, info: &mut ThreatInfo) {
    // Detect batteries globally for both sides.
    // This feeds into HCE.

    for side in [WHITE, BLACK] {
        let mut score = 0;
        let rooks = state.bitboards[if side == WHITE { R } else { r }];
        let bishops = state.bitboards[if side == WHITE { B } else { b }];
        let queens = state.bitboards[if side == WHITE { Q } else { q }];
        let occ = state.occupancies[BOTH];

        // R+R or R+Q
        let mut r_iter = rooks;
        while r_iter.0 != 0 {
            let sq = r_iter.get_lsb_index() as u8;
            r_iter.pop_bit(sq);
            let att = bitboard::get_rook_attacks(sq, occ);

            // Check for alignment with other R or Q
            if (att & (rooks | queens)).0 != 0 {
                score += 10;
            }
        }

        // B+B or B+Q
        let mut b_iter = bishops;
        while b_iter.0 != 0 {
            let sq = b_iter.get_lsb_index() as u8;
            b_iter.pop_bit(sq);
            let att = bitboard::get_bishop_attacks(sq, occ);
             if (att & (bishops | queens)).0 != 0 {
                score += 10;
            }
        }

        // Q+R or Q+B
        // Counted above?
        // If R sees Q, we added 10. If Q sees R, we should add 10?
        // Yes, battery is strong both ways.
        // But Q seeing R is same battery.
        // We can just iterate all heavy pieces.
        // It's okay to double count, as battery with 3 pieces is stronger.

        info.coordination_score[side] = score;
    }
}

// Feature #4: Enhanced Sacrificial Calculation (2-Ply Forcing Probe)
fn compute_forcing_threats(state: &GameState, info: &mut ThreatInfo) {
    let mut enemy_state = state.clone();
    enemy_state.side_to_move = 1 - state.side_to_move;
    enemy_state.hash = crate::zobrist::side_key() ^ state.hash;

    let mut generator = movegen::MoveGenerator::new();
    generator.generate_moves(&enemy_state);

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

        // Check legality
        let enemy_king = if enemy_state.side_to_move == WHITE { K } else { k };
        let k_sq = next_state.bitboards[enemy_king].get_lsb_index() as u8;
        if movegen::is_square_attacked(&next_state, k_sq, next_state.side_to_move) {
             continue; // Illegal move by enemy
        }

        // --- NEW: 2-Ply Lookahead for Critical Lines ---
        // If this enemy move causes a significant drop, we check our reply.
        // But wait. compute_forcing_threats is trying to find if *WE* are under threat.
        // "If opponent makes a good move, Next should be lower for US."

        // So we are looking for the opponent's best move.
        // If they have a move that drops our eval significantly, we are in danger.

        // Feature #4 says "Check for follow-up threats 1-2 plies ahead".
        // This usually applies to *our* sacrifices.
        // But here we are computing `ThreatInfo` for the *current* state.

        // If we want to detect if *our* sacrifice was good, we do that in search (extensions).
        // If we want to detect if *opponent* has a sacrifice, we do it here.

        // Let's keep this focused on "Opponent Threats".
        // If opponent plays a sacrifice (capture/check), can we survive?
        // If we can't refute it, it's a valid threat.

        // We need to see if we have a reply that restores the eval.
        // 1. Generate OUR moves from next_state.
        // 2. Find best response (shallow).
        // 3. If even with best response, eval is low -> Real Threat.

        let mut best_response_eval = -30000;

        // Only do response check if the drop seems high?
        // Or if the opponent move was a sacrifice?
        // Let's do a quick check.
        let temp_eval = eval::evaluate_hce(&next_state, &ThreatInfo::default());

        // If temp_eval is bad for US (low), we check if we can save it.
        if current_eval - temp_eval > 100 {
            // Check recovery
            let mut my_gen = movegen::MoveGenerator::new();
            my_gen.generate_moves(&next_state);

            for j in 0..my_gen.list.count {
                let my_mv = my_gen.list.moves[j];
                // Try only captures/checks/escapes?
                // For speed, just captures and king moves if in check.
                // Or simplified qsearch logic.

                // Let's just pick one best move via SEE or simple eval?
                // Too slow to run all.
                // Let's just assume we can recapture if available.

                // Use a simplified logic: if we can capture the piece that just moved?
                if my_mv.target == mv.target {
                     // Recapture!
                     let recovery_state = next_state.make_move(my_mv);
                     // Check legality...
                     let my_k = if next_state.side_to_move == WHITE { K } else { k };
                     let mk_sq = recovery_state.bitboards[my_k].get_lsb_index() as u8;
                     if !movegen::is_square_attacked(&recovery_state, mk_sq, recovery_state.side_to_move) {
                          let rec_eval = eval::evaluate_hce(&recovery_state, &ThreatInfo::default());
                          if rec_eval > best_response_eval {
                              best_response_eval = rec_eval;
                          }
                     }
                }
            }

            // If we found a recapture that saves the score, use that.
            if best_response_eval > -29000 {
                // If recapture restores score close to original, then the threat is not real.
                 let drop = current_eval - best_response_eval;
                 if drop > max_drop { max_drop = drop; }
            } else {
                // No recapture? Use the raw drop (assuming we are screwed)
                let drop = current_eval - temp_eval;
                if drop > max_drop { max_drop = drop; }
            }
        } else {
             let drop = current_eval - temp_eval;
             if drop > max_drop { max_drop = drop; }
        }

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

fn get_piece_value_simple(piece_type: usize) -> i32 {
    match piece_type {
        0 => 100,
        1 => 320,
        2 => 330,
        3 => 500,
        4 => 900,
        5 => 20000,
        _ => 0
    }
}

fn get_piece_type_safe(state: &GameState, square: u8) -> usize {
    for piece in 0..12 {
        if state.bitboards[piece].get_bit(square) {
            return piece;
        }
    }
    12
}
