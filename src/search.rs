// src/search.rs
use crate::bitboard::{self, Bitboard};
use crate::eval;
use crate::syzygy;
use crate::movegen::{self, MoveGenerator};
use crate::state::{b, k, n, p, q, r, GameState, Move, B, BLACK, BOTH, K, N, P, Q, R, WHITE};
use crate::threat::{self, ThreatDeltaScore, ThreatInfo};
use crate::time::TimeManager;
use crate::tt::{TranspositionTable, FLAG_ALPHA, FLAG_BETA, FLAG_EXACT};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

const MAX_PLY: usize = 128;
const INFINITY: i32 = 32000;
const MATE_VALUE: i32 = 31000;
const MATE_SCORE: i32 = 30000;

// Reduced LMP Table
const LMP_TABLE: [usize; 16] = [
    0, 2, 4, 7, 10, 15, 20, 28, 38, 50, 65, 80, 100, 120, 150, 200,
];

// Continuation History
type ContHistTable = [[[i32; 64]; 12]; 768];

// LMR Table wrapped in OnceLock for safe runtime initialization
static LMR_TABLE: OnceLock<[[u8; 64]; 64]> = OnceLock::new();

#[derive(Clone, Copy)]
pub enum Limits {
    Infinite,
    FixedDepth(u8),
    FixedNodes(u64),
    FixedTime(TimeManager),
}

pub struct SearchData {
    pub killers: [[Option<Move>; 2]; MAX_PLY + 1],
    pub history: [[i32; 64]; 64],
    pub capture_history: Box<[[[i32; 6]; 64]; 12]>,

    // [Piece][ToSquare] -> Best Reply Move
    pub counter_moves: [[Option<Move>; 64]; 12],

    pub cont_history: Box<ContHistTable>,

    // Correction History: [Piece][Square] -> Error Adjustment
    // Piece index 0-11 covers both Side and PieceType
    pub correction_history: [[i16; 64]; 12],
}

impl SearchData {
    pub fn new() -> Self {
        Self {
            killers: [[None; 2]; MAX_PLY + 1],
            history: [[0; 64]; 64],
            capture_history: Box::new([[[0; 6]; 64]; 12]),
            counter_moves: [[None; 64]; 12],
            cont_history: Box::new([[[0; 64]; 12]; 768]),
            correction_history: [[0; 64]; 12],
        }
    }

    pub fn clear(&mut self) {
        self.killers = [[None; 2]; MAX_PLY + 1];
        self.history = [[0; 64]; 64];
        self.capture_history.fill_with(|| [[0; 6]; 64]);
        self.counter_moves = [[None; 64]; 12];
        self.cont_history.fill_with(|| [[0; 64]; 12]);
        self.correction_history = [[0; 64]; 12];
    }
}

pub struct SearchInfo<'a> {
    pub data: &'a mut SearchData,
    pub static_evals: [i32; MAX_PLY + 1],
    pub nodes: u64,
    pub seldepth: u8,
    pub limits: Limits,
    pub stop_signal: Arc<AtomicBool>,
    pub stopped: bool,
    pub tt: &'a TranspositionTable,
    pub main_thread: bool,
}

impl<'a> SearchInfo<'a> {
    pub fn new(
        data: &'a mut SearchData,
        limits: Limits,
        stop: Arc<AtomicBool>,
        tt: &'a TranspositionTable,
        main: bool,
    ) -> Self {
        LMR_TABLE.get_or_init(|| {
            let mut table = [[0; 64]; 64];
            for d in 0..64 {
                for m in 0..64 {
                    if d > 2 && m > 2 {
                        let lmr = 1.0 + (d as f64).ln() * (m as f64).ln() / 2.5;
                        table[d][m] = lmr as u8;
                    }
                }
            }
            table
        });

        Self {
            data,
            static_evals: [0; MAX_PLY + 1],
            nodes: 0,
            seldepth: 0,
            limits,
            stop_signal: stop,
            stopped: false,
            tt,
            main_thread: main,
        }
    }

    #[inline(always)]
    pub fn check_time(&mut self) {
        if self.nodes % 1024 == 0 {
            if self.stop_signal.load(Ordering::Relaxed) {
                self.stopped = true;
                return;
            }

            match &self.limits {
                Limits::FixedNodes(limit) => {
                    if self.nodes >= *limit {
                        self.stopped = true;
                        self.stop_signal.store(true, Ordering::Relaxed);
                    }
                }
                Limits::FixedTime(tm) => {
                    if self.main_thread && tm.check_hard_limit() {
                        self.stopped = true;
                        self.stop_signal.store(true, Ordering::Relaxed);
                    }
                }
                Limits::FixedDepth(_) | Limits::Infinite => {
                    // Strictly infinite time for these modes.
                    // Only manual stop (handled above) can terminate.
                }
            }
        }
    }
}

// --- FAST CHECK DETECTION ---
fn gives_check_fast(state: &GameState, mv: Move) -> bool {
    let side = state.side_to_move;
    let enemy = 1 - side;
    let enemy_king_sq = state.bitboards[if enemy == WHITE { K } else { k }].get_lsb_index() as u8;

    let mut piece = get_piece_type_safe(state, mv.source);
    if let Some(p_promo) = mv.promotion {
        piece = if side == WHITE { p_promo } else { p_promo + 6 };
    }

    // Chess960 Castling Check: Target is own rook?
    // If so, treat as "not giving check" via direct attack (King doesn't attack enemy king)
    // But we must perform slow check because castling is complex.
    if piece == K || piece == k {
        // If target has friendly rook, castling
        let friendly_rooks = state.bitboards[if side == WHITE { R } else { r }];
        if friendly_rooks.get_bit(mv.target) {
            // It's castling. Fallback to slow check.
            return false;
        }
    }

    let attacks = match piece {
        1 | 7 => crate::movegen::get_knight_attacks(mv.target),
        0 => bitboard::pawn_attacks(Bitboard(1 << mv.target), WHITE),
        6 => bitboard::pawn_attacks(Bitboard(1 << mv.target), BLACK),
        _ => {
            let occ = state.occupancies[crate::state::BOTH];
            let occ_adjusted = (occ.0 & !(1u64 << mv.source)) | (1u64 << mv.target);
            match piece {
                2 | 8 => bitboard::get_bishop_attacks(mv.target, Bitboard(occ_adjusted)),
                3 | 9 => bitboard::get_rook_attacks(mv.target, Bitboard(occ_adjusted)),
                4 | 10 => bitboard::get_queen_attacks(mv.target, Bitboard(occ_adjusted)),
                _ => Bitboard(0),
            }
        }
    };

    if attacks.get_bit(enemy_king_sq) {
        return true;
    }

    let occ_no_source = Bitboard(state.occupancies[crate::state::BOTH].0 & !(1u64 << mv.source));
    let bishops = state.bitboards[if side == WHITE { B } else { b }]
        | state.bitboards[if side == WHITE { Q } else { q }];
    let rooks = state.bitboards[if side == WHITE { R } else { r }]
        | state.bitboards[if side == WHITE { Q } else { q }];

    if (bitboard::get_bishop_attacks(enemy_king_sq, occ_no_source) & bishops).0 != 0 {
        return true;
    }
    if (bitboard::get_rook_attacks(enemy_king_sq, occ_no_source) & rooks).0 != 0 {
        return true;
    }

    false
}

// --- SEE (Static Exchange Evaluation) ---
fn see(state: &GameState, mv: Move) -> i32 {
    let mut gain = [0; 32];
    let mut d = 0;

    let from = mv.source;
    let to = mv.target;
    let mut piece = get_piece_type_safe(state, from);
    let victim = get_piece_type_safe(state, to);

    let piece_vals = [
        100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000, 0,
    ];
    gain[d] = piece_vals[victim];
    d += 1;

    let mut occ = state.occupancies[crate::state::BOTH];
    if mv.is_capture && victim == 12 {
        gain[0] = 100;
        let ep_sq = if state.side_to_move == WHITE {
            to - 8
        } else {
            to + 8
        };
        occ.pop_bit(ep_sq);
    }

    let mut side = state.side_to_move;
    side = 1 - side;

    let mut attackers = get_attackers(state, to, occ);
    attackers.pop_bit(from);
    occ.pop_bit(from);

    loop {
        let lva = get_least_valuable_attacker(state, attackers, side, &mut piece);
        if lva == 64 {
            break;
        }

        gain[d] = piece_vals[piece] - gain[d - 1];
        if (-gain[d - 1]).max(0) + piece_vals[piece] < 0 {
            break;
        }

        d += 1;
        side = 1 - side;
        attackers.pop_bit(lva);
        occ.pop_bit(lva);
    }

    while d > 1 {
        d -= 1;
        gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
    }
    gain[0]
}

fn get_attackers(state: &GameState, sq: u8, occ: Bitboard) -> Bitboard {
    use crate::bitboard::*;
    let mut attackers = Bitboard(0);

    if sq > 8 {
        if (sq % 8) > 0 && state.bitboards[0].get_bit(sq - 9) {
            attackers.set_bit(sq - 9);
        }
        if (sq % 8) < 7 && state.bitboards[0].get_bit(sq - 7) {
            attackers.set_bit(sq - 7);
        }
    }
    if sq < 56 {
        if (sq % 8) > 0 && state.bitboards[6].get_bit(sq + 7) {
            attackers.set_bit(sq + 7);
        }
        if (sq % 8) < 7 && state.bitboards[6].get_bit(sq + 9) {
            attackers.set_bit(sq + 9);
        }
    }

    attackers = attackers | (mask_knight_attacks(sq) & (state.bitboards[1] | state.bitboards[7]));
    attackers = attackers | (mask_king_attacks(sq) & (state.bitboards[5] | state.bitboards[11]));

    let rooks = state.bitboards[3] | state.bitboards[9] | state.bitboards[4] | state.bitboards[10];
    let bishops =
        state.bitboards[2] | state.bitboards[8] | state.bitboards[4] | state.bitboards[10];

    attackers = attackers | (get_rook_attacks(sq, occ) & rooks);
    attackers = attackers | (get_bishop_attacks(sq, occ) & bishops);

    attackers
}

fn get_least_valuable_attacker(
    state: &GameState,
    attackers: Bitboard,
    side: usize,
    piece_type: &mut usize,
) -> u8 {
    let start = if side == WHITE { 0 } else { 6 };
    let end = if side == WHITE { 5 } else { 11 };

    for piece_idx in start..=end {
        let subset = attackers & state.bitboards[piece_idx];
        if subset.0 != 0 {
            *piece_type = piece_idx % 6;
            return subset.get_lsb_index() as u8;
        }
    }
    64
}

fn update_history(entry: &mut i32, bonus: i32) {
    *entry += bonus - (*entry * bonus.abs()) / 16384;
}

fn update_capture_history(info: &mut SearchInfo, mv: Move, state: &GameState, bonus: i32) {
    if !mv.is_capture {
        return;
    }
    let attacker = get_piece_type_safe(state, mv.source);
    let victim = get_victim_type(state, mv.target);

    if victim < 12 {
        let entry = &mut info.data.capture_history[attacker][mv.target as usize][victim % 6];
        *entry += bonus - (*entry * bonus.abs()) / 1024;
    }
}

fn update_continuation_history(
    info: &mut SearchInfo,
    mv: Move,
    prev_move: Option<Move>,
    state: &GameState,
    bonus: i32,
) {
    if let Some(pm) = prev_move {
        let p_piece = get_piece_type_safe(state, pm.target);
        let p_to = pm.target as usize;
        let idx = p_piece * 64 + p_to;

        let c_piece = get_piece_type_safe(state, mv.source);
        let c_to = mv.target as usize;

        update_history(&mut info.data.cont_history[idx][c_piece][c_to], bonus);
    }
}

fn update_correction_history(
    info: &mut SearchInfo,
    prev_move: Option<Move>,
    state: &GameState,
    diff: i32,
    depth: u8,
) {
    if let Some(mv) = prev_move {
        let piece = get_piece_type_safe(state, mv.target);
        let to = mv.target as usize;
        let entry = &mut info.data.correction_history[piece][to];

        let scaled_diff = diff.clamp(-512, 512);
        let weight = (depth as i32).min(16);

        // Update formula: Move towards diff
        let new_val = *entry as i32 + (scaled_diff - *entry as i32) * weight / 64;
        *entry = new_val.clamp(-16000, 16000) as i16;
    }
}

fn score_move(
    mv: Move,
    tt_move: Option<Move>,
    info: &SearchInfo,
    ply: usize,
    state: &GameState,
    prev_move: Option<Move>,
) -> i32 {
    if let Some(tm) = tt_move {
        if mv == tm {
            return 200000;
        }
    }

    let attacker = get_piece_type_safe(state, mv.source);

    if mv.is_capture {
        let see_val = see(state, mv);
        let victim = get_victim_type(state, mv.target);

        let mvv_lva = [
            [105, 104, 103, 102, 101, 100],
            [205, 204, 203, 202, 201, 200],
            [305, 304, 303, 302, 301, 300],
            [405, 404, 403, 402, 401, 400],
            [505, 504, 503, 502, 501, 500],
            [605, 604, 603, 602, 601, 600],
        ];
        let mut score = 100000 + mvv_lva[victim % 6][attacker % 6];

        if victim < 12 {
            score += info.data.capture_history[attacker][mv.target as usize][victim % 6] / 16;
        }

        return if see_val >= 0 { score } else { 0 }; // Bad captures ranked below history/killers (which are > 0)
    }

    if mv.promotion.is_some() {
        return 90000;
    }

    let mut score = 0;
    if ply < MAX_PLY {
        if let Some(k1) = info.data.killers[ply][0] {
            if mv == k1 {
                return 80000;
            }
        }
        if let Some(k2) = info.data.killers[ply][1] {
            if mv == k2 {
                return 79000;
            }
        }
    }

    if let Some(pm) = prev_move {
        let p_piece = get_piece_type_safe(state, pm.target);
        if let Some(cm) = info.data.counter_moves[p_piece][pm.target as usize] {
            if mv == cm {
                return 78000;
            }
        }
    }

    score += info.data.history[mv.source as usize][mv.target as usize];

    if let Some(pm) = prev_move {
        let p_piece = get_piece_type_safe(state, pm.target);
        let idx = p_piece * 64 + pm.target as usize;
        score += info.data.cont_history[idx][attacker][mv.target as usize];
    }

    // REMOVED: Tactical Quiet Move Boosts (Heavy Logic)
    // We now rely on pure history and killers.

    score.min(70000)
}

// Optimized: Checks side occupancy first to halve the bitboard lookups
fn get_piece_type_safe(state: &GameState, square: u8) -> usize {
    // Check White pieces
    if state.occupancies[WHITE].get_bit(square) {
        for piece in 0..6 {
            if state.bitboards[piece].get_bit(square) {
                return piece;
            }
        }
    }
    // Check Black pieces
    else if state.occupancies[BLACK].get_bit(square) {
        for piece in 6..12 {
            if state.bitboards[piece].get_bit(square) {
                return piece;
            }
        }
    }
    // Empty or Invalid
    12
}

fn get_victim_type(state: &GameState, square: u8) -> usize {
    let start = if state.side_to_move == WHITE { 6 } else { 0 };
    let end = if state.side_to_move == WHITE { 11 } else { 5 };
    for piece in start..=end {
        if state.bitboards[piece].get_bit(square) {
            return piece;
        }
    }
    12
}

fn get_pv_line(state: &GameState, tt: &TranspositionTable, depth: u8) -> (String, Option<Move>) {
    let mut pv_str = String::new();
    let mut curr_state = *state;
    let mut seen_hashes = Vec::new();
    let mut ponder_move = None;
    let mut first = true;
    for _ in 0..depth {
        if let Some(mv) = tt.get_move(curr_state.hash) {
            if !tt.is_pseudo_legal(&curr_state, mv) {
                break;
            }
            if seen_hashes.contains(&curr_state.hash) {
                break;
            }
            seen_hashes.push(curr_state.hash);
            pv_str.push_str(&format_move_uci(mv, &curr_state));

            pv_str.push(' ');
            if !first && ponder_move.is_none() {
                ponder_move = Some(mv);
            }
            first = false;
            curr_state = curr_state.make_move(mv);
        } else {
            break;
        }
    }
    (pv_str, ponder_move)
}

fn quiescence(
    state: &GameState,
    mut alpha: i32,
    beta: i32,
    info: &mut SearchInfo,
    ply: usize,
) -> i32 {
    // REMOVED: Heavy Threat Analysis
    // We strictly use `eval::evaluate` which now handles fallback internally if needed.

    if ply > info.seldepth as usize {
        info.seldepth = ply as u8;
    }

    if ply >= MAX_PLY {
        return eval::evaluate(state, alpha, beta);
    }
    info.nodes += 1;
    if info.nodes % 1024 == 0 {
        info.check_time();
    }
    if info.stopped {
        return 0;
    }

    let in_check = is_in_check(state);

    if !in_check {
        let stand_pat = eval::evaluate(state, alpha, beta);
        if stand_pat >= beta {
            return beta;
        }

        let delta = 975;
        use crate::state::{q, Q};
        let is_endgame = (state.bitboards[Q].0 | state.bitboards[q].0) == 0;

        // OPTIMIZATION: Improved Delta Pruning Logic
        if !is_endgame {
            if stand_pat + delta < alpha {
                // We are far below alpha. Even a queen capture won't help.
                return alpha;
            }
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    let mut generator = movegen::MoveGenerator::new();
    generator.generate_moves(state);
    let mut scores = [0; 256];
    for i in 0..generator.list.count {
        scores[i] = score_move(generator.list.moves[i], None, info, ply, state, None);
    }

    let mut legal_moves_found = 0;

    for i in 0..generator.list.count {
        let mut best_idx = i;
        for j in (i + 1)..generator.list.count {
            if scores[j] > scores[best_idx] {
                best_idx = j;
            }
        }
        scores.swap(i, best_idx);
        let mv = generator.list.moves[best_idx];
        generator.list.moves.swap(i, best_idx);

        if !in_check {
            // STRICT Q-Search: Only Captures and Promotions
            if !mv.is_capture && mv.promotion.is_none() {
                continue;
            }
            if mv.is_capture {
                let see_val = see(state, mv);

                // OPTIMIZATION: Prune bad captures in QSearch
                if see_val < 0 {
                    continue; // Aggressively prune bad captures
                }
            }
        }

        if !info.tt.is_pseudo_legal(state, mv) {
            continue;
        }

        let next_state = state.make_move(mv);
        let our_side = state.side_to_move;
        let our_king = if our_side == WHITE { K } else { k };
        let king_sq = next_state.bitboards[our_king].get_lsb_index() as u8;
        if movegen::is_square_attacked(&next_state, king_sq, next_state.side_to_move) {
            continue;
        }

        legal_moves_found += 1;

        let score = -quiescence(&next_state, -beta, -alpha, info, ply + 1);

        if info.stopped {
            return 0;
        }
        if score >= beta {
            update_capture_history(info, mv, state, 100);
            return beta;
        }
        if score > alpha {
            alpha = score;
            update_capture_history(info, mv, state, 50);
        }
    }

    if in_check && legal_moves_found == 0 {
        return -MATE_VALUE + (ply as i32);
    }

    alpha
}

pub fn search(
    state: &GameState,
    limits: Limits,
    tt: &TranspositionTable,
    stop_signal: Arc<AtomicBool>,
    main_thread: bool,
    history: Vec<u64>,
    search_data: &mut SearchData,
) -> (i32, Option<Move>) {
    let mut best_move: Option<Move> = None;
    let mut ponder_move = None;

    // Determine max depth
    let max_depth = match limits {
        Limits::FixedDepth(d) => d,
        _ => MAX_PLY as u8,
    };

    // Syzygy Root Probe
    if main_thread {
        if let Some((tb_move, tb_score)) = syzygy::probe_root(state) {
            println!("info string Syzygy Found: Score {} Move {:?}", tb_score, tb_move);
            // If winning/decisive, just play it.
            // But we might want to search a bit if we want PV.
            // For now, if TB returns, we can trust it.
            // If it's a win, score is high.
            // Let's print info and return.
            let score_str = if tb_score > MATE_SCORE {
                 format!("mate {}", (MATE_VALUE - tb_score + 1) / 2)
            } else if tb_score < -MATE_SCORE {
                 format!("mate -{}", (MATE_VALUE + tb_score + 1) / 2)
            } else {
                 format!("cp {}", tb_score)
            };

            println!("info depth 1 seldepth 1 score {} nodes 0 nps 0 hashfull {} time 0 pv {}",
                     score_str, tt.hashfull(), format_move_uci(tb_move, state));

            println!("bestmove {}", format_move_uci(tb_move, state));
            return (tb_score, Some(tb_move));
        }
    }

    // Store start time for info reporting (independent of limits)
    let start_time = std::time::Instant::now();

    let mut info = Box::new(SearchInfo::new(
        search_data,
        limits,
        stop_signal,
        tt,
        main_thread,
    ));
    let mut path = history;
    let mut last_score = 0;

    // Time Management Variables
    let _nodes_at_root = 0;
    let mut best_move_stability = 0;
    let mut previous_best_move: Option<Move> = None;

    for depth in 1..=max_depth {
        info.seldepth = 0;
        let mut alpha = -INFINITY;
        let mut beta = INFINITY;

        if depth >= 5 && main_thread {
            alpha = last_score - 100;
            beta = last_score + 100;
        }

        let mut score;
        let mut delta = 20; // Narrow window initially

        // Aspiration Windows Loop
        loop {
            // If window is huge, just set to INFINITY
            if alpha < -3000 {
                alpha = -INFINITY;
            }
            if beta > 3000 {
                beta = INFINITY;
            }

            score = negamax(
                state, depth, alpha, beta, &mut info, 0, true, &mut path, None, None, None, false,
            );
            if info.stopped {
                break;
            }

            if score <= alpha {
                beta = (alpha + beta) / 2;
                alpha = (-INFINITY).max(alpha - delta);
                delta += delta / 2 + delta / 4; // Widen faster
            } else if score >= beta {
                beta = (INFINITY).min(beta + delta);
                delta += delta / 2 + delta / 4;
            } else {
                // Exact score found within window
                break;
            }
        }

        last_score = score;

        if info.stopped {
            break;
        }

        // OPTIMIZATION: Dynamic Time Management & Verification
        // Only apply soft limits if we are in a Time Controlled search
        if let Limits::FixedTime(ref mut tm) = info.limits {
            if main_thread && depth > 4 {
                let current_best_move = tt.get_move(state.hash);
                if current_best_move == previous_best_move {
                    best_move_stability += 1;
                } else {
                    best_move_stability = 0;
                }
                previous_best_move = current_best_move;

                let stability_factor = match best_move_stability {
                    0 => 2.50,
                    1 => 1.20,
                    2 => 0.90,
                    3 => 0.80,
                    _ => 0.75,
                };

                tm.set_stability_factor(stability_factor);
            }

            if main_thread && tm.check_soft_limit() {
                info.stopped = true;
                info.stop_signal.store(true, Ordering::Relaxed);
            }
        }

        if main_thread {
            let elapsed = start_time.elapsed().as_secs_f64();
            let nps = if elapsed > 0.0 {
                (info.nodes as f64 / elapsed) as u64
            } else {
                0
            };
            let score_str = if score > MATE_SCORE {
                format!("mate {}", (MATE_VALUE - score + 1) / 2)
            } else if score < -MATE_SCORE {
                format!("mate -{}", (MATE_VALUE + score + 1) / 2)
            } else {
                format!("cp {}", score)
            };

            let mut pv_line = String::new();
            if let Some(mv) = tt.get_move(state.hash) {
                if info.tt.is_pseudo_legal(state, mv) {
                    best_move = Some(mv);
                    let (line, p_move) = get_pv_line(state, tt, depth);
                    pv_line = line;
                    ponder_move = p_move;
                }
            }

            println!(
                "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
                depth,
                info.seldepth,
                score_str,
                info.nodes,
                nps,
                tt.hashfull(),
                start_time.elapsed().as_millis(),
                pv_line
            );
        }
    }

    let mut final_move = best_move;
    let mut generator = movegen::MoveGenerator::new();
    generator.generate_moves(state);
    let mut legal_moves = Vec::new();
    for i in 0..generator.list.count {
        let mv = generator.list.moves[i];
        let next_state = state.make_move(mv);
        let our_king = if state.side_to_move == WHITE { K } else { k };
        let king_sq = next_state.bitboards[our_king].get_lsb_index() as u8;
        if !movegen::is_square_attacked(&next_state, king_sq, next_state.side_to_move) {
            legal_moves.push(mv);
        }
    }

    if let Some(bm) = final_move {
        let mut found = false;
        for &lm in &legal_moves {
            if bm.source == lm.source && bm.target == lm.target && bm.promotion == lm.promotion {
                found = true;
                final_move = Some(lm);
                break;
            }
        }
        if !found {
            final_move = None;
        }
    }

    if final_move.is_none() && !legal_moves.is_empty() {
        final_move = Some(legal_moves[0]);
    }

    if main_thread {
        if let Some(bm) = final_move {
            print!("bestmove {}", format_move_uci(bm, state));
            if let Some(pm) = ponder_move {
                print!(
                    " ponder {}",
                    format_move_uci(pm, state) // state is technically wrong here for ponder move as it's next state, but format_move_uci only needs state for castling detection at source sq
                );
            }
            println!();
        } else {
            println!("bestmove (none)");
        }
    }

    (last_score, final_move)
}

fn negamax(
    state: &GameState,
    depth: u8,
    mut alpha: i32,
    beta: i32,
    info: &mut SearchInfo,
    ply: usize,
    is_pv: bool,
    path: &mut Vec<u64>,
    prev_move: Option<Move>,
    prev_prev_move: Option<Move>,
    excluded_move: Option<Move>,
    was_sacrifice: bool,
) -> i32 {
    if state.halfmove_clock >= 100 {
        return 0;
    }
    if ply > 0 && path.iter().any(|&h| h == state.hash) {
        return 0;
    }

    let mate_value = MATE_VALUE - (ply as i32);
    if alpha < -mate_value {
        alpha = -mate_value;
    }
    if alpha >= beta {
        return alpha;
    }

    if ply >= MAX_PLY {
        return eval::evaluate(state, alpha, beta);
    }

    // Syzygy Probing in Search (WDL)
    // Only probe if not at root (root handled separately), and not too deep (efficiency).
    // Usually probe if ply > 0 and piece count is low.
    // Optimization: Check piece count < X.
    // fathom handles piece count check internally usually? No, we should check.
    // 5-man or 6-man TBs.
    // Probe if <= 6 pieces (arbitrary limit for performance, assume 6-man TB max)
    if ply > 0 && state.occupancies[BOTH].count_bits() <= 6 && state.castling_rights == 0 {
        if let Some(wdl_score) = syzygy::probe_wdl(state) {
            // Syzygy says result.
            // If winning, wdl_score ~ 20000.
            // If losing, wdl_score ~ -20000.
            // Draw ~ 0.

            // Bounds check
            if wdl_score >= beta {
                return wdl_score; // Beta Cutoff
            }
            if wdl_score <= alpha {
                return wdl_score; // Alpha Cutoff (Fail Low)
            }
            // Exact score?
            return wdl_score;
        }
    }

    info.nodes += 1;
    if ply > info.seldepth as usize {
        info.seldepth = ply as u8;
    }
    info.check_time();
    if info.stopped {
        return 0;
    }

    let in_check = is_check(state, state.side_to_move);

    // IMPROVEMENT: Re-enable Check Extension
    // Extend the search depth by 1 when in check to find mates and tactical defenses.
    let mut new_depth = depth;
    if in_check {
        new_depth = new_depth.saturating_add(1);
    }

    if new_depth == 0 {
        return quiescence(state, alpha, beta, info, ply);
    }

    // REMOVED: Expensive Threat Analysis
    // info.current_threat = Some(threat_info);

    // Extensions
    let mut extensions = 0;

    // Tactical Extensions: Pawn to 7th
    // If the previous move put a pawn on the 7th rank (for White) or 2nd rank (for Black), extend.
    if let Some(pm) = prev_move {
        let p_piece = get_piece_type_safe(state, pm.target);
        let rank = pm.target / 8;
        // P=0, p=6. Rank 6 is 7th, Rank 1 is 2nd.
        if (p_piece == P && rank == 6) || (p_piece == p && rank == 1) {
            extensions += 1;
        }
    }

    let mut tt_move = None;
    let mut tt_score = -INFINITY;
    let mut tt_depth = 0;
    let mut tt_flag = FLAG_ALPHA;

    let depth_with_ext = new_depth.saturating_add(extensions);

    if let Some((score, d, flag, mv)) = info.tt.probe_data(state.hash) {
        tt_score = score;
        tt_depth = d;
        tt_flag = flag;
        tt_move = mv;

        if ply > 0 && excluded_move.is_none() && d >= depth_with_ext {
            if flag == FLAG_EXACT {
                return score;
            }
            if flag == FLAG_ALPHA && score <= alpha {
                return alpha;
            }
            if flag == FLAG_BETA && score >= beta {
                return beta;
            }
        }
    }

    let mut raw_eval = -INFINITY;
    let static_eval = if in_check {
        -INFINITY
    } else {
        raw_eval = eval::evaluate(state, alpha, beta);
        let mut correction = 0;
        if let Some(pm) = prev_move {
            let piece = get_piece_type_safe(state, pm.target);
            correction = info.data.correction_history[piece][pm.target as usize] as i32;
        }
        let eval = raw_eval + correction;
        info.static_evals[ply] = eval;
        eval
    };

    let improving = ply >= 2 && !in_check && static_eval >= info.static_evals[ply - 2];

    // --- AGGRESSIVE RAZORING ---
    // If static eval is way below alpha, drop directly to QSearch.
    if !is_pv && !in_check && excluded_move.is_none() && new_depth <= 3 {
        // Margin scales with depth: 450 (d1), 600 (d2), 750 (d3)
        let razor_margin = 300 + (new_depth as i32 * 150);

        if static_eval + razor_margin < alpha {
            let v = quiescence(state, alpha, beta, info, ply);
            if v < alpha {
                return v;
            }
        }
    }

    // RFP (Reverse Futility Pruning) - TUNED: margin = 60 * depth
    if !is_pv
        && !in_check
        && excluded_move.is_none()
        && new_depth < 7
        && static_eval - (60 * new_depth as i32) >= beta
    {
        return static_eval;
    }

    // NULL MOVE PRUNING
    // Removed dependency on tactical_instability
    if new_depth >= 3
        && ply > 0
        && !in_check
        && !is_pv
        && excluded_move.is_none()
        && static_eval >= beta
        && !was_sacrifice
    {
        use crate::state::{b, n, q, r, B, N, Q, R};
        let has_pieces = (state.bitboards[N]
            | state.bitboards[B]
            | state.bitboards[R]
            | state.bitboards[Q]
            | state.bitboards[n]
            | state.bitboards[b]
            | state.bitboards[r]
            | state.bitboards[q])
            .0
            != 0;

        if has_pieces {
            let reduction_depth = 3 + new_depth / 6;
            let null_state = state.make_null_move();
            let reduced_depth = new_depth.saturating_sub(reduction_depth as u8);
            let score = -negamax(
                &null_state,
                reduced_depth,
                -beta,
                -beta + 1,
                info,
                ply + 1,
                false,
                path,
                None,
                None,
                None,
                false, // Null move is not a sacrifice
            );
            if info.stopped {
                return 0;
            }
            if score >= beta {
                return beta;
            }
        }
    }

    // --- PROBCUT ---
    // Optimization: If a reduced depth search returns a value significantly above beta,
    // we can prune this node. (Stockfish / Ethereal Logic)
    if !is_pv
        && new_depth >= 5
        && !in_check
        && excluded_move.is_none()
        && beta.abs() < MATE_SCORE
    {
        let prob_beta = beta + 200;
        let prob_depth = new_depth - 4; // Significantly reduced depth

        let prob_score = -negamax(
            state,
            prob_depth,
            -prob_beta,
            -prob_beta + 1,
            info,
            ply + 1,
            false,
            path,
            prev_move,
            prev_prev_move,
            None,
            was_sacrifice,
        );

        if prob_score >= prob_beta {
            return prob_score;
        }
    }

    if is_pv && tt_move.is_none() && new_depth > 4 {
        let iid_depth = new_depth - 2;
        negamax(
            state,
            iid_depth,
            alpha,
            beta,
            info,
            ply,
            is_pv,
            path,
            prev_move,
            prev_prev_move,
            None,
            was_sacrifice,
        );
        if let Some(mv) = info.tt.get_move(state.hash) {
            tt_move = Some(mv);
        }
    }

    let mut extension = 0;
    if ply > 0
        && new_depth >= 10
        && tt_move.is_some()
        && excluded_move.is_none()
        && tt_depth >= new_depth.saturating_sub(3)
        && tt_flag == FLAG_EXACT
        && tt_score.abs() < MATE_SCORE
    {
        let singular_beta = tt_score.saturating_sub(new_depth as i32);
        let reduced_depth = (new_depth - 1) / 2;

        let score = negamax(
            state,
            reduced_depth,
            singular_beta - 1,
            singular_beta,
            info,
            ply,
            false,
            path,
            prev_move,
            prev_prev_move,
            tt_move,
            was_sacrifice,
        );

        if score < singular_beta {
            extension = 1;
            if !is_pv && score < singular_beta - 16 {
                extension = 2;
            }
        } else if singular_beta >= beta {
            return singular_beta;
        }
    }

    let mut generator = MoveGenerator::new();
    generator.generate_moves(state);
    let mut scores = [0; 256];

    for i in 0..generator.list.count {
        scores[i] = score_move(
            generator.list.moves[i],
            tt_move,
            info,
            ply,
            state,
            prev_move,
        );
    }

    let mut max_score = -INFINITY;
    let mut best_move = None;
    let mut moves_searched = 0;
    let mut quiets_checked = 0;

    path.push(state.hash);

    for i in 0..generator.list.count {
        let mut best_idx = i;
        for j in (i + 1)..generator.list.count {
            if scores[j] > scores[best_idx] {
                best_idx = j;
            }
        }
        scores.swap(i, best_idx);
        let mv = generator.list.moves[best_idx];
        generator.list.moves.swap(i, best_idx);

        if Some(mv) == excluded_move {
            continue;
        }
        if !info.tt.is_pseudo_legal(state, mv) {
            continue;
        }

        let is_quiet = !mv.is_capture && mv.promotion.is_none();

        // 1. FUTILITY PRUNING
        // Condition: Low depth, not in check, not PV, quiet move
        if new_depth < 5 && !in_check && !is_pv && is_quiet {
            let futility_margin = 150 * new_depth as i32;
            if static_eval + futility_margin <= alpha {
                quiets_checked += 1;
                continue;
            }
        }

        if !is_pv && !in_check && new_depth <= 8 && is_quiet {
            if quiets_checked >= LMP_TABLE[new_depth as usize] {
                continue;
            }
        }

        let next_state = state.make_move(mv);
        let our_side = state.side_to_move;
        let our_king = if our_side == WHITE { K } else { k };
        let king_sq = next_state.bitboards[our_king].get_lsb_index() as u8;
        if movegen::is_square_attacked(&next_state, king_sq, next_state.side_to_move) {
            continue;
        }

        info.tt.prefetch(next_state.hash);
        moves_searched += 1;
        if is_quiet {
            quiets_checked += 1;
        }

        let move_extension = 0;
        // Removed sacrifice candidate extension

        let mut score;
        if moves_searched == 1 {
            let total_extension = (extensions + move_extension + extension).min(1);
            let extended_depth = new_depth.saturating_add(total_extension);
            score = -negamax(
                &next_state,
                extended_depth - 1,
                -beta,
                -alpha,
                info,
                ply + 1,
                true,
                path,
                Some(mv),
                prev_move,
                None,
                false, // Removed sacrifice prediction
            );
        } else {
            // OPTIMIZATION: Use next_state for check detection instead of re-calculating
            let gives_check = is_in_check(&next_state);
            let mut reduction = 0;

            // Adjusted LMR
            // Condition: Depth >= 3, Moved > 3, Quiet, Not in check, Not giving check, Not killer
            if new_depth >= 3
                && moves_searched > 3
                && is_quiet
                && !gives_check
                && !in_check
                && (ply >= MAX_PLY
                    || (info.data.killers[ply][0] != Some(mv)
                        && info.data.killers[ply][1] != Some(mv)))
            {
                let d_idx = new_depth.min(63) as usize;
                let m_idx = moves_searched.min(63) as usize;
                let mut lmr_r = LMR_TABLE.get().unwrap()[d_idx][m_idx] as i32;

                let history = info.data.history[mv.source as usize][mv.target as usize];
                lmr_r -= history / 8192;

                reduction = lmr_r.max(0) as u8;
            }

            let d = new_depth.saturating_sub(1 + reduction);
            score = -negamax(
                &next_state,
                d,
                -alpha - 1,
                -alpha,
                info,
                ply + 1,
                false,
                path,
                Some(mv),
                prev_move,
                None,
                false,
            );

            if score > alpha && reduction > 0 {
                score = -negamax(
                    &next_state,
                    new_depth - 1,
                    -alpha - 1,
                    -alpha,
                    info,
                    ply + 1,
                    false,
                    path,
                    Some(mv),
                    prev_move,
                    None,
                    false,
                );
            }
            if score > alpha && score < beta {
                let total_extension = (extensions + move_extension + extension).min(1);
                let extended_depth = new_depth.saturating_add(total_extension);

                score = -negamax(
                    &next_state,
                    extended_depth - 1,
                    -beta,
                    -alpha,
                    info,
                    ply + 1,
                    true,
                    path,
                    Some(mv),
                    prev_move,
                    None,
                    false,
                );
            }
        }

        if info.stopped {
            path.pop();
            return 0;
        }

        if score > max_score {
            max_score = score;
            best_move = Some(mv);
            if score > alpha {
                alpha = score;
                if is_quiet {
                    let from = mv.source as usize;
                    let to = mv.target as usize;
                    let bonus = (new_depth as i32) * (new_depth as i32);

                    update_history(&mut info.data.history[from][to], bonus);

                    if let Some(pm) = prev_move {
                        let p_piece = get_piece_type_safe(state, pm.target);
                        let p_to = pm.target as usize;
                        let idx = p_piece * 64 + p_to;
                        let c_piece = get_piece_type_safe(state, mv.source);
                        let c_to = mv.target as usize;
                        update_history(&mut info.data.cont_history[idx][c_piece][c_to], bonus);

                        info.data.counter_moves[p_piece][pm.target as usize] = Some(mv);
                    }
                } else {
                    update_capture_history(
                        info,
                        mv,
                        state,
                        (new_depth as i32) * (new_depth as i32),
                    );
                }
            }
        } else {
            // HISTORY PENALTY
            if is_quiet {
                let from = mv.source as usize;
                let to = mv.target as usize;
                let bonus = (new_depth as i32) * (new_depth as i32);
                // Penalize logic: -bonus
                update_history(&mut info.data.history[from][to], -bonus);

                if let Some(pm) = prev_move {
                    let p_piece = get_piece_type_safe(state, pm.target);
                    let p_to = pm.target as usize;
                    let idx = p_piece * 64 + p_to;
                    let c_piece = get_piece_type_safe(state, mv.source);
                    let c_to = mv.target as usize;
                    update_history(&mut info.data.cont_history[idx][c_piece][c_to], -bonus);
                }
            }
        }

        if alpha >= beta {
            if is_quiet {
                if ply < MAX_PLY {
                    info.data.killers[ply][1] = info.data.killers[ply][0];
                    info.data.killers[ply][0] = Some(mv);
                }
            } else {
                update_capture_history(info, mv, state, (new_depth as i32) * (new_depth as i32));
            }
            break;
        }
    }

    path.pop();

    if moves_searched == 0 {
        if in_check {
            return -MATE_VALUE + (ply as i32);
        } else {
            return 0;
        }
    }

    // Update Correction History
    if excluded_move.is_none() && !in_check && raw_eval != -INFINITY && max_score.abs() < MATE_SCORE
    {
        let diff = max_score - raw_eval;
        update_correction_history(info, prev_move, state, diff, new_depth);
    }

    let flag = if max_score <= alpha {
        FLAG_ALPHA
    } else if max_score >= beta {
        FLAG_BETA
    } else {
        FLAG_EXACT
    };
    if excluded_move.is_none() {
        info.tt
            .store(state.hash, max_score, best_move, new_depth, flag);
    }
    max_score
}

// CHANGE: pub fn is_check (renamed/exposed)
pub fn is_check(state: &GameState, side: usize) -> bool {
    let king_type = if side == WHITE { K } else { k };
    let king_sq = state.bitboards[king_type].get_lsb_index() as u8;
    movegen::is_square_attacked(state, king_sq, 1 - side)
}

pub fn is_in_check(state: &GameState) -> bool {
    is_check(state, state.side_to_move)
}

fn moves_gives_check(state: &GameState, mv: Move) -> bool {
    if gives_check_fast(state, mv) {
        return true;
    }
    let next_state = state.make_move(mv);
    is_in_check(&next_state)
}

pub fn square_to_coord(s: u8) -> String {
    let f = (b'a' + (s % 8)) as char;
    let rank_char = (b'1' + (s / 8)) as char;
    format!("{}{}", f, rank_char)
}

pub fn format_move_uci(mv: Move, state: &GameState) -> String {
    let chess960 = crate::uci::UCI_CHESS960.load(Ordering::Relaxed);
    let from = mv.source;
    let mut to = mv.target;

    // Check for castling
    // Logic: Internal is King -> Rook.
    let piece = get_piece_type_safe(state, from);
    let is_castling = (piece == 5 || piece == 11) &&
                      (get_piece_type_safe(state, to) == if piece == 5 { 3 } else { 9 }) &&
                      // Ensure target contains friendly rook
                      (state.bitboards[if piece == 5 { R } else { r }].get_bit(to));

    if !chess960 && is_castling {
        if from == 4 {
            // White King on e1
            // Check Kingside (target > source or file compare)
            if to > from {
                to = 6;
            }
            // g1
            else {
                to = 2;
            } // c1
        } else if from == 60 {
            // Black King on e8
            if to > from {
                to = 62;
            }
            // g8
            else {
                to = 58;
            } // c8
        }
    }

    let mut s = format!("{}{}", square_to_coord(from), square_to_coord(to));

    if let Some(promo) = mv.promotion {
        let char = match promo {
            4 | 10 => 'q',
            3 | 9 => 'r',
            2 | 8 => 'b',
            1 | 7 => 'n',
            _ => ' ',
        };
        if char != ' ' {
            s.push(char);
        }
    }
    s
}
