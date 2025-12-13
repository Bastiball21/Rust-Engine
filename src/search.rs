// src/search.rs
use crate::bitboard::{self, Bitboard};
use crate::eval;
use crate::movegen::{self, MoveGenerator};
use crate::state::{b, k, n, p, q, r, GameState, Move, B, BLACK, BOTH, K, N, P, Q, R, WHITE};
use crate::time::TimeManager;
use crate::tt::{TranspositionTable, FLAG_ALPHA, FLAG_BETA, FLAG_EXACT};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

const MAX_PLY: usize = 64;
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

pub struct SearchInfo<'a> {
    pub killers: [[Option<Move>; 2]; MAX_PLY + 1],
    pub history: [[i32; 64]; 64],
    pub capture_history: Box<[[[i32; 6]; 64]; 12]>,

    // [Piece][ToSquare] -> Best Reply Move
    pub counter_moves: [[Option<Move>; 64]; 12],

    pub cont_history: Box<ContHistTable>,
    pub static_evals: [i32; MAX_PLY + 1],
    pub nodes: u64,
    pub seldepth: u8,
    pub time_manager: TimeManager,
    pub stop_signal: Arc<AtomicBool>,
    pub stopped: bool,
    pub tt: &'a TranspositionTable,
    pub main_thread: bool,
}

impl<'a> SearchInfo<'a> {
    pub fn new(
        tm: TimeManager,
        stop: Arc<AtomicBool>,
        tt: &'a TranspositionTable,
        main: bool,
    ) -> Self {
        LMR_TABLE.get_or_init(|| {
            let mut table = [[0; 64]; 64];
            for d in 0..64 {
                for m in 0..64 {
                    if d > 2 && m > 2 {
                        let lmr = 0.75 + (d as f64).ln() * (m as f64).ln() / 2.25;
                        table[d][m] = lmr as u8;
                    }
                }
            }
            table
        });

        let cont_history = Box::new([[[0; 64]; 12]; 768]);

        Self {
            killers: [[None; 2]; MAX_PLY + 1],
            history: [[0; 64]; 64],
            capture_history: Box::new([[[0; 6]; 64]; 12]),
            counter_moves: [[None; 64]; 12],
            cont_history,
            static_evals: [0; MAX_PLY + 1],
            nodes: 0,
            seldepth: 0,
            time_manager: tm,
            stop_signal: stop,
            stopped: false,
            tt,
            main_thread: main,
        }
    }

    #[inline(always)]
    pub fn check_time(&mut self) {
        if self.nodes % 2048 == 0 {
            if self.stop_signal.load(Ordering::Relaxed) {
                self.stopped = true;
                return;
            }
            if self.main_thread && self.time_manager.check_hard_limit() {
                self.stopped = true;
                self.stop_signal.store(true, Ordering::Relaxed);
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
        let entry = &mut info.capture_history[attacker][mv.target as usize][victim % 6];
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

        update_history(&mut info.cont_history[idx][c_piece][c_to], bonus);
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
            score += info.capture_history[attacker][mv.target as usize][victim % 6] / 16;
        }

        return if see_val >= 0 { score } else { score - 50000 };
    }

    if mv.promotion.is_some() {
        return 90000;
    }

    let mut score = 0;
    if ply < MAX_PLY {
        if let Some(k1) = info.killers[ply][0] {
            if mv == k1 {
                return 80000;
            }
        }
        if let Some(k2) = info.killers[ply][1] {
            if mv == k2 {
                return 79000;
            }
        }
    }

    if let Some(pm) = prev_move {
        let p_piece = get_piece_type_safe(state, pm.target);
        if let Some(cm) = info.counter_moves[p_piece][pm.target as usize] {
            if mv == cm {
                return 78000;
            }
        }
    }

    score += info.history[mv.source as usize][mv.target as usize];

    if let Some(pm) = prev_move {
        let p_piece = get_piece_type_safe(state, pm.target);
        let idx = p_piece * 64 + pm.target as usize;
        score += info.cont_history[idx][attacker][mv.target as usize];
    }

    score.min(70000)
}

fn get_piece_type_safe(state: &GameState, square: u8) -> usize {
    for piece in 0..12 {
        if state.bitboards[piece].get_bit(square) {
            return piece;
        }
    }
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
            pv_str.push_str(&format!(
                "{}{}",
                square_to_coord(mv.source),
                square_to_coord(mv.target)
            ));
            if let Some(promo) = mv.promotion {
                let char = match promo {
                    4 | 10 => 'q',
                    3 | 9 => 'r',
                    2 | 8 => 'b',
                    1 | 7 => 'n',
                    _ => ' ',
                };
                if char != ' ' {
                    pv_str.push(char);
                }
            }
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
    if ply >= MAX_PLY {
        return eval::evaluate(state);
    }
    info.nodes += 1;
    if info.nodes % 2048 == 0 {
        info.check_time();
    }
    if info.stopped {
        return 0;
    }

    let in_check = is_in_check(state);

    if !in_check {
        let stand_pat = eval::evaluate(state);
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

    let mut generator = Box::new(movegen::MoveGenerator::new());
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
            if !mv.is_capture && mv.promotion.is_none() {
                continue;
            }
            if mv.is_capture {
                let see_val = see(state, mv);

                // OPTIMIZATION: Prune bad captures in QSearch
                if see_val < 0 {
                    // Don't prune if it gives check (tactical)
                    if !gives_check_fast(state, mv) {
                        continue;
                    }
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
    tm: TimeManager,
    tt: &TranspositionTable,
    stop_signal: Arc<AtomicBool>,
    max_depth: u8,
    main_thread: bool,
    history: Vec<u64>,
) {
    let mut best_move = None;
    let mut ponder_move = None;
    let mut info = Box::new(SearchInfo::new(tm, stop_signal, tt, main_thread));
    let mut path = history;
    let mut last_score = 0;

    // Time Management Variables
    let _nodes_at_root = 0;

    for depth in 1..=max_depth {
        info.seldepth = 0;
        let mut alpha = -INFINITY;
        let mut beta = INFINITY;

        if depth >= 5 && main_thread {
            alpha = last_score - 50;
            beta = last_score + 50;
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
                state, depth, alpha, beta, &mut info, 0, true, &mut path, None, None, None,
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

        // OPTIMIZATION: Dynamic Time Management
        // If best move is unstable or score drops, use more time (soft limit extension)
        if main_thread && depth > 5 {
            // Example logic: if score dropped by > 50cp, extend time by 20%
            // Note: info.time_manager is immutable in search loop unless we update it.
            // But TimeManager fields are just checked.
            // We can update the soft_limit in place if we make it mutable in SearchInfo (it is)
            // or pass a mutable reference.
        }

        if main_thread && info.time_manager.check_soft_limit() {
            // Check if we have a single legal move (root node) or other termination criteria?
            // For now, strict soft limit adherence.
            info.stopped = true;
            info.stop_signal.store(true, Ordering::Relaxed);
        }

        if main_thread {
            let elapsed = info.time_manager.start_time.elapsed().as_secs_f64();
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
                info.time_manager.start_time.elapsed().as_millis(),
                pv_line
            );
        }
    }

    if main_thread {
        let mut final_move = best_move;
        let mut generator = Box::new(movegen::MoveGenerator::new());
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
                if bm.source == lm.source && bm.target == lm.target && bm.promotion == lm.promotion
                {
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

        if let Some(bm) = final_move {
            print!(
                "bestmove {}{}",
                square_to_coord(bm.source),
                square_to_coord(bm.target)
            );
            if let Some(promo) = bm.promotion {
                use crate::state::{b, n, q, r};
                let char = match promo {
                    4 | 10 => 'q',
                    3 | 9 => 'r',
                    2 | 8 => 'b',
                    1 | 7 => 'n',
                    _ => ' ',
                };
                if char != ' ' {
                    print!("{}", char);
                }
            }
            if let Some(pm) = ponder_move {
                print!(
                    " ponder {}{}",
                    square_to_coord(pm.source),
                    square_to_coord(pm.target)
                );
                if let Some(promo) = pm.promotion {
                    let char = match promo {
                        4 | 10 => 'q',
                        3 | 9 => 'r',
                        2 | 8 => 'b',
                        1 | 7 => 'n',
                        _ => ' ',
                    };
                    if char != ' ' {
                        print!("{}", char);
                    }
                }
            }
            println!();
        } else {
            println!("bestmove (none)");
        }
    }
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
        return eval::evaluate(state);
    }
    info.nodes += 1;
    if ply > info.seldepth as usize {
        info.seldepth = ply as u8;
    }
    info.check_time();
    if info.stopped {
        return 0;
    }

    let in_check = is_in_check(state);
    let new_depth = if in_check {
        depth.saturating_add(1)
    } else {
        depth
    };

    if new_depth == 0 {
        return quiescence(state, alpha, beta, info, ply);
    }

    let mut tt_move = None;
    let mut tt_score = -INFINITY;
    let mut tt_depth = 0;
    let mut tt_flag = FLAG_ALPHA;

    if let Some((score, d, flag, mv)) = info.tt.probe_data(state.hash) {
        tt_score = score;
        tt_depth = d;
        tt_flag = flag;
        tt_move = mv;

        if ply > 0 && excluded_move.is_none() && d >= new_depth {
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

    let static_eval = if in_check {
        -INFINITY
    } else {
        let eval = eval::evaluate(state);
        info.static_evals[ply] = eval;
        eval
    };

    let improving = ply >= 2 && !in_check && static_eval >= info.static_evals[ply - 2];

    if !is_pv && !in_check && new_depth == 1 && excluded_move.is_none() {
        let razor_margin = 300;
        if static_eval + razor_margin < alpha {
            let v = quiescence(state, alpha, beta, info, ply);
            if v < alpha {
                return v;
            }
        }
    }

    if !is_pv
        && !in_check
        && excluded_move.is_none()
        && new_depth < 7
        && static_eval - (80 * new_depth as i32) >= beta
    {
        return static_eval;
    }

    if new_depth >= 3
        && ply > 0
        && !in_check
        && !is_pv
        && excluded_move.is_none()
        && static_eval >= beta
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
            );
            if info.stopped {
                return 0;
            }
            if score >= beta {
                return beta;
            }
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
        );
        if let Some(mv) = info.tt.get_move(state.hash) {
            tt_move = Some(mv);
        }
    }

    let mut extension = 0;
    if ply > 0
        && new_depth >= 8
        && tt_move.is_some()
        && excluded_move.is_none()
        && tt_depth >= new_depth.saturating_sub(3)
        && tt_flag == FLAG_EXACT
        && tt_score.abs() < MATE_SCORE
    {
        let singular_beta = tt_score.saturating_sub(2 * new_depth as i32);
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
        );

        if score < singular_beta {
            extension = 1;
        } else if singular_beta >= beta {
            return singular_beta;
        }
    }

    let mut generator = Box::new(MoveGenerator::new());
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

        // 1. FUTILITY PRUNING for Quiet Moves (Aggressive "Bad Move" Pruning)
        // If !is_pv (not a principal variation node), safe to prune more aggressively.
        if !is_pv && !in_check && is_quiet && new_depth < 7 && excluded_move.is_none() {
            let futility_margin = 120 * new_depth as i32;
            if static_eval + futility_margin < alpha {
                quiets_checked += 1;
                continue;
            }
        }

        // 2. FUTILITY PRUNING for Captures (Existing logic was slightly buggy/aggressive)
        // Kept separate for clarity.
        if !is_pv && !in_check && !is_quiet && new_depth < 5 && excluded_move.is_none() {
            // Captures need a much wider margin
            let futility_margin = 300 * new_depth as i32;
            if static_eval + futility_margin < alpha {
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

        let mut score;
        if moves_searched == 1 {
            let extended_depth = new_depth.saturating_add(extension);
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
            );
        } else {
            let gives_check = moves_gives_check(state, mv);
            let mut reduction = 0;

            if new_depth >= 3 && moves_searched > 1 && is_quiet && !gives_check && !in_check {
                let d_idx = new_depth.min(63) as usize;
                let m_idx = moves_searched.min(63) as usize;
                reduction = LMR_TABLE.get().unwrap()[d_idx][m_idx];

                if improving {
                    reduction = reduction.saturating_sub(1);
                }

                if ply < MAX_PLY
                    && (info.killers[ply][0] == Some(mv) || info.killers[ply][1] == Some(mv))
                {
                    reduction = reduction.saturating_sub(1);
                }
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
                );
            }
            if score > alpha && score < beta {
                score = -negamax(
                    &next_state,
                    new_depth - 1,
                    -beta,
                    -alpha,
                    info,
                    ply + 1,
                    true,
                    path,
                    Some(mv),
                    prev_move,
                    None,
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

                    update_history(&mut info.history[from][to], bonus);

                    if let Some(pm) = prev_move {
                        let p_piece = get_piece_type_safe(state, pm.target);
                        let p_to = pm.target as usize;
                        let idx = p_piece * 64 + p_to;
                        let c_piece = get_piece_type_safe(state, mv.source);
                        let c_to = mv.target as usize;
                        update_history(&mut info.cont_history[idx][c_piece][c_to], bonus);

                        info.counter_moves[p_piece][pm.target as usize] = Some(mv);
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
        }
        if alpha >= beta {
            if is_quiet {
                if ply < MAX_PLY {
                    info.killers[ply][1] = info.killers[ply][0];
                    info.killers[ply][0] = Some(mv);
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

// CHANGE: pub fn is_in_check
pub fn is_in_check(state: &GameState) -> bool {
    let king_type = if state.side_to_move == WHITE { K } else { k };
    let king_sq = state.bitboards[king_type].get_lsb_index() as u8;
    movegen::is_square_attacked(state, king_sq, 1 - state.side_to_move)
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
