use crate::state::{GameState, Move, WHITE, K, k, N, B, R, Q, n, b, r, q}; 
use crate::movegen::{self, MoveGenerator};
use crate::eval;
use crate::tt::{TranspositionTable, FLAG_EXACT, FLAG_ALPHA, FLAG_BETA}; 
use crate::time::TimeManager;
use crate::bitboard::{self, Bitboard};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const MAX_PLY: usize = 64; 
const INFINITY: i32 = 50000;
const MATE_VALUE: i32 = 49000;
const MATE_SCORE: i32 = 48000; 

pub struct SearchInfo<'a> {
    pub killers: [[Option<Move>; 2]; MAX_PLY + 1],  
    pub history: [[i32; 64]; 64],
    pub nodes: u64,
    pub seldepth: u8,
    pub time_manager: TimeManager,
    pub stop_signal: Arc<AtomicBool>, 
    pub stopped: bool,
    pub tt: &'a TranspositionTable,
    pub main_thread: bool,
}

impl<'a> SearchInfo<'a> {
    pub fn new(tm: TimeManager, stop: Arc<AtomicBool>, tt: &'a TranspositionTable, main: bool) -> Self {
        Self { 
            killers: [[None; 2]; MAX_PLY + 1], 
            history: [[0; 64]; 64], 
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

// --- HELPERS ---
fn see(state: &GameState, mv: Move) -> i32 {
    let from = mv.source;
    let to = mv.target;
    let victim_type = get_piece_type_safe(state, to);
    if victim_type == 12 { return 0; } // En passant or error
    let values = [100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000];
    let victim_val = values[victim_type % 6];
    let attacker_type = get_piece_type_safe(state, from);
    let attacker_val = values[attacker_type % 6];
    
    // Simple SEE approximation: if victim > attacker, it's good.
    if victim_val >= attacker_val { return 1; }
    
    // If victim < attacker, check if the square is defended
    let enemy = 1 - state.side_to_move;
    if is_square_attacked_by_pawn(state, to, enemy) {
        return victim_val - attacker_val; 
    }
    0 
}

fn is_square_attacked_by_pawn(state: &GameState, square: u8, attacker_side: usize) -> bool {
    use crate::state::{P, p};
    if attacker_side == WHITE {
        if square > 8 {
            if (square % 8) > 0 && state.bitboards[P].get_bit(square - 9) { return true; }
            if (square % 8) < 7 && state.bitboards[P].get_bit(square - 7) { return true; }
        }
    } else {
        if square < 56 {
            if (square % 8) > 0 && state.bitboards[p].get_bit(square + 7) { return true; }
            if (square % 8) < 7 && state.bitboards[p].get_bit(square + 9) { return true; }
        }
    }
    false
}

// Fast pseudo-legal validation using TT logic
fn is_valid_tt_move(state: &GameState, mv: Move, tt: &TranspositionTable) -> bool {
    if !tt.is_pseudo_legal(state, mv) { return false; }
    let next_state = state.make_move(mv);
    let our_side = state.side_to_move;
    let our_king_type = if our_side == WHITE { K } else { k };
    let our_king_sq = next_state.bitboards[our_king_type].get_lsb_index() as u8;
    !movegen::is_square_attacked(&next_state, our_king_sq, next_state.side_to_move)
}

fn is_safe(state: &GameState, mv: Move) -> bool {
    let next_state = state.make_move(mv);
    let our_side = state.side_to_move;
    let our_king_type = if our_side == WHITE { K } else { k };
    if next_state.bitboards[our_king_type].0 == 0 { return false; }
    let our_king_sq = next_state.bitboards[our_king_type].get_lsb_index() as u8;
    !movegen::is_square_attacked(&next_state, our_king_sq, next_state.side_to_move)
}

fn get_pv_line(state: &GameState, tt: &TranspositionTable, depth: u8) -> (String, Option<Move>) {
    let mut pv_str = String::new();
    let mut curr_state = *state;
    let mut seen_hashes = Vec::new();
    let mut ponder_move = None;
    let mut first = true;
    for _ in 0..depth {
        if let Some(mv) = tt.get_move(curr_state.hash) {
            if !is_valid_tt_move(&curr_state, mv, tt) { break; }
            if seen_hashes.contains(&curr_state.hash) { break; }
            seen_hashes.push(curr_state.hash);
            pv_str.push_str(&format!("{}{} ", square_to_coord(mv.source), square_to_coord(mv.target)));
            if !first && ponder_move.is_none() { ponder_move = Some(mv); }
            first = false;
            curr_state = curr_state.make_move(mv);
        } else { break; }
    }
    (pv_str, ponder_move)
}

fn score_move(mv: Move, tt_move: Option<Move>, info: &SearchInfo, ply: usize, state: &GameState) -> i32 {
    if let Some(tm) = tt_move { if mv == tm { return 100000; } }
    if mv.is_capture {
        let see_val = see(state, mv);
        let attacker = get_piece_type_safe(state, mv.source) % 6;
        let victim = get_victim_type(state, mv.target) % 6;
        let mvv_lva = [[105, 104, 103, 102, 101, 100], [205, 204, 203, 202, 201, 200], [305, 304, 303, 302, 301, 300], [405, 404, 403, 402, 401, 400], [505, 504, 503, 502, 501, 500], [605, 604, 603, 602, 601, 600]];
        let score = 50000 + mvv_lva[victim][attacker];
        return if see_val < 0 { score - 40000 } else { score }; 
    }
    if mv.promotion.is_some() { return 40000; }
    if ply < MAX_PLY {
        if let Some(k1) = info.killers[ply][0] { if mv == k1 { return 30000; } }
        if let Some(k2) = info.killers[ply][1] { if mv == k2 { return 29000; } }
    }
    let hist = info.history[mv.source as usize][mv.target as usize];
    1000 + (hist / 16)
}

fn get_piece_type_safe(state: &GameState, square: u8) -> usize {
    for p in 0..12 { if state.bitboards[p].get_bit(square) { return p; } } 0 
}

fn get_victim_type(state: &GameState, square: u8) -> usize {
    let start = if state.side_to_move == WHITE { 6 } else { 0 };
    let end = if state.side_to_move == WHITE { 11 } else { 5 };
    for p in start..=end { if state.bitboards[p].get_bit(square) { return p; } } 0
}

fn quiescence(state: &GameState, mut alpha: i32, beta: i32, info: &mut SearchInfo, ply: usize) -> i32 {
    if ply >= MAX_PLY { return eval::evaluate(state); }
    info.nodes += 1;
    if info.nodes % 2048 == 0 { info.check_time(); }
    if info.stopped { return 0; }

    let stand_pat = eval::evaluate(state);
    if stand_pat >= beta { return beta; }
    if stand_pat > alpha { alpha = stand_pat; }

    let delta = 975; 
    let is_endgame = (state.bitboards[Q].0 | state.bitboards[q].0) == 0;
    if !is_endgame && stand_pat + delta < alpha { return alpha; }

    let mut generator = Box::new(movegen::MoveGenerator::new());
    generator.generate_moves(state);
    let mut scores = [0; 256];
    for i in 0..generator.list.count { scores[i] = score_move(generator.list.moves[i], None, info, 0, state); }
    let in_check = is_in_check(state);

    for i in 0..generator.list.count {
        let mut best_idx = i;
        for j in (i+1)..generator.list.count { if scores[j] > scores[best_idx] { best_idx = j; } }
        scores.swap(i, best_idx);
        let mv = generator.list.moves[best_idx];
        generator.list.moves.swap(i, best_idx);

        let gives_check = moves_gives_check(state, mv);
        if !in_check {
            if !mv.is_capture && mv.promotion.is_none() {
                if !gives_check || ply > 0 { continue; }
            }
            if mv.is_capture && see(state, mv) < 0 { continue; }
        }

        if !is_valid_tt_move(state, mv, info.tt) { continue; }

        let next_state = state.make_move(mv);
        let score = -quiescence(&next_state, -beta, -alpha, info, ply + 1);
        
        if info.stopped { return 0; }
        if score >= beta { return beta; }
        if score > alpha { alpha = score; }
    }
    alpha
}

pub fn search(state: &GameState, tm: TimeManager, tt: &TranspositionTable, stop_signal: Arc<AtomicBool>, max_depth: u8, main_thread: bool, history: Vec<u64>) {
    let mut best_move = None;
    let mut ponder_move = None;
    let mut info = Box::new(SearchInfo::new(tm, stop_signal, tt, main_thread));
    let mut path = history; 
    let mut last_score = 0;
    
    for depth in 1..=max_depth {
        info.seldepth = 0;
        
        let (mut alpha, mut beta) = if depth >= 5 && main_thread {
            let window = 50;
            (last_score - window, last_score + window)
        } else {
            (-INFINITY, INFINITY)
        };
        
        let mut score = negamax(state, depth, alpha, beta, &mut info, 0, true, &mut path);
        
        if main_thread && depth >= 5 {
            if score <= alpha {
                alpha = -INFINITY;
                score = negamax(state, depth, alpha, beta, &mut info, 0, true, &mut path);
            } else if score >= beta {
                beta = INFINITY;
                score = negamax(state, depth, alpha, beta, &mut info, 0, true, &mut path);
            }
        }
        
        last_score = score;
        
        if info.stopped { break; }
        if main_thread && info.time_manager.check_soft_limit() { info.stopped = true; info.stop_signal.store(true, Ordering::Relaxed); }

        if main_thread {
            let elapsed = info.time_manager.start_time.elapsed().as_secs_f64();
            let nps = if elapsed > 0.0 { (info.nodes as f64 / elapsed) as u64 } else { 0 };
            let score_str = if score > MATE_SCORE { format!("mate {}", (MATE_VALUE - score + 1) / 2) } else if score < -MATE_SCORE { format!("mate -{}", (MATE_VALUE + score + 1) / 2) } else { format!("cp {}", score) };

            if let Some(mv) = tt.get_move(state.hash) {
                if is_valid_tt_move(state, mv, tt) {
                    best_move = Some(mv);
                    let (pv_line, p_move) = get_pv_line(state, tt, depth);
                    ponder_move = p_move;
                    println!("info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}", 
                        depth, info.seldepth, score_str, info.nodes, nps, tt.hashfull(), info.time_manager.start_time.elapsed().as_millis(), pv_line
                    );
                }
            }
        }
    }
    
    if main_thread {
        let mut final_move = best_move;
        if let Some(mv) = final_move { if !is_valid_tt_move(state, mv, tt) { final_move = None; } }

        if let Some(bm) = final_move {
            print!("bestmove {}{}", square_to_coord(bm.source), square_to_coord(bm.target));
            if let Some(promo) = bm.promotion { let char = match promo { 4|10=>'q', 3|9=>'r', 2|8=>'b', 1|7=>'n', _=>' ' }; if char != ' ' { print!("{}", char); } }
            if let Some(pm) = ponder_move { print!(" ponder {}{}", square_to_coord(pm.source), square_to_coord(pm.target)); if let Some(promo) = pm.promotion { let char = match promo { 4|10=>'q', 3|9=>'r', 2|8=>'b', 1|7=>'n', _=>' ' }; if char != ' ' { print!("{}", char); } } }
            println!();
        } else {
             let mut generator = Box::new(movegen::MoveGenerator::new()); generator.generate_moves(state); 
             for i in 0..generator.list.count { let mv = generator.list.moves[i]; if is_safe(state, mv) { print!("bestmove {}{}", square_to_coord(mv.source), square_to_coord(mv.target)); if let Some(promo) = mv.promotion { let char = match promo { 4|10=>'q', 3|9=>'r', 2|8=>'b', 1|7=>'n', _=>' ' }; if char != ' ' { print!("{}", char); } } println!(); break; } }
        }
    }
}

fn negamax(state: &GameState, mut depth: u8, mut alpha: i32, mut beta: i32, 
           info: &mut SearchInfo, ply: usize, is_pv: bool, path: &mut Vec<u64>) -> i32 {
    
    if state.halfmove_clock >= 100 { return 0; }
    if ply > 0 && path.iter().any(|&h| h == state.hash) { return 0; }

    let mate_value = MATE_VALUE - (ply as i32);
    if alpha < -mate_value { alpha = -mate_value; }
    if alpha >= beta { return alpha; }

    if ply >= MAX_PLY { return eval::evaluate(state); }
    info.nodes += 1;
    if ply > info.seldepth as usize { info.seldepth = ply as u8; }
    info.check_time();
    if info.stopped { return 0; } 

    let mut tt_move = None;
    if let Some(tm) = info.tt.get_move(state.hash) {
        if is_valid_tt_move(state, tm, info.tt) { tt_move = Some(tm); }
    }
    
    if ply > 0 {
        if let Some(score) = info.tt.probe(state.hash, depth, alpha, beta) {
            if !is_pv || (score > alpha && score < beta) { return score; }
        }
    }

    if depth == 0 { return quiescence(state, alpha, beta, info, ply); }

    let in_check = is_in_check(state);
    
    // --- FUTILITY PRUNING ---
    if depth <= 3 && !in_check && !is_pv && ply > 0 {
        let eval = eval::evaluate(state);
        let margins = [0, 100, 300, 600];
        if eval + margins[depth as usize] < alpha {
             return eval;
        }
    }
    
    // --- RAZORING ---
    if depth <= 2 && !in_check && !is_pv && ply > 0 {
        let eval = eval::evaluate(state);
        if eval + 200 < alpha {
            let q_score = quiescence(state, alpha, beta, info, ply);
            if q_score < alpha { return q_score; }
        }
    }

    if in_check { depth = depth.saturating_add(1); } 

    // --- INTERNAL ITERATIVE DEEPENING (IID) ---
    // If we are in a PV node, have good depth, but NO hash move, 
    // we do a quick shallow search to get a best move to guide sorting.
    if is_pv && tt_move.is_none() && depth > 4 {
        let iid_depth = depth - 2;
        negamax(state, iid_depth, alpha, beta, info, ply, is_pv, path);
        // Try to retrieve the move found by the shallow search
        if let Some(tm) = info.tt.get_move(state.hash) {
             if is_valid_tt_move(state, tm, info.tt) { tt_move = Some(tm); }
        }
    }

    // --- ADAPTIVE NULL MOVE PRUNING ---
    if depth >= 3 && ply > 0 && !in_check && !is_pv && eval::evaluate(state) >= beta {
        let has_pieces = (state.bitboards[N] | state.bitboards[B] | state.bitboards[R] | state.bitboards[Q] |
                          state.bitboards[n] | state.bitboards[b] | state.bitboards[r] | state.bitboards[q]).0 != 0;
        
        if has_pieces {
            let reduction_depth = 3 + depth / 6; 
            let null_state = state.make_null_move();
            let reduced_depth = depth.saturating_sub(reduction_depth as u8);
            let score = -negamax(&null_state, reduced_depth, -beta, -beta + 1, info, ply + 1, false, path);
            if info.stopped { return 0; }
            if score >= beta { return beta; }
        }
    }

    let mut generator = Box::new(MoveGenerator::new());
    generator.generate_moves(state);
    let mut scores = [0; 256];
    for i in 0..generator.list.count { scores[i] = score_move(generator.list.moves[i], tt_move, info, ply, state); }

    let mut max_score = -INFINITY;
    let mut best_move = None;
    let mut moves_searched = 0;

    path.push(state.hash);

    for i in 0..generator.list.count {
        let mut best_idx = i;
        for j in (i+1)..generator.list.count { if scores[j] > scores[best_idx] { best_idx = j; } }
        scores.swap(i, best_idx);
        let mv = generator.list.moves[best_idx];
        generator.list.moves.swap(i, best_idx);

        if !is_valid_tt_move(state, mv, info.tt) { continue; }

        let next_state = state.make_move(mv);
        info.tt.prefetch(next_state.hash);
        moves_searched += 1;

        let mut score;
        if moves_searched == 1 {
            score = -negamax(&next_state, depth - 1, -beta, -alpha, info, ply + 1, true, path);
        } else {
            let is_quiet = !mv.is_capture && mv.promotion.is_none();
            let gives_check = moves_gives_check(state, mv);
            let mut reduction = 0;
            
            // --- OPTIMIZED LMR ---
            if depth >= 3 && moves_searched > 1 && is_quiet && !gives_check && !in_check {
                // Formula: 0.75 + ln(depth) * ln(moves_searched) / 2.25
                let lmr_val = 0.75 + (depth as f32).ln() * (moves_searched as f32).ln() / 2.25;
                reduction = lmr_val as u8;
                // Clamp reduction to ensure we don't reduce below depth 1
                reduction = reduction.min(depth - 1);
                
                // Reduce LMR for Killers (optional tweak)
                if ply < MAX_PLY && (info.killers[ply][0] == Some(mv) || info.killers[ply][1] == Some(mv)) {
                    reduction = reduction.saturating_sub(1);
                }
            }
            
            let d = depth.saturating_sub(1 + reduction);
            score = -negamax(&next_state, d, -alpha - 1, -alpha, info, ply + 1, false, path);
            
            if score > alpha && reduction > 0 { 
                score = -negamax(&next_state, depth - 1, -alpha - 1, -alpha, info, ply + 1, false, path); 
            }
            if score > alpha && score < beta { 
                score = -negamax(&next_state, depth - 1, -beta, -alpha, info, ply + 1, true, path); 
            }
        }

        if info.stopped { path.pop(); return 0; }

        if score > max_score {
            max_score = score;
            best_move = Some(mv);
            if score > alpha {
                alpha = score;
                if !mv.is_capture && mv.promotion.is_none() {
                    let from = mv.source as usize; let to = mv.target as usize;
                    let bonus = (depth as i32) * (depth as i32);
                    info.history[from][to] += bonus;
                    if info.history[from][to] > 25000 { // Increased cap
                        for f in 0..64 { for t in 0..64 { info.history[f][t] /= 2; } } 
                    }
                }
            }
        }
        if alpha >= beta {
            if !mv.is_capture && mv.promotion.is_none() {
                if ply < MAX_PLY { info.killers[ply][1] = info.killers[ply][0]; info.killers[ply][0] = Some(mv); }
            }
            break;
        }
    }

    path.pop();

    if moves_searched == 0 { if in_check { return -MATE_VALUE + (ply as i32); } else { return 0; } }

    let flag = if max_score <= alpha { FLAG_ALPHA } else if max_score >= beta { FLAG_BETA } else { FLAG_EXACT };
    info.tt.store(state.hash, max_score, best_move, depth, flag);
    max_score
}

fn is_in_check(state: &GameState) -> bool {
    let king_type = if state.side_to_move == WHITE { K } else { k };
    let king_sq = state.bitboards[king_type].get_lsb_index() as u8;
    movegen::is_square_attacked(state, king_sq, 1 - state.side_to_move)
}

fn moves_gives_check(state: &GameState, mv: Move) -> bool {
    let next_state = state.make_move(mv);
    is_in_check(&next_state)
}

fn square_to_coord(s: u8) -> String {
    let f = (b'a' + (s % 8)) as char;
    let rank_char = (b'1' + (s / 8)) as char;
    format!("{}{}", f, rank_char)
}