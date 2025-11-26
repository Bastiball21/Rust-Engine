use crate::state::{GameState, Move, WHITE, BLACK, K, k};
use crate::movegen::{self, MoveGenerator};
use crate::eval;
use crate::tt::{TranspositionTable, FLAG_EXACT, FLAG_ALPHA, FLAG_BETA}; 
use crate::time::TimeManager;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MAX_PLY: usize = 64; 
const FUTILITY_MARGIN: [i32; 4] = [0, 200, 350, 500];
const RAZORING_MARGIN: i32 = 400; 

pub struct SearchInfo {
    pub killers: [[Option<Move>; 2]; MAX_PLY + 1],  
    pub history: [[i32; 64]; 12],
    pub counter_moves: Vec<Option<Move>>, 
    pub nodes: u64,
    pub seldepth: u8,
    pub time_manager: TimeManager,
    pub stop_signal: Arc<AtomicBool>, 
    pub stopped: bool,
}

impl SearchInfo {
    pub fn new(tm: TimeManager, stop: Arc<AtomicBool>) -> Self {
        Self { 
            killers: [[None; 2]; MAX_PLY + 1], 
            history: [[0; 64]; 12], 
            counter_moves: vec![None; 64 * 64],
            nodes: 0, 
            seldepth: 0, 
            time_manager: tm,
            stop_signal: stop,
            stopped: false,
        }
    }
    
    #[inline(always)]
    pub fn check_time(&mut self) {
        if self.nodes % 2048 == 0 {
            if self.stop_signal.load(Ordering::Relaxed) {
                self.stopped = true;
                return;
            }
            if self.time_manager.check_hard_limit() {
                self.stopped = true;
            }
        }
    }
    
    #[inline(always)]
    pub fn should_stop(&self) -> bool {
        self.stopped
    }
}

// --- CRITICAL FIX: Check if a move is legal (doesn't leave king in check) ---
fn is_legal(state: &GameState, move_to_make: Move) -> bool {
    // Make the move on a temporary board
    let next_state = state.make_move(move_to_make);
    
    // Identify "Our" King (the side that moved)
    let our_side = state.side_to_move;
    let our_king_type = if our_side == WHITE { K } else { k };
    
    // If King is gone (captured), it's illegal (though this shouldn't happen in standard chess logic)
    if next_state.bitboards[our_king_type].0 == 0 { return false; }
    
    let our_king_sq = next_state.bitboards[our_king_type].get_lsb_index() as u8;
    
    // Check if the King is attacked by the opponent (who is now next_state.side_to_move)
    // If attacked, the move was illegal (self-check)
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
            if seen_hashes.contains(&curr_state.hash) { break; }
            seen_hashes.push(curr_state.hash);
            pv_str.push_str(&format!("{}{} ", square_to_coord(mv.source), square_to_coord(mv.target)));
            if !first && ponder_move.is_none() { ponder_move = Some(mv); }
            first = false;
            
            // Safety check for PV line display
            if is_legal(&curr_state, mv) {
                curr_state = curr_state.make_move(mv);
            } else {
                break;
            }
        } else { break; }
    }
    (pv_str, ponder_move)
}

fn score_move(mv: Move, tt_move: Option<Move>, info: &SearchInfo, ply: usize, state: &GameState, prev_move: Option<Move>) -> i32 {
    if let Some(tm) = tt_move { if mv == tm { return 30000; } }
    if mv.is_capture {
        let attacker = get_piece_type_safe(state, mv.source) % 6;
        let victim = get_victim_type(state, mv.target) % 6;
        let mvv_lva = [
            [105, 205, 305, 405, 505, 605], [104, 204, 304, 404, 504, 604],
            [103, 203, 303, 403, 503, 603], [102, 202, 302, 402, 502, 602],
            [101, 201, 301, 401, 501, 601], [100, 200, 300, 400, 500, 600],
        ];
        return 20000 + mvv_lva[victim][attacker];
    }
    if mv.promotion.is_some() { return 15000; }
    if ply < MAX_PLY {
        if let Some(k1) = info.killers[ply][0] { if mv == k1 { return 9000; } }
        if let Some(k2) = info.killers[ply][1] { if mv == k2 { return 8000; } }
    }
    let piece = get_piece_type_safe(state, mv.source);
    let hist = info.history[piece][mv.target as usize];
    if hist > 5000 { 5000 } else { hist }
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
    info.check_time();
    if info.should_stop() { return 0; }

    let stand_pat = eval::evaluate(state);
    if stand_pat >= beta { return beta; }
    if stand_pat > alpha { alpha = stand_pat; }

    let mut generator = Box::new(MoveGenerator::new());
    generator.generate_moves(state);
    let mut scores = [0; 256];
    for i in 0..generator.list.count { scores[i] = score_move(generator.list.moves[i], None, info, 0, state, None); }
    
    for i in 0..generator.list.count {
        let mut best_idx = i;
        for j in (i+1)..generator.list.count { if scores[j] > scores[best_idx] { best_idx = j; } }
        scores.swap(i, best_idx);
        let mv = generator.list.moves[best_idx];
        generator.list.moves.swap(i, best_idx);

        if !mv.is_capture && mv.promotion.is_none() { continue; }
        
        // --- LEGALITY CHECK ---
        if !is_legal(state, mv) { continue; }
        // ----------------------

        let next_state = state.make_move(mv);
        
        let score = -quiescence(&next_state, -beta, -alpha, info, ply + 1);
        if info.should_stop() { return 0; }
        if score >= beta { return beta; }
        if score > alpha { alpha = score; }
    }
    alpha
}

pub fn search(state: &GameState, tm: TimeManager, tt: &mut TranspositionTable, stop_signal: Arc<AtomicBool>, max_depth: u8) {
    let mut best_move = None;
    let mut ponder_move = None;
    let mut info = Box::new(SearchInfo::new(tm, stop_signal));
    
    let alpha = -50000;
    let beta = 50000;
    
    for depth in 1..=max_depth {
        info.seldepth = 0;
        let score = negamax(state, depth, alpha, beta, tt, &mut info, 0, None, true);
        
        if info.should_stop() { break; }

        if info.time_manager.check_soft_limit() {
            info.stopped = true; 
        }

        let elapsed = info.time_manager.start_time.elapsed().as_secs_f64();
        let nps = if elapsed > 0.0 { (info.nodes as f64 / elapsed) as u64 } else { 0 };
        
        if let Some(mv) = tt.get_move(state.hash) {
            best_move = Some(mv);
            let (pv_line, p_move) = get_pv_line(state, tt, depth);
            ponder_move = p_move;
            println!("info depth {} seldepth {} multipv 1 score cp {} nodes {} nps {} hashfull {} tbhits 0 time {} pv {}", 
                depth, info.seldepth, score, info.nodes, nps, tt.hashfull(), info.time_manager.start_time.elapsed().as_millis(), pv_line
            );
        }
    }
    
    if let Some(bm) = best_move {
        print!("bestmove {}{}", square_to_coord(bm.source), square_to_coord(bm.target));
        if let Some(promo) = bm.promotion {
            let char = match promo { 4|10=>'q', 3|9=>'r', 2|8=>'b', 1|7=>'n', _=>' ' };
            if char != ' ' { print!("{}", char); }
        }
        if let Some(pm) = ponder_move {
             print!(" ponder {}{}", square_to_coord(pm.source), square_to_coord(pm.target));
             if let Some(promo) = pm.promotion {
                let char = match promo { 4|10=>'q', 3|9=>'r', 2|8=>'b', 1|7=>'n', _=>' ' };
                if char != ' ' { print!("{}", char); }
             }
        }
        println!();
    } else {
        println!("bestmove a1a1");
    }
}

fn negamax(state: &GameState, mut depth: u8, mut alpha: i32, beta: i32, 
           tt: &mut TranspositionTable, info: &mut SearchInfo, ply: usize, prev_move: Option<Move>, is_pv: bool) -> i32 {
    if ply >= MAX_PLY { return eval::evaluate(state); }
    info.nodes += 1;
    if ply > info.seldepth as usize { info.seldepth = ply as u8; }
    info.check_time();
    if info.should_stop() { return 0; } 

    let tt_hit = tt.probe(state.hash, depth, alpha, beta);
    if ply > 0 {
        if let Some(score) = tt_hit { 
            if !is_pv || (score > alpha && score < beta) { return score; }
        }
    }
    
    if depth == 0 { return quiescence(state, alpha, beta, info, ply); }

    let in_check = is_in_check(state);
    if in_check && depth < 255 { depth += 1; }

    let static_eval = eval::evaluate(state);

    if !is_pv && !in_check && depth <= 3 {
        if static_eval + RAZORING_MARGIN + (depth as i32 * 50) < alpha {
            let score = quiescence(state, alpha, beta, info, ply);
            if score <= alpha { return score; }
        }
    }

    if depth >= 3 && ply > 0 && !in_check && !is_pv && static_eval >= beta {
        let r = 3 + (depth / 6);
        let null_state = state.make_null_move();
        let score = -negamax(&null_state, depth.saturating_sub(r), -beta, -beta + 1, tt, info, ply + 1, None, false);
        if info.should_stop() { return 0; }
        if score >= beta { return beta; }
    }

    if is_pv && tt_hit.is_none() && depth > 4 { depth -= 1; }

    let mut futility_prune = false;
    if depth <= 3 && !in_check && !is_pv && alpha.abs() < 20000 && beta.abs() < 20000 {
        if static_eval + FUTILITY_MARGIN[depth as usize] <= alpha { futility_prune = true; }
    }

    let mut generator = Box::new(MoveGenerator::new());
    generator.generate_moves(state);
    let mut max_score = -50000;
    let mut best_move = None;
    let mut moves_searched = 0;
    let mut skipped_moves = false;
    let tt_move = tt.get_move(state.hash);

    let mut scores = [0; 256];
    for i in 0..generator.list.count { scores[i] = score_move(generator.list.moves[i], tt_move, info, ply, state, prev_move); }

    for i in 0..generator.list.count {
        let mut best_idx = i;
        for j in (i+1)..generator.list.count { if scores[j] > scores[best_idx] { best_idx = j; } }
        scores.swap(i, best_idx);
        let mv = generator.list.moves[best_idx];
        generator.list.moves.swap(i, best_idx);
        
        let is_quiet = !mv.is_capture && mv.promotion.is_none();
        if futility_prune && is_quiet && scores[i] < 10000 { skipped_moves = true; continue; }

        // --- CORE LEGALITY CHECK ---
        if !is_legal(state, mv) { continue; }
        // ---------------------------

        let next_state = state.make_move(mv);
        tt.prefetch(next_state.hash);
        
        moves_searched += 1;

        let mut score;
        if moves_searched == 1 {
            score = -negamax(&next_state, depth - 1, -beta, -alpha, tt, info, ply + 1, Some(mv), true);
        } else {
            let mut reduction = 0;
            if depth >= 3 && moves_searched > 1 && is_quiet {
                let ld = (depth as f32).ln();
                let lm = (moves_searched as f32).ln();
                reduction = (0.77 + (ld * lm) / 2.6).round() as u8;
                if info.killers[ply][0] == Some(mv) || info.killers[ply][1] == Some(mv) { reduction = reduction.saturating_sub(1); }
            }
            let d = depth.saturating_sub(1 + reduction);
            score = -negamax(&next_state, d, -alpha - 1, -alpha, tt, info, ply + 1, Some(mv), false);
            if score > alpha && reduction > 0 {
                score = -negamax(&next_state, depth - 1, -alpha - 1, -alpha, tt, info, ply + 1, Some(mv), false);
            }
            if score > alpha && score < beta {
                score = -negamax(&next_state, depth - 1, -beta, -alpha, tt, info, ply + 1, Some(mv), true);
            }
        }

        if info.should_stop() { return 0; }
        if score > max_score {
            max_score = score;
            best_move = Some(mv);
            if score > alpha {
                alpha = score;
                if is_quiet {
                    let p_type = get_piece_type_safe(state, mv.source);
                    info.history[p_type][mv.target as usize] += (depth * depth) as i32;
                    if info.history[p_type][mv.target as usize] > 10000 { info.history[p_type][mv.target as usize] /= 2; }
                }
            }
        }
        if alpha >= beta { 
            if is_quiet {
                if ply < MAX_PLY { 
                    info.killers[ply][1] = info.killers[ply][0]; 
                    info.killers[ply][0] = Some(mv); 
                }
            }
            break; 
        }
    }

    // --- STALEMATE / CHECKMATE DETECTION ---
    if moves_searched == 0 {
        if skipped_moves { return alpha; } // Fail low if we pruned possible moves
        
        if in_check {
            return -49000 + (ply as i32); // CHECKMATE (High negative score)
        } else {
            return 0; // STALEMATE (Draw)
        }
    }
    // ---------------------------------------

    let flag = if max_score <= alpha { FLAG_ALPHA } else if max_score >= beta { FLAG_BETA } else { FLAG_EXACT };
    tt.store(state.hash, max_score, best_move, depth, flag);
    max_score
}

fn is_in_check(state: &GameState) -> bool {
    let king_type = if state.side_to_move == WHITE { K } else { k };
    if state.bitboards[king_type].0 == 0 { return true; } // Should not happen
    let king_sq = state.bitboards[king_type].get_lsb_index() as u8;
    movegen::is_square_attacked(state, king_sq, 1 - state.side_to_move)
}

fn square_to_coord(s: u8) -> String {
    let f = (b'a' + (s % 8)) as char;
    let r = (b'1' + (s / 8)) as char;
    format!("{}{}", f, r)
}