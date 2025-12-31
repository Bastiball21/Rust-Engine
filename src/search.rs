// src/search.rs
use crate::bitboard::{self, Bitboard};
use crate::eval;
use crate::syzygy;
use crate::movegen::{self, GenType, MoveGenerator, MAX_MOVES};
use crate::state::{b, k, n, p, q, r, GameState, Move, B, BLACK, BOTH, K, N, NO_PIECE, P, Q, R, WHITE};
use crate::threat::{self, ThreatDeltaScore, ThreatInfo, MoveTag};
use crate::time::TimeManager;
use crate::tt::{TranspositionTable, FLAG_ALPHA, FLAG_BETA, FLAG_EXACT};
use crate::parameters::SearchParameters;
use crate::nnue::Accumulator;
use crate::nnue_scratch::NNUEScratch;
use crate::history::ContinuationHistory;
use crate::correction::CorrectionHistory;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicI16, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use smallvec::SmallVec;

// --- FEATURE TOGGLES ---
pub const ENABLE_ASPIRATION: bool = true;
pub const ENABLE_NULL_MOVE: bool = true;
pub const ENABLE_LMR: bool = true;
pub const ENABLE_LMP: bool = true;
pub const ENABLE_SEE_GATE_MAIN: bool = true;
pub const ENABLE_SEE_GATE_QS: bool = true;
pub const ENABLE_IIR: bool = true;


// --- SEARCH MODES (Play vs Datagen) ---
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    Play,
    Datagen,
}

#[derive(Clone, Copy, Debug)]
pub struct SearchTuning {
    pub allow_nullmove: bool,
    pub null_min_depth: u8,
    pub lmr_min_depth: u8,
    pub lmr_min_move_index: usize,
    pub enable_lmp: bool,
    pub enable_futility: bool,
    pub enable_history_pruning: bool,
    pub qsearch_include_checks: bool,
    pub qsearch_delta_dynamic: bool,
}

pub const TUNING_PLAY: SearchTuning = SearchTuning {
    allow_nullmove: true,
    null_min_depth: 3,
    lmr_min_depth: 3,
    lmr_min_move_index: 5,
    enable_lmp: true,
    enable_futility: true,
    enable_history_pruning: true,
    qsearch_include_checks: true,
    qsearch_delta_dynamic: true,
};

pub const TUNING_DATAGEN: SearchTuning = SearchTuning {
    allow_nullmove: false,
    null_min_depth: 99,
    lmr_min_depth: 4,
    lmr_min_move_index: 8,
    enable_lmp: false,
    enable_futility: false,
    enable_history_pruning: false,
    qsearch_include_checks: true,
    qsearch_delta_dynamic: true,
};

impl Default for SearchTuning {
    fn default() -> Self {
        TUNING_PLAY
    }
}

// --- TUNING CONSTANTS ---
pub const ASP_WINDOW_CP: i32 = 50;
pub const ASP_WIDEN_1: i32 = 150;
pub const ASP_WIDEN_2: i32 = 500;

pub const LMR_MIN_DEPTH: u8 = 3;
pub const LMR_MIN_MOVE_INDEX: usize = 4;

pub const LMP_DEPTH_MAX: u8 = 3;

pub const FUTILITY_MARGIN_PER_PLY: i32 = 100;

pub const NULL_MIN_DEPTH: u8 = 3;

pub const MAX_PLY: usize = 128;
pub const STACK_SIZE: usize = MAX_PLY + 10;

// Max game length we support in search path
const MAX_GAME_PLY: usize = 1024;
const INFINITY: i32 = 32000;
const MATE_VALUE: i32 = 31000;
const MATE_SCORE: i32 = 30000;

pub const WINNING_CAPTURE_BONUS: i32 = 10_000_000;
pub const MIN_WINNING_SEE_SCORE: i32 = WINNING_CAPTURE_BONUS - 16384;

static START_PLY: [i16; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];
static SKIP_SIZE: [i16; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];

#[derive(Clone, Copy)]
pub struct StackEntry {
    pub current_move: Move,   // Use Move::default() if none
    pub excluded_move: Move,
    pub killers: [Move; 2],
    pub static_eval: i32,
    pub move_count: usize,

    // CACHED CONTEXT (Must be populated BEFORE recursion)
    // These define the context of the move stored in `current_move`.
    pub moved_piece: usize, // Must be 0..11 (White: 0-5, Black: 6-11)
    pub to_sq: usize,       // 0..63
    pub is_capture: bool,
    pub in_check: bool,     // Context: Side-to-move is in check at this node (bucket key)
    pub stat_score: i32,
}

impl Default for StackEntry {
    fn default() -> Self {
        Self {
            current_move: Move::default(),
            excluded_move: Move::default(),
            killers: [Move::default(); 2],
            static_eval: 0,
            move_count: 0,
            moved_piece: 0,
            to_sq: 0,
            is_capture: false,
            in_check: false,
            stat_score: 0,
        }
    }
}

#[derive(Clone, Copy)]
pub enum Limits {
    Infinite,
    FixedDepth(u8),
    FixedNodes(u64),
    FixedTime(TimeManager),
    Smart { node_limit: u64, min_depth: u8 },
}

pub struct SearchPath {
    pub keys: [u64; MAX_GAME_PLY],
    pub len: usize,
}

impl SearchPath {
    pub fn new() -> Self {
        Self {
            keys: [0; MAX_GAME_PLY],
            len: 0,
        }
    }

    pub fn push(&mut self, key: u64) {
        if self.len < MAX_GAME_PLY {
            self.keys[self.len] = key;
            self.len += 1;
        }
    }

    pub fn pop(&mut self) {
        if self.len > 0 {
            self.len -= 1;
        }
    }

    pub fn load_from(&mut self, history: &[u64]) {
        self.len = 0;
        for &key in history {
            self.push(key);
        }
    }
}

pub struct SearchData {
    // Note: 'stack' is now allocated by caller (e.g., in SearchThread) and passed to search.

    // Global (thread-local) History Tables
    pub history: [[i32; 64]; 64],
    pub capture_history: Box<[[[i32; 6]; 64]; 12]>,

    // [Piece][ToSquare] -> Best Reply Move
    pub counter_moves: [[Option<Move>; 64]; 12],

    // New Deep Continuation History
    pub cont_history: ContinuationHistory,

    // Correction History (Thread Local): fixes systematic eval bias for pruning
    pub correction_history: CorrectionHistory,

    // NNUE Accumulators (Thread Local)
    pub accumulators: [Accumulator; 2],

    // NNUE Scratch (Thread Local)
    pub nnue_scratch: Box<NNUEScratch>,
}

impl SearchData {
    pub fn new() -> Self {
        Self {
            history: [[0; 64]; 64],
            capture_history: Box::new([[[0; 6]; 64]; 12]),
            counter_moves: [[None; 64]; 12],
            cont_history: ContinuationHistory::new(),
            correction_history: CorrectionHistory::new(),
            accumulators: [Accumulator::default(); 2],
            nnue_scratch: Box::new(NNUEScratch::default()),
        }
    }

    pub fn clear(&mut self) {
        self.history = [[0; 64]; 64];
        self.capture_history.fill_with(|| [[0; 6]; 64]);
        self.counter_moves = [[None; 64]; 12];
        self.cont_history.clear();
        self.correction_history.clear();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MovePickerStage {
    TtMove,
    GenerateCaptures,
    YieldGoodCaptures,
    Killers,
    GenerateQuiets,
    YieldRemaining,
    Done,
}

pub struct MovePicker<'a> {
    stage: MovePickerStage,
    tt_move: Option<Move>,
    killers: [Move; 2], // Direct Moves, Move::default() is None
    counter_move: Move, // Direct Move, Move::default() is None
    move_list: [Move; MAX_MOVES],
    move_scores: [i32; MAX_MOVES],
    move_count: usize,
    move_index: usize,
    tt: &'a TranspositionTable,
    captures_only: bool,
    thread_id: Option<usize>,
    params: Option<&'a SearchParameters>,
    is_pv_node: bool,
    ply: usize,
}

impl<'a> MovePicker<'a> {
    pub fn new(
        data: &SearchData,
        tt: &'a TranspositionTable,
        state: &GameState,
        stack: &[StackEntry], // Pass stack slice
        ply: usize,
        tt_move: Option<Move>,
        captures_only: bool,
        thread_id: Option<usize>,
        params: Option<&'a SearchParameters>,
        is_pv_node: bool,
    ) -> Self {
        let mut killers = [Move::default(); 2];
        let mut counter_move = Move::default();

        if !captures_only && ply < MAX_PLY {
            killers = stack[ply].killers;

            // Access previous move from stack.
            // stack[ply] holds the move that led TO this node?
            // No, stack[ply] is CURRENT node data.
            // stack[ply-1] is PREVIOUS node data (move that led here).
            // Wait, logic in negamax: "info.data.stack[next_ply].current_move = mv;"
            // So stack[ply] holds context of the move we are *currently* exploring/generating from?
            // "stack[ply] holds the move that led TO this node." -> This assumes we populated it in previous negamax call.
            // Yes: negamax(..., ply) writes to stack[ply+1] before recursing to ply+1.
            // So stack[ply].current_move is the move that brought us to 'state'.

            if ply > 0 {
                let prev = &stack[ply];
                // prev.current_move is the move that created 'state'.
                if prev.current_move != Move::default() {
                    if let Some(cm) = data.counter_moves[prev.moved_piece][prev.to_sq] {
                        counter_move = cm;
                    }
                }
            }
        }

        Self {
            stage: if tt_move.is_some() {
                MovePickerStage::TtMove
            } else {
                MovePickerStage::GenerateCaptures
            },
            tt_move,
            killers,
            counter_move,
            move_list: [Move::default(); MAX_MOVES],
            move_scores: [0; MAX_MOVES],
            move_count: 0,
            move_index: 0,
            tt,
            captures_only,
            thread_id,
            params,
            is_pv_node,
            ply,
        }
    }

    pub fn next_move(&mut self, state: &GameState, data: &SearchData, stack: &[StackEntry]) -> Option<Move> {
        loop {
            match self.stage {
                MovePickerStage::TtMove => {
                    self.stage = MovePickerStage::GenerateCaptures;
                    if let Some(mv) = self.tt_move {
                        // TT Move Verification
                        let from = mv.source();
                        let piece_on_src = get_piece_type_safe(state, from);
                        if piece_on_src == NO_PIECE {
                            continue;
                        }

                        let side = state.side_to_move;
                        if side == WHITE {
                            if piece_on_src > 5 { continue; }
                        } else {
                            if piece_on_src < 6 { continue; }
                        }

                        if self.tt.is_pseudo_legal(state, mv) {
                            if mv.is_capture() {
                                let target_piece = get_piece_type_safe(state, mv.target());
                                if target_piece == NO_PIECE {
                                    let piece = piece_on_src;
                                    let is_ep = mv.target() == state.en_passant
                                        && (piece == P || piece == p);
                                    if !is_ep {
                                        continue;
                                    }
                                }
                            } else {
                                let target_piece = get_piece_type_safe(state, mv.target());
                                if target_piece != NO_PIECE {
                                    continue;
                                }
                            }
                            return Some(mv);
                        }
                    }
                }
                MovePickerStage::GenerateCaptures => {
                    self.generate_moves(state, GenType::Captures);
                    self.score_captures(state, data);
                    self.stage = MovePickerStage::YieldGoodCaptures;
                }
                MovePickerStage::YieldGoodCaptures => {
                    loop {
                        let mv = match self.pick_best_move() {
                            Some(m) => m,
                            None => break, // No more moves
                        };

                        if self.move_scores[self.move_index - 1] < MIN_WINNING_SEE_SCORE {
                             self.move_index -= 1;
                             break;
                        }

                        if Some(mv) == self.tt_move {
                             continue;
                        }

                        if !see_ge(state, mv, 0) {
                            self.move_scores[self.move_index - 1] -= WINNING_CAPTURE_BONUS;
                            self.move_index -= 1;
                            continue;
                        }

                        return Some(mv);
                    }

                    self.stage = if self.captures_only {
                        MovePickerStage::YieldRemaining
                    } else {
                        MovePickerStage::Killers
                    };
                }
                MovePickerStage::Killers => {
                    self.stage = MovePickerStage::GenerateQuiets;
                    let k1 = self.killers[0];
                    if k1 != Move::default() {
                        if Some(k1) != self.tt_move && self.tt.is_pseudo_legal(state, k1) && !k1.is_capture() {
                             return Some(k1);
                        }
                    }
                    let k2 = self.killers[1];
                    if k2 != Move::default() {
                        if Some(k2) != self.tt_move && k2 != k1 && self.tt.is_pseudo_legal(state, k2) && !k2.is_capture() {
                             return Some(k2);
                        }
                    }
                    let cm = self.counter_move;
                    if cm != Move::default() {
                         if Some(cm) != self.tt_move && cm != k1 && cm != k2 && self.tt.is_pseudo_legal(state, cm) && !cm.is_capture() {
                             return Some(cm);
                         }
                    }
                }
                MovePickerStage::GenerateQuiets => {
                    let start = self.move_count;
                    self.generate_moves_append(state, GenType::Quiets);
                    self.score_quiets(state, data, stack, start);
                    self.stage = MovePickerStage::YieldRemaining;
                }
                MovePickerStage::YieldRemaining => {
                    while self.move_index < self.move_count {
                        let mv = match self.pick_best_move() {
                             Some(m) => m,
                             None => break,
                        };

                        if Some(mv) == self.tt_move
                           || (!self.captures_only && (mv == self.killers[0] || mv == self.killers[1] || mv == self.counter_move))
                        {
                            continue;
                        }
                        return Some(mv);
                    }
                    self.stage = MovePickerStage::Done;
                }
                MovePickerStage::Done => {
                    return None;
                }
            }
        }
    }

    fn generate_moves(&mut self, state: &GameState, gen_type: GenType) {
        let mut generator = MoveGenerator::new();
        generator.generate_moves_type(state, gen_type);
        self.move_list = generator.list.moves;
        self.move_count = generator.list.count;
        self.move_index = 0;
    }

    fn generate_moves_append(&mut self, state: &GameState, gen_type: GenType) {
         let mut generator = MoveGenerator::new();
         generator.generate_moves_type(state, gen_type);
         for i in 0..generator.list.count {
             if self.move_count < MAX_MOVES {
                 self.move_list[self.move_count] = generator.list.moves[i];
                 self.move_count += 1;
             }
         }
    }

    fn pick_best_move(&mut self) -> Option<Move> {
        if self.move_index >= self.move_count {
            return None;
        }

        let mut best_score = -INFINITY;
        let mut best_idx = self.move_index;

        for i in self.move_index..self.move_count {
            if self.move_scores[i] > best_score {
                best_score = self.move_scores[i];
                best_idx = i;
            }
        }

        if best_idx != self.move_index {
            self.move_list.swap(self.move_index, best_idx);
            self.move_scores.swap(self.move_index, best_idx);
        }

        let mv = self.move_list[self.move_index];
        self.move_index += 1;
        Some(mv)
    }

    fn score_captures(&mut self, state: &GameState, data: &SearchData) {
        let mvv_lva = [
            [105, 104, 103, 102, 101, 100],
            [205, 204, 203, 202, 201, 200],
            [305, 304, 303, 302, 301, 300],
            [405, 404, 403, 402, 401, 400],
            [505, 504, 503, 502, 501, 500],
            [605, 604, 603, 602, 601, 600],
        ];

        for i in 0..self.move_count {
            let mv = self.move_list[i];
            let attacker = get_piece_type_safe(state, mv.source());
            let victim = get_victim_type(state, mv.target());

            let mut score = WINNING_CAPTURE_BONUS;

            if victim < 12 {
                 score += mvv_lva[victim % 6][attacker % 6];
            } else {
                 score += 100;
            }

            if attacker < 12 && victim < 12 {
                score += data.capture_history[attacker][mv.target() as usize][victim % 6] / 16;
            }
            self.move_scores[i] = score;
        }
    }

    fn score_quiets(&mut self, state: &GameState, data: &SearchData, stack: &[StackEntry], start_idx: usize) {
        let us = state.side_to_move;
        let enemy_pawns = state.bitboards[if us == WHITE { p } else { P }];
        let enemy_pawn_attacks = bitboard::pawn_attacks(enemy_pawns, 1 - us);

        let jitter_base = if let Some(tid) = self.thread_id {
            if tid > 0 {
                state.hash.wrapping_add(tid as u64)
            } else { 0 }
        } else { 0 };

        for i in start_idx..self.move_count {
            let mv = self.move_list[i];
            let from = mv.source() as usize;
            let to = mv.target() as usize;
            let piece = get_piece_type_safe(state, mv.source());

            let mut score = data.history[from][to];

            // Deep Continuation History Logic
            // stack[ply] is current node (context of prev move).
            // So stack[ply] corresponds to "1 ply back" relative to the move we are picking now.

            // Deep Continuation History Logic
            // stack[ply] holds the move that led to the current position (1-ply back).
            // stack[ply-1] holds 2-ply back, etc.

            if self.ply < stack.len() {
                // 1-ply back
                let prev = &stack[self.ply];
                if prev.current_move != Move::default() {
                     score += data.cont_history.get(prev.in_check, prev.is_capture, prev.current_move.source() as usize, prev.current_move.target() as usize, from, to) as i32;
                }

                // 2-ply back
                if self.ply >= 1 {
                    let prev2 = &stack[self.ply - 1];
                    if prev2.current_move != Move::default() {
                        score += data.cont_history.get(prev2.in_check, prev2.is_capture, prev2.current_move.source() as usize, prev2.current_move.target() as usize, from, to) as i32;
                    }
                }

                // 4-ply back
                if self.ply >= 3 {
                    let prev4 = &stack[self.ply - 3];
                    if prev4.current_move != Move::default() {
                        score += data.cont_history.get(prev4.in_check, prev4.is_capture, prev4.current_move.source() as usize, prev4.current_move.target() as usize, from, to) as i32;
                    }
                }

                 // 6-ply back
                 if self.ply >= 5 {
                    let prev6 = &stack[self.ply - 5];
                    if prev6.current_move != Move::default() {
                        score += data.cont_history.get(prev6.in_check, prev6.is_capture, prev6.current_move.source() as usize, prev6.current_move.target() as usize, from, to) as i32;
                    }
                }
            }

            if enemy_pawn_attacks.get_bit(from as u8) {
                score += 500;
            }
            if enemy_pawn_attacks.get_bit(to as u8) {
                score -= 1000;
            }

            if jitter_base != 0 {
                let noise = ((jitter_base.wrapping_add(mv.0 as u64) % 32) as i32) - 16;
                score += noise;
            }

            self.move_scores[i] = score;
        }
    }
}

pub struct SearchInfo<'a> {
    pub data: &'a mut SearchData,
    pub nodes: u64,
    pub global_nodes: Option<&'a AtomicU64>,
    pub seldepth: u8,
    pub root_depth: u8,
    pub limits: Limits,
    pub stop_signal: Arc<AtomicBool>,
    pub stopped: bool,
    pub tt: &'a TranspositionTable,
    pub main_thread: bool,
    pub params: &'a SearchParameters,
    pub thread_id: Option<usize>,
    pub tuning: SearchTuning,
    pub path: SearchPath,
    pub tt_hit_avg: i32,
}

impl<'a> SearchInfo<'a> {
    pub fn new(
        data: &'a mut SearchData,
        limits: Limits,
        stop: Arc<AtomicBool>,
        tt: &'a TranspositionTable,
        main: bool,
        params: &'a SearchParameters,
        global_nodes: Option<&'a AtomicU64>,
        thread_id: Option<usize>,
        mode: SearchMode,
    ) -> Self {
        Self {
            data,
            nodes: 0,
            global_nodes,
            seldepth: 0,
            root_depth: 0,
            limits,
            stop_signal: stop,
            stopped: false,
            tt,
            main_thread: main,
            params,
            thread_id,
            tuning: match mode { SearchMode::Datagen => TUNING_DATAGEN, SearchMode::Play => TUNING_PLAY },
            path: SearchPath::new(),
            tt_hit_avg: 0,
        }
    }

    #[inline(always)]
    pub fn check_time(&mut self) {
        if (self.nodes & 1023) == 0 {
            if let Some(gn) = self.global_nodes {
                gn.fetch_add(1024, Ordering::Relaxed);
            }

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
                Limits::Smart { node_limit, min_depth } => {
                    if self.nodes >= *node_limit && self.root_depth >= *min_depth {
                        self.stopped = true;
                        self.stop_signal.store(true, Ordering::Relaxed);
                    }
                }
                Limits::FixedDepth(_) | Limits::Infinite => {}
            }
        }
    }
}

// ... helpers ...
fn gives_check_fast(state: &GameState, mv: Move) -> bool {
    let side = state.side_to_move;
    let enemy = 1 - side;
    let enemy_king_sq = state.bitboards[if enemy == WHITE { K } else { k }].get_lsb_index() as u8;

    let mut piece = get_piece_type_safe(state, mv.source());
    if let Some(p_promo) = mv.promotion() {
        piece = if side == WHITE { p_promo } else { p_promo + 6 };
    }

    if piece == K || piece == k {
        let friendly_rooks = state.bitboards[if side == WHITE { R } else { r }];
        if friendly_rooks.get_bit(mv.target()) {
            return false;
        }
    }

    let attacks = match piece {
        1 | 7 => crate::bitboard::get_knight_attacks(mv.target()),
        0 => bitboard::pawn_attacks(Bitboard(1 << mv.target()), WHITE),
        6 => bitboard::pawn_attacks(Bitboard(1 << mv.target()), BLACK),
        _ => {
            let occ = state.occupancies[crate::state::BOTH];
            let occ_adjusted = (occ.0 & !(1u64 << mv.source())) | (1u64 << mv.target());
            match piece {
                2 | 8 => bitboard::get_bishop_attacks(mv.target(), Bitboard(occ_adjusted)),
                3 | 9 => bitboard::get_rook_attacks(mv.target(), Bitboard(occ_adjusted)),
                4 | 10 => bitboard::get_queen_attacks(mv.target(), Bitboard(occ_adjusted)),
                _ => Bitboard(0),
            }
        }
    };

    if attacks.get_bit(enemy_king_sq) {
        return true;
    }

    let occ_no_source = Bitboard(state.occupancies[crate::state::BOTH].0 & !(1u64 << mv.source()));
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

fn see(state: &GameState, mv: Move) -> i32 {
    let mut gain = [0; 32];
    let mut d = 0;

    let from = mv.source();
    let to = mv.target();
    let mut piece = get_piece_type_safe(state, from);
    let victim = get_piece_type_safe(state, to);

    let piece_vals = [
        100, 320, 330, 500, 900, 20000, 100, 320, 330, 500, 900, 20000, 0,
    ];
    gain[d] = piece_vals[victim];
    d += 1;

    let mut occ = state.occupancies[crate::state::BOTH];
    if mv.is_capture() && victim == 12 {
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
    if !mv.is_capture() {
        return;
    }
    let attacker = get_piece_type_safe(state, mv.source());
    let victim = get_victim_type(state, mv.target());

    if victim < 12 {
        let entry = &mut info.data.capture_history[attacker][mv.target() as usize][victim % 6];
        *entry += bonus - (*entry * bonus.abs()) / 1024;
    }
}

#[inline(always)]
fn get_piece_type_safe(state: &GameState, square: u8) -> usize {
    state.board[square as usize] as usize
}

#[inline(always)]
fn get_moved_piece(state: &GameState, mv: Move) -> usize {
    let piece = get_piece_type_safe(state, mv.target());
    if piece != 12 {
        piece
    } else {
        if state.side_to_move == BLACK {
            K
        } else {
            k
        }
    }
}

#[inline(always)]
fn get_victim_type(state: &GameState, square: u8) -> usize {
    state.board[square as usize] as usize
}

fn get_pv_line(state: &GameState, tt: &TranspositionTable, depth: u8, thread_id: Option<usize>) -> (String, Option<Move>) {
    let mut pv_str = String::new();
    let mut curr_state = *state;
    let mut seen_hashes = Vec::new();
    let mut ponder_move = None;
    let mut first = true;
    for _ in 0..depth {
        if let Some(mv) = tt.get_move(curr_state.hash, &curr_state, thread_id) {
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
    state: &mut GameState,
    stack: &mut [StackEntry],
    mut alpha: i32,
    beta: i32,
    info: &mut SearchInfo,
    ply: usize,
) -> i32 {
    if ply > info.seldepth as usize {
        info.seldepth = ply as u8;
    }

    if ply >= MAX_PLY {
        return eval::evaluate_light(state);
    }
    info.nodes += 1;
    if info.nodes % 1024 == 0 {
        info.check_time();
    }
    if info.stopped {
        return 0;
    }

    // Safety Verification
    #[cfg(debug_assertions)]
    if info.nodes % 4096 == 0 {
         crate::nnue::verify_accumulator(state, &info.data.accumulators[0], 0);
         crate::nnue::verify_accumulator(state, &info.data.accumulators[1], 1);
    }

    // TT Probe in QSearch
    if let Some((score, _d, flag, _m)) = info.tt.probe_data(state.hash, state, info.thread_id) {
        if flag == FLAG_EXACT {
            return score;
        }
        if flag == FLAG_ALPHA && score <= alpha {
            return score;
        }
        if flag == FLAG_BETA && score >= beta {
            return score;
        }
    }

    let in_check = is_in_check(state);

    if !in_check {
        let stand_pat = eval::evaluate_light(state);
        if stand_pat >= beta {
            return beta;
        }

        use crate::state::{q, Q};
        let us = state.side_to_move;
        let enemy_queens = state.bitboards[if us == WHITE { q } else { Q }];

        let delta = if info.tuning.qsearch_delta_dynamic {
            if enemy_queens.0 != 0 { 975 } else { 600 }
        } else {
            975
        };

        let is_endgame = (state.bitboards[Q].0 | state.bitboards[q].0) == 0;

        if !is_endgame {
            if stand_pat + delta < alpha {
                return alpha;
            }
        }

        if stand_pat > alpha {
            alpha = stand_pat;
        }
    }

    // Use default values for MovePicker in QSearch
    let mut picker = MovePicker::new(info.data, info.tt, state, stack, ply, None, true, info.thread_id, Some(info.params), false);

    let mut legal_moves_found = 0;

    while let Some(mv) = picker.next_move(state, info.data, stack) {
        if !in_check {
            let is_quiet = !mv.is_capture() && mv.promotion().is_none();
            if is_quiet {
                if !(info.tuning.qsearch_include_checks && gives_check_fast(state, mv)) {
                    continue;
                }
            }
        }

        // SEE Gating for QSearch
        if !in_check {
            if ENABLE_SEE_GATE_QS && mv.is_capture() && mv.promotion().is_none() {
                 if !gives_check_fast(state, mv) {
                     if !see_ge(state, mv, 0) {
                         continue;
                     }
                 }
            }
        }

        let unmake_info = state.make_move_inplace(mv, &mut Some(&mut info.data.accumulators));

        let our_side = state.side_to_move;
        let mover = 1 - our_side;
        let mover_king = if mover == WHITE { K } else { k };
        let king_sq = state.bitboards[mover_king].get_lsb_index() as u8;

        if movegen::is_square_attacked(state, king_sq, our_side) {
            state.unmake_move(mv, unmake_info, &mut Some(&mut info.data.accumulators));
            continue;
        }

        legal_moves_found += 1;

        let score = -quiescence(state, stack, -beta, -alpha, info, ply + 1);

        state.unmake_move(mv, unmake_info, &mut Some(&mut info.data.accumulators));

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

#[derive(Clone, Debug)]
pub struct RootMove {
    pub score: i32,
    pub prev_score: i32,
    pub mv: Move,
    pub pv: Vec<Move>,
}

impl RootMove {
    pub fn new(mv: Move) -> Self {
        Self {
            score: -INFINITY,
            prev_score: -INFINITY,
            mv,
            pv: Vec::new(),
        }
    }
}

pub fn search(
    state: &GameState,
    limits: Limits,
    tt: &TranspositionTable,
    stop_signal: Arc<AtomicBool>,
    main_thread: bool,
    history: &[u64],
    search_data: &mut SearchData,
    stack: &mut [StackEntry], // PASSED STACK
    params: &SearchParameters,
    mode: SearchMode,
    global_nodes: Option<&AtomicU64>,
    thread_id: Option<usize>,
) -> (i32, Option<Move>) {
    let mut best_move: Option<Move> = None;
    let mut ponder_move = None;

    let max_depth = match limits {
        Limits::FixedDepth(d) => d,
        Limits::Smart { .. } => MAX_PLY as u8,
        _ => MAX_PLY as u8,
    };

    state.refresh_accumulator(&mut search_data.accumulators);

    if main_thread {
        if let Some((tb_move, tb_score)) = syzygy::probe_root(state) {
            println!("info string Syzygy Found: Score {} Move {:?}", tb_score, tb_move);
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

    let start_time = std::time::Instant::now();

    let mut info = Box::new(SearchInfo::new(
        search_data,
        limits,
        stop_signal,
        tt,
        main_thread,
        params,
        global_nodes,
        thread_id,
        mode,
    ));

    // Copy history to stack-based SearchPath
    info.path.load_from(history);

    // Initialize Stack Root
    stack[0] = StackEntry::default();
    stack[0].current_move = Move::default(); // Ensure root move is default (none)
    stack[0].in_check = is_in_check(state);

    // 1. Generate Root Moves
    let mut root_moves = Vec::new();
    let mut generator = MoveGenerator::new();
    generator.generate_moves(state);

    // Check legality and create RootMoves
    for i in 0..generator.list.count {
        let mv = generator.list.moves[i];
        let mut next_state = *state;
        next_state.make_move_inplace(mv, &mut None);
        let our_king = if state.side_to_move == WHITE { K } else { k };
        let king_sq = next_state.bitboards[our_king].get_lsb_index() as u8;

        if !movegen::is_square_attacked(&next_state, king_sq, next_state.side_to_move) {
            root_moves.push(RootMove::new(mv));
        }
    }

    if root_moves.is_empty() {
        if stack[0].in_check {
            if main_thread { println!("bestmove (none)"); }
            return (-MATE_VALUE, None);
        } else {
            if main_thread { println!("bestmove (none)"); }
            return (0, None);
        }
    }

    let mut last_score = 0;
    let mut depth = 1;
    let mut best_move_changes = 0;
    let mut prev_best_move: Option<Move> = None;

    // --- ITERATIVE DEEPENING LOOP ---
    while depth <= max_depth {
        info.root_depth = depth;
        info.seldepth = 0;

        // Sort Root Moves
        // If depth > 1, sort by score (which is prev_score from depth-1)
        if depth > 1 {
            root_moves.sort_by(|r1, r2| r2.score.cmp(&r1.score));
        }

        let mut alpha = -INFINITY;
        let mut beta = INFINITY;

        if ENABLE_ASPIRATION && depth >= 5 && main_thread {
            alpha = last_score - ASP_WINDOW_CP;
            beta = last_score + ASP_WINDOW_CP;
        }

        let mut delta = ASP_WIDEN_1;
        let mut best_score = -INFINITY;
        let mut best_move_this_iteration = None;

        // Aspiration Window Loop
        loop {
            if alpha < -3000 { alpha = -INFINITY; }
            if beta > 3000 { beta = INFINITY; }

            best_score = -INFINITY;

            // Iterate Root Moves
            for (i, rm) in root_moves.iter_mut().enumerate() {
                let mut root_state = *state;
                let mv = rm.mv;

                // Populate Stack[1]
                let next_ply = 1;
                stack[next_ply].current_move = mv;
                stack[next_ply].to_sq = mv.target() as usize;
                stack[next_ply].moved_piece = get_piece_type_safe(&root_state, mv.source());
                stack[next_ply].is_capture = mv.is_capture();
                stack[next_ply].in_check = stack[0].in_check; // CMH bucket key: in-check at this node (before mv)
                let unmake_info = root_state.make_move_inplace(mv, &mut Some(&mut info.data.accumulators));
                info.path.push(root_state.hash);

                let mut score;

                // PVS Logic at Root
                if i == 0 {
                    score = -negamax(&mut root_state, stack, depth - 1, -beta, -alpha, &mut info, next_ply, true, false, false);
                } else {
                    // Start with Null Window
                    score = -negamax(&mut root_state, stack, depth - 1, -alpha - 1, -alpha, &mut info, next_ply, false, false, false);
                    if score > alpha && score < beta {
                        // Research with full window
                        score = -negamax(&mut root_state, stack, depth - 1, -beta, -alpha, &mut info, next_ply, true, false, false);
                    }
                }

                info.path.pop();
                root_state.unmake_move(mv, unmake_info, &mut Some(&mut info.data.accumulators));

                if info.stopped {
                    break;
                }

                rm.score = score;

                if score > best_score {
                    best_score = score;
                    best_move_this_iteration = Some(mv);
                }

                if score > alpha {
                    alpha = score;
                }

                if score >= beta {
                    break;
                }
            }

            if info.stopped {
                break;
            }

            // Aspiration Logic
            if best_score <= alpha {
                if let Limits::FixedTime(ref mut tm) = info.limits {
                    tm.report_aspiration_fail(2); // Fail Low
                }
                beta = (alpha + beta) / 2;
                alpha = (-INFINITY).max(alpha - delta);
                if delta == ASP_WIDEN_1 { delta = ASP_WIDEN_2; } else { delta = INFINITY; }
            } else if best_score >= beta {
                beta = (INFINITY).min(beta + delta);
                if delta == ASP_WIDEN_1 { delta = ASP_WIDEN_2; } else { delta = INFINITY; }
            } else {
                break;
            }
        }

        if info.stopped {
            break;
        }

        last_score = best_score;
        best_move = best_move_this_iteration;

        // Store best move in TT (Root)
        if let Some(bm) = best_move {
             info.tt.store(state.hash, last_score, Some(bm), depth, FLAG_EXACT, info.thread_id);
        }

        // Time Management Updates (Root Loop)
        if main_thread {
            let current_best_move = best_move;
            if current_best_move != prev_best_move {
                best_move_changes += 1;
                prev_best_move = current_best_move;
            }

            if let Limits::FixedTime(ref mut tm) = info.limits {
                let instability = (1.0 + 0.15 * best_move_changes as f64).min(1.6);
                let new_soft = (tm.base_soft_limit as f64 * instability) as u128;
                tm.soft_limit = new_soft.min(tm.hard_limit);
            }
        }

        // Forced Move Check
        let mut forced_margin = None;
        if let Limits::FixedTime(ref mut tm) = info.limits {
            if main_thread {
                if let Some(bm) = best_move {
                     tm.report_completed_depth(depth as i32, last_score, bm);
                }
                forced_margin = tm.check_for_forced_move(depth as i32);
                if tm.check_soft_limit() {
                    info.stopped = true;
                    info.stop_signal.store(true, Ordering::Relaxed);
                }
            }
        }

        if let Some(margin) = forced_margin {
             if let Some(bm) = best_move {
                  let is_forced = is_forced_move(&mut (*state).clone(), stack, margin, &mut info, bm, last_score, depth);
                  if is_forced {
                       if let Limits::FixedTime(ref mut tm) = info.limits {
                           tm.report_forced_move(depth as i32);
                       }
                  }
             }
        }

        if main_thread {
            let elapsed = start_time.elapsed().as_secs_f64();
            let total_nodes = if let Some(gn) = global_nodes {
                gn.load(Ordering::Relaxed)
            } else {
                info.nodes
            };

            let nps = if elapsed > 0.0 {
                (total_nodes as f64 / elapsed) as u64
            } else {
                0
            };
            let mut score_str = if last_score > MATE_SCORE {
                format!("mate {}", (MATE_VALUE - last_score + 1) / 2)
            } else if last_score < -MATE_SCORE {
                format!("mate -{}", (MATE_VALUE + last_score + 1) / 2)
            } else {
                format!("cp {}", last_score)
            };

            if crate::uci::UCI_SHOW_WDL.load(Ordering::Relaxed) {
                if last_score.abs() < MATE_SCORE {
                    let (w, d, l) = to_wdl(last_score);
                    score_str.push_str(&format!(" wdl {} {} {}", w, d, l));
                }
            }

            let mut pv_line = String::new();
            if let Some(mv) = best_move {
                 let (line, p_move) = get_pv_line(state, tt, depth, thread_id);
                 pv_line = line;
                 ponder_move = p_move;
            }

            println!(
                "info depth {} seldepth {} score {} nodes {} nps {} hashfull {} time {} pv {}",
                depth,
                info.seldepth,
                score_str,
                total_nodes,
                nps,
                tt.hashfull(),
                start_time.elapsed().as_millis(),
                pv_line
            );
        }

        depth += 1;
    }

    // Fallback if no best move found (e.g. stopped early)
    if best_move.is_none() && !root_moves.is_empty() {
        best_move = Some(root_moves[0].mv);
    }

    if main_thread {
        if let Some(bm) = best_move {
            print!("bestmove {}", format_move_uci(bm, state));
            if let Some(pm) = ponder_move {
                print!(
                    " ponder {}",
                    format_move_uci(pm, state)
                );
            }
            println!();
        } else {
            println!("bestmove (none)");
        }
    }

    (last_score, best_move)
}

fn negamax(
    state: &mut GameState,
    stack: &mut [StackEntry],
    depth: u8,
    mut alpha: i32,
    beta: i32,
    info: &mut SearchInfo,
    ply: usize,
    is_pv: bool,
    was_sacrifice: bool,
    is_null_node: bool,
) -> i32 {
    if state.halfmove_clock >= 100 {
        return 0;
    }

    let alpha0 = alpha;

    // OPTIMIZED REPETITION CHECK using SearchPath
    if ply > 0 {
        let path_len = info.path.len;
        let start_idx = path_len.saturating_sub(state.halfmove_clock as usize);
        let end_idx = path_len.saturating_sub(1);

        // Step by 2
        let mut i = end_idx;
        while i >= start_idx {
             if info.path.keys[i] == state.hash {
                 return 0;
             }
             if i < 2 { break; }
             i -= 2;
        }
    }

    let mate_value = MATE_VALUE - (ply as i32);
    if alpha < -mate_value {
        alpha = -mate_value;
    }
    if alpha >= beta {
        return alpha;
    }

    if ply >= MAX_PLY {
        return eval::evaluate(state, Some(&mut info.data.accumulators), Some(&mut info.data.nnue_scratch), alpha, beta);
    }

    if ply > 0 && state.occupancies[BOTH].count_bits() <= 6 && state.castling_rights == 0 {
        if let Some(wdl_score) = syzygy::probe_wdl(state) {
            if wdl_score >= beta { return wdl_score; }
            if wdl_score <= alpha { return wdl_score; }
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

    let in_check = stack[ply].in_check; // Use stack context

    let mut new_depth = depth;
    if in_check {
        new_depth = new_depth.saturating_add(1);
    }

    if new_depth == 0 {
        return quiescence(state, stack, alpha, beta, info, ply);
    }

    let excluded_move = stack[ply].excluded_move;
    let prev_move = if ply > 0 { Some(stack[ply].current_move) } else { None };

    let mut extensions = 0;
    if let Some(pm) = prev_move {
        let p_piece = get_piece_type_safe(state, pm.target());
        let rank = pm.target() / 8;
        if (p_piece == P && rank == 6) || (p_piece == p && rank == 1) {
            extensions += 1;
        }
    }

    let mut tt_move = None;
    let mut tt_score = -INFINITY;
    let mut tt_depth = 0;
    let mut tt_flag = FLAG_ALPHA;

    let depth_with_ext = new_depth.saturating_add(extensions);

    let tt_hit;
    if let Some((score, d, flag, mv)) = info.tt.probe_data(state.hash, state, info.thread_id) {
        tt_hit = true;
        tt_score = score;
        tt_depth = d;
        tt_flag = flag;
        tt_move = mv;

        if ply > 0 && excluded_move == Move::default() && d >= depth_with_ext {
            if flag == FLAG_EXACT {
                info.tt_hit_avg += (1024 - info.tt_hit_avg) >> 6;
                return score;
            }
            if flag == FLAG_ALPHA && score <= alpha {
                info.tt_hit_avg += (1024 - info.tt_hit_avg) >> 6;
                return alpha;
            }
            if flag == FLAG_BETA && score >= beta {
                info.tt_hit_avg += (1024 - info.tt_hit_avg) >> 6;
                return beta;
            }
        }
    } else {
        tt_hit = false;
    }

    // TT Hit Avg Update
    if tt_hit {
        info.tt_hit_avg += (1024 - info.tt_hit_avg) >> 6;
    } else {
        info.tt_hit_avg -= info.tt_hit_avg >> 6;
    }

    // Internal Iterative Reduction (IIR)
    if ENABLE_IIR
        && ply > 0
        && !is_pv
        && !in_check
        && excluded_move == Move::default()
        && tt_move.is_none()
        && new_depth >= 6
    {
        new_depth -= 1;
    }

    let static_eval = if in_check {
        -INFINITY
    } else {
        let ev = eval::evaluate(state, Some(&mut info.data.accumulators), Some(&mut info.data.nnue_scratch), alpha, beta);
        stack[ply].static_eval = ev;
        ev
    };
    // Correction history: adjust static eval for pruning stability (NMP/RFP/Razoring/Futility)
    let correction = info
        .data
        .correction_history
        .get(state.pawn_key, state.side_to_move);
    let corrected_eval = static_eval + correction;


    let improving = ply >= 2 && !in_check && static_eval >= stack[ply - 2].static_eval;

    if !is_pv && !in_check && excluded_move == Move::default() && new_depth <= 3 {
        let razor_margin = info.params.razoring_base + (new_depth as i32 * info.params.razoring_multiplier);
        if corrected_eval + razor_margin < alpha {
            let v = quiescence(state, stack, alpha, beta, info, ply);
            if v < alpha {
                return v;
            }
        }
    }

    if !is_pv
        && !in_check
        && excluded_move == Move::default()
        && new_depth < 7
        && corrected_eval - (info.params.rfp_margin * new_depth as i32) >= beta
    {
        return corrected_eval;
    }

    if ENABLE_NULL_MOVE
        && info.tuning.allow_nullmove && new_depth >= info.tuning.null_min_depth
        && ply > 0
        && !in_check
        && !is_pv
        && excluded_move == Move::default()
        && corrected_eval >= beta
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
            let reduction_depth = info.params.nmp_base + new_depth as i32 / info.params.nmp_divisor;
            let unmake_info = state.make_null_move_inplace();

            let next_ply = ply + 1;
// Threat context for this node (used to avoid reducing genuinely tactical quiet moves).
// Kept behind depth/in-check guards to avoid adding overhead in shallow nodes.
let threat_info: Option<ThreatInfo> = if ENABLE_LMR && new_depth > 4 && !in_check {
    Some(threat::analyze(state))
} else {
    None
};

            stack[next_ply] = StackEntry::default();
            stack[next_ply].in_check = false;

            let reduced_depth = new_depth.saturating_sub(reduction_depth as u8);
            let score = -negamax(
                state,
                stack,
                reduced_depth,
                -beta,
                -beta + 1,
                info,
                next_ply,
                false,
                false,
            true,
            );

            state.unmake_null_move(unmake_info);

            if info.stopped {
                return 0;
            }
            if score >= beta {
                return beta;
            }
        }
    }

    if !is_pv
        && new_depth >= 5
        && !in_check
        && excluded_move == Move::default()
        && beta.abs() < MATE_SCORE
    {
        let prob_beta = beta + 200;
        let prob_depth = new_depth - 4;

        let prob_score = -negamax(
            state,
            stack,
            prob_depth,
            -prob_beta,
            -prob_beta + 1,
            info,
            ply + 1,
            false,
            was_sacrifice,
        false,
        );

        if prob_score >= prob_beta {
            return prob_score;
        }
    }

    if is_pv && tt_move.is_none() && new_depth > 4 {
        let iid_depth = new_depth - 2;
        negamax(
            state,
            stack,
            iid_depth,
            alpha,
            beta,
            info,
            ply,
            is_pv,
            was_sacrifice,
        false,
        );
        if let Some(mv) = info.tt.get_move(state.hash, state, info.thread_id) {
            tt_move = Some(mv);
        }
    }

    let mut extension = 0;
    if ply > 0
        && new_depth >= 8
        && tt_move.is_some()
        && excluded_move == Move::default()
        && tt_depth >= new_depth.saturating_sub(3)
        && tt_flag == FLAG_EXACT
        && tt_score.abs() < MATE_SCORE
    {
        let margin = 2 * new_depth as i32;
        let singular_beta = tt_score.saturating_sub(margin);
        let reduced_depth = new_depth.saturating_sub(3);

        stack[ply].excluded_move = tt_move.unwrap();

        let score = negamax(
            state,
            stack,
            reduced_depth,
            singular_beta - 1,
            singular_beta,
            info,
            ply,
            false,
            was_sacrifice,
        false,
        );

        stack[ply].excluded_move = Move::default();

        if score < singular_beta {
            extension = 1;
            if !is_pv && score < singular_beta - (new_depth as i32 * 2) {
                extension = 2;
            }
        } else if singular_beta >= beta {
            return singular_beta;
        }
    }

    let mut picker = MovePicker::new(info.data, info.tt, state, stack, ply, tt_move, false, info.thread_id, Some(info.params), is_pv);

    let mut max_score = -INFINITY;
    let mut best_move = None;
    let mut moves_searched = 0;
    let mut quiets_checked = 0;
    let cut_node = beta - alpha == 1;

    info.path.push(state.hash);

    let pinned = if in_check {
        Bitboard(0)
    } else {
        movegen::get_pinned_mask(state, state.side_to_move)
    };

    let next_ply = ply + 1;

    while let Some(mv) = picker.next_move(state, info.data, stack) {
        if Some(mv) == excluded_move.into() {
            continue;
        }

        let is_capture_move = mv.is_capture();
        let is_quiet = !is_capture_move && mv.promotion().is_none();
// If this quiet move creates a meaningful tactical shift, don't reduce it later with LMR.
let is_tactical_for_lmr = if is_quiet {
    threat_info
        .as_ref()
        .map(|ti| threat::analyze_move_threat_impact(state, mv, ti).is_tactical)
        .unwrap_or(false)
} else {
    false
};


        // Futility Pruning
        if info.tuning.enable_futility && new_depth < 5 && !in_check && !is_pv && is_quiet {
            let futility_margin = FUTILITY_MARGIN_PER_PLY * new_depth as i32;
            if corrected_eval + futility_margin <= alpha {
                quiets_checked += 1;
                continue;
            }
        }

        // History Pruning
        if info.tuning.enable_history_pruning && new_depth < 8 && !in_check && !is_pv && is_quiet {
            let history = info.data.history[mv.source() as usize][mv.target() as usize];
            if history < -8192 {
                quiets_checked += 1;
                continue;
            }
        }

        // Late Move Pruning
        if ENABLE_LMP && info.tuning.enable_lmp && !is_pv && !in_check && new_depth <= LMP_DEPTH_MAX && is_quiet {
            if quiets_checked >= info.params.lmp_table[new_depth as usize] {
                continue;
            }
        }

        // SEE Gating for Captures
        if ENABLE_SEE_GATE_MAIN && is_capture_move && moves_searched >= 6 && new_depth <= 8 && !in_check {
             if !gives_check_fast(state, mv) && !see_ge(state, mv, 0) {
                 continue;
             }
        }

        // SEE Pruning for Quiet Moves
        if !is_pv
           && !in_check
           && is_quiet
           && new_depth < 8
           && !see_ge(state, mv, -50 * (new_depth as i32))
        {
             continue;
        }

        // 1. POPULATE STACK (Pre-recursion)
        stack[next_ply].current_move = mv;
        stack[next_ply].to_sq = mv.target() as usize;
        stack[next_ply].moved_piece = get_piece_type_safe(state, mv.source());
        stack[next_ply].is_capture = is_capture_move;
        // Check propagation: Does this move give check?
        // We need to calculate if the opponent will be in check in the NEXT state.
        stack[next_ply].in_check = in_check; // CMH bucket key: in-check at this node (before mv)
        let unmake_info = state.make_move_inplace(mv, &mut Some(&mut info.data.accumulators));

        let our_side = state.side_to_move;
        let mover = 1 - our_side;
        let mover_king_type = if mover == WHITE { K } else { k };
        let king_sq = state.bitboards[mover_king_type].get_lsb_index() as u8;

        let from = mv.source();
        let piece = state.board[mv.target() as usize] as usize;

        let is_king = piece == K || piece == k;
        let was_ep = (piece == P || piece == p) && mv.target() == unmake_info.en_passant;

        let needs_check = in_check || is_king || was_ep || pinned.get_bit(from);

        if needs_check {
            if movegen::is_square_attacked(state, king_sq, our_side) {
                state.unmake_move(mv, unmake_info, &mut Some(&mut info.data.accumulators));
                continue;
            }
        }

        info.tt.prefetch(state.hash, info.thread_id);
        moves_searched += 1;
        if is_quiet {
            quiets_checked += 1;
        }

        let move_extension = 0;
        let mut score;
        if moves_searched == 1 {
            let total_extension = (extensions + move_extension + extension).min(1);
            let extended_depth = new_depth.saturating_add(total_extension);
            score = -negamax(
                state,
                stack,
                extended_depth - 1,
                -beta,
                -alpha,
                info,
                next_ply,
                true,
                false,
            false,
            );
        } else {
            let gives_check = is_in_check(state);
            let mut reduction = 0;

            if ENABLE_LMR
                && new_depth >= info.tuning.lmr_min_depth
                && moves_searched >= info.tuning.lmr_min_move_index
                && is_quiet
                && !is_tactical_for_lmr
                && !gives_check
                && !in_check
                && Some(mv) != tt_move
                && (ply >= MAX_PLY
                    || (stack[ply].killers[0] != mv
                        && stack[ply].killers[1] != mv))
            {
                let is_counter = if ply > 0 {
                    let pm = stack[ply-1].current_move;
                    if pm != Move::default() {
                        let p_piece = stack[ply-1].moved_piece;
                        let p_to = pm.target() as usize;
                        info.data.counter_moves[p_piece][p_to] == Some(mv)
                    } else { false }
                } else { false };

                let src_sq = mv.source();
                let piece_type = state.board[src_sq as usize] as usize % 6;

                let is_advanced_pawn = piece_type == 0 && (
                    (state.side_to_move == WHITE && mv.target() >= 48) ||
                    (state.side_to_move == BLACK && mv.target() <= 15)
                );

                if !is_counter && !is_advanced_pawn {
                    let d_idx = new_depth.min(63) as usize;
                    let m_idx = moves_searched.min(63) as usize;
                    let mut lmr_r = info.params.lmr_table[d_idx][m_idx] as i32;

                    let history = info.data.history[mv.source() as usize][mv.target() as usize];
                    lmr_r -= history / 8192;
                    if !improving {
                        lmr_r += 1;
                    }
                    if info.tt_hit_avg > 384 {
                        lmr_r -= 1;
                    }
                    if cut_node {
                        lmr_r += 1;
                    }

                    reduction = lmr_r.max(0) as u8;
                }
            }

            let d = new_depth.saturating_sub(1 + reduction);
            score = -negamax(
                state,
                stack,
                d,
                -alpha - 1,
                -alpha,
                info,
                next_ply,
                false,
                false,
            false,
            false,
            );

            if score >= beta && reduction > 0 {
                score = -negamax(
                    state,
                    stack,
                    new_depth - 1,
                    -alpha - 1,
                    -alpha,
                    info,
                    next_ply,
                    false,
                    false,
                false,
                );
            }
            if score > alpha && score < beta {
                let total_extension = (extensions + move_extension + extension).min(1);
                let extended_depth = new_depth.saturating_add(total_extension);

                score = -negamax(
                    state,
                    stack,
                    extended_depth - 1,
                    -beta,
                    -alpha,
                    info,
                    next_ply,
                    true,
                    false,
                );
            }
        }

        state.unmake_move(mv, unmake_info, &mut Some(&mut info.data.accumulators));

        #[cfg(debug_assertions)]
        if (info.nodes & 0xFFFF) == 0 {
            if let Err(e) = state.validate_consistency() {
                 eprintln!("CRITICAL: Consistency failure after unmake move {:?}", mv);
                 eprintln!("Error: {}", e);
                 state.dump_diagnostics(mv, "Unmake Failure");
                 panic!("Consistency Check Failed");
            }
        }

        if info.stopped {
            info.path.pop();
            return 0;
        }

        if score > max_score {
            max_score = score;
            best_move = Some(mv);
            if score > alpha {
                alpha = score;
            }
        }

        if score >= beta {
            let curr_entry = stack[next_ply];
            let raw_bonus = (new_depth as i32 * new_depth as i32).min(400);
            let bonus = if curr_entry.is_capture { raw_bonus / 2 } else { raw_bonus };

            if is_quiet {
                 let from = mv.source() as usize;
                 let to = mv.target() as usize;
                 update_history(&mut info.data.history[from][to], bonus);

                 if ply < MAX_PLY {
                    stack[ply].killers[1] = stack[ply].killers[0];
                    stack[ply].killers[0] = mv;
                 }
            } else {
                 update_capture_history(info, mv, state, bonus);
            }

            for &back in &[1, 2, 4, 6] {
                if next_ply >= back {
                    let prev_idx = next_ply - back;
                    let prev_entry = stack[prev_idx];

                    if prev_entry.current_move != Move::default() {
                        info.data.cont_history.update(
                            curr_entry.in_check,
                            curr_entry.is_capture,
                            prev_entry.current_move.source() as usize, prev_entry.current_move.target() as usize,
                            curr_entry.current_move.source() as usize, curr_entry.current_move.target() as usize,
                            bonus
                        );
                    }
                }
            }

            break; // Beta Cutoff
        } else {
             if is_quiet {
                 let from = mv.source() as usize;
                 let to = mv.target() as usize;
                 let penalty_bonus = -((new_depth as i32 * new_depth as i32).min(400));
                 update_history(&mut info.data.history[from][to], penalty_bonus);

                for &back in &[1, 2, 4, 6] {
                    if next_ply >= back {
                        let prev_idx = next_ply - back;
                        let prev_entry = stack[prev_idx];
                         let curr_entry = stack[next_ply];

                        if prev_entry.current_move != Move::default() {
                            info.data.cont_history.update(
                                curr_entry.in_check,
                                curr_entry.is_capture,
                                prev_entry.current_move.source() as usize, prev_entry.current_move.target() as usize,
                            curr_entry.current_move.source() as usize, curr_entry.current_move.target() as usize,
                                penalty_bonus
                            );
                        }
                    }
                }
             }
        }
    }

    info.path.pop();

    if moves_searched == 0 {
        if in_check {
            return -MATE_VALUE + (ply as i32);
        } else {
            return 0;
        }
    }

    // Update correction history using the final searched score (skip null-move nodes / mates / fail-lows).
    if !is_null_node
        && excluded_move == Move::default()
        && max_score.abs() < MATE_SCORE - 100
    {
        // Only learn from beta cutoffs or exact nodes (fail-lows are not reliable training signals).
        if max_score > alpha0 {
            info.data.correction_history.update(
                state.pawn_key,
                state.side_to_move,
                static_eval,
                max_score,
                new_depth,
            );
        }
    }

    let flag = if max_score <= alpha0 {
        FLAG_ALPHA
    } else if max_score >= beta {
        FLAG_BETA
    } else {
        FLAG_EXACT
    };
    if excluded_move == Move::default() {
        info.tt
            .store(state.hash, max_score, best_move, new_depth, flag, info.thread_id);
    }
    max_score
}

fn is_forced_move(
    state: &mut GameState,
    stack: &mut [StackEntry],
    margin: i32,
    info: &mut SearchInfo,
    best_move: Move,
    value: i32,
    depth: u8
) -> bool {
    let r_beta = (value - margin).max(-MATE_SCORE + 1);
    let r_depth = (depth.saturating_sub(1)) / 2;

    stack[0].excluded_move = best_move;

    let score = negamax(
        state,
        stack,
        r_depth,
        r_beta - 1,
        r_beta,
        info,
        0,
        false,
        false,
    false,
    );

    stack[0].excluded_move = Move::default();

    score < r_beta
}

pub(crate) fn see_ge(state: &GameState, mv: Move, threshold: i32) -> bool {
    see(state, mv) >= threshold
}

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

pub fn to_wdl(cp: i32) -> (i32, i32, i32) {
    let score = cp as f64;
    let win_p = 1.0 / (1.0 + (-score / 400.0).exp());
    let draw_mass = 250.0 * (- (score.abs() / 200.0).powi(2)).exp(); // Bell curve peak at 0
    let d = draw_mass as i32;
    let w = ((1000.0 - draw_mass) * win_p) as i32;
    let l = 1000 - w - d;
    (w, d, l)
}

pub fn format_move_uci(mv: Move, state: &GameState) -> String {
    let chess960 = crate::uci::UCI_CHESS960.load(Ordering::Relaxed);
    let from = mv.source();
    let mut to = mv.target();

    let piece = get_piece_type_safe(state, from);
    let is_castling = (piece == 5 || piece == 11) &&
                      (get_piece_type_safe(state, to) == if piece == 5 { 3 } else { 9 }) &&
                      (state.bitboards[if piece == 5 { R } else { r }].get_bit(to));

    if !chess960 && is_castling {
        if from == 4 {
            if to > from {
                to = 6;
            }
            else {
                to = 2;
            }
        } else if from == 60 {
            if to > from {
                to = 62;
            }
            else {
                to = 58;
            }
        }
    }

    let mut s = format!("{}{}", square_to_coord(from), square_to_coord(to));

    if let Some(promo) = mv.promotion() {
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