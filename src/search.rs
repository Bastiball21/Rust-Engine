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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicI16, Ordering};
use std::sync::{Arc, OnceLock};
use smallvec::SmallVec;

// --- FEATURE TOGGLES ---
pub const ENABLE_ASPIRATION: bool = true;
pub const ENABLE_NULL_MOVE: bool = true;
pub const ENABLE_LMR: bool = true;
pub const ENABLE_LMP: bool = true;
pub const ENABLE_SEE_GATE_MAIN: bool = true;
pub const ENABLE_SEE_GATE_QS: bool = true;
pub const ENABLE_IIR: bool = true;

// --- TUNING CONSTANTS ---
pub const ASP_WINDOW_CP: i32 = 50;
pub const ASP_WIDEN_1: i32 = 150;
pub const ASP_WIDEN_2: i32 = 500;

pub const LMR_MIN_DEPTH: u8 = 3;
pub const LMR_MIN_MOVE_INDEX: usize = 4;

pub const LMP_DEPTH_MAX: u8 = 3;

pub const FUTILITY_MARGIN_PER_PLY: i32 = 100;

pub const NULL_MIN_DEPTH: u8 = 3;

const MAX_PLY: usize = 128;
// Max game length we support in search path
const MAX_GAME_PLY: usize = 1024;
const INFINITY: i32 = 32000;
const MATE_VALUE: i32 = 31000;
const MATE_SCORE: i32 = 30000;

pub const WINNING_CAPTURE_BONUS: i32 = 10_000_000;
pub const MIN_WINNING_SEE_SCORE: i32 = WINNING_CAPTURE_BONUS - 16384;

static START_PLY: [i16; 20] = [0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7];
static SKIP_SIZE: [i16; 20] = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4];

// Continuation History
type ContHistTable = [[[i32; 64]; 12]; 768];

// NEW: Shared Correction History
pub struct CorrectionTable(pub [[AtomicI16; 64]; 12]);

impl CorrectionTable {
    pub fn new() -> Self {
        // Initialize with 0
        let mut table: [[AtomicI16; 64]; 12] = unsafe { std::mem::zeroed() };
        for i in 0..12 {
            for j in 0..64 {
                table[i][j] = AtomicI16::new(0);
            }
        }
        CorrectionTable(table)
    }
}

pub struct ThreatStats {
    pub precise_calls: AtomicU64,
    pub approx_calls: AtomicU64,
}

impl Default for ThreatStats {
    fn default() -> Self {
        Self {
            precise_calls: AtomicU64::new(0),
            approx_calls: AtomicU64::new(0),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Limits {
    Infinite,
    FixedDepth(u8),
    FixedNodes(u64),
    FixedTime(TimeManager),
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
    pub killers: [[Option<Move>; 2]; MAX_PLY + 1],
    pub history: [[i32; 64]; 64],
    pub capture_history: Box<[[[i32; 6]; 64]; 12]>,

    // [Piece][ToSquare] -> Best Reply Move
    pub counter_moves: [[Option<Move>; 64]; 12],

    pub cont_history: Box<ContHistTable>,

    // Correction History: [Piece][Square] -> Error Adjustment
    // Changed from local array to Arc<CorrectionTable>
    pub correction_history: Arc<CorrectionTable>,

    // NNUE Accumulators (Thread Local)
    pub accumulators: [Accumulator; 2],

    pub threat_stats: Arc<ThreatStats>,
}

impl SearchData {
    pub fn new(correction_history: Arc<CorrectionTable>) -> Self {
        Self {
            killers: [[None; 2]; MAX_PLY + 1],
            history: [[0; 64]; 64],
            capture_history: Box::new([[[0; 6]; 64]; 12]),
            counter_moves: [[None; 64]; 12],
            cont_history: Box::new([[[0; 64]; 12]; 768]),
            correction_history,
            accumulators: [Accumulator::default(); 2],
            threat_stats: Arc::new(ThreatStats::default()),
        }
    }

    pub fn clear(&mut self) {
        self.killers = [[None; 2]; MAX_PLY + 1];
        self.history = [[0; 64]; 64];
        self.capture_history.fill_with(|| [[0; 6]; 64]);
        self.counter_moves = [[None; 64]; 12];
        self.cont_history.fill_with(|| [[0; 64]; 12]);
        // Correction History is shared and persistent across moves in a search,
        // but we might want to clear it between searches?
        // Usually correction history is cleared per search or per game.
        // Since it's in Arc, `SearchData::clear` can't easily clear it without interior mutability on all threads.
        // Typically it is cleared in `uci.rs` when starting a new search if needed, or aged.
        // For now, we leave it as is (persistent during search phase).
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
    killers: [Option<Move>; 2],
    counter_move: Option<Move>,
    move_list: [Move; MAX_MOVES],
    move_scores: [i32; MAX_MOVES],
    move_count: usize,
    move_index: usize,
    tt: &'a TranspositionTable,
    captures_only: bool,
    cont_index: Option<usize>,
    thread_id: Option<usize>,
    params: Option<&'a SearchParameters>,
    is_pv_node: bool,
}

impl<'a> MovePicker<'a> {
    pub fn new(
        data: &SearchData,
        tt: &'a TranspositionTable,
        state: &GameState,
        ply: usize,
        tt_move: Option<Move>,
        prev_move: Option<Move>,
        captures_only: bool,
        thread_id: Option<usize>,
        params: Option<&'a SearchParameters>,
        is_pv_node: bool,
    ) -> Self {
        let mut killers = [None; 2];
        let mut counter_move = None;
        let mut cont_index = None;

        if !captures_only && ply < MAX_PLY {
            killers = data.killers[ply];
            if let Some(pm) = prev_move {
                let p_piece = get_moved_piece(state, pm);
                let p_to = pm.target() as usize;
                counter_move = data.counter_moves[p_piece][p_to];
                cont_index = Some(p_piece * 64 + p_to);
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
            cont_index,
            thread_id,
            params,
            is_pv_node,
        }
    }

    pub fn next_move(&mut self, state: &GameState, data: &SearchData) -> Option<Move> {
        loop {
            match self.stage {
                MovePickerStage::TtMove => {
                    self.stage = MovePickerStage::GenerateCaptures;
                    if let Some(mv) = self.tt_move {
                        // TT Move Verification
                        // 1. Check if source square has a piece of our color
                        let from = mv.source();
                        let piece_on_src = get_piece_type_safe(state, from);
                        if piece_on_src == NO_PIECE {
                            continue;
                        }

                        let side = state.side_to_move;
                        if side == WHITE {
                            if piece_on_src > 5 { continue; } // Not white piece
                        } else {
                            if piece_on_src < 6 { continue; } // Not black piece
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
                                // Explicitly reject quiet moves to occupied squares (double safety)
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

                        // Check if we ran out of "Good" captures (score < MIN_WINNING_SEE_SCORE)
                        // self.move_index has been incremented by pick_best_move,
                        // so the move we just picked is at self.move_index - 1
                        if self.move_scores[self.move_index - 1] < MIN_WINNING_SEE_SCORE {
                             // Put back and transition to next stage
                             self.move_index -= 1;
                             break;
                        }

                        if Some(mv) == self.tt_move {
                             continue;
                        }

                        // Lazy SEE
                        if !see_ge(state, mv, 0) {
                            // Penalize
                            self.move_scores[self.move_index - 1] -= WINNING_CAPTURE_BONUS;
                            self.move_index -= 1; // Put back
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
                    if let Some(k1) = self.killers[0] {
                        if Some(k1) != self.tt_move && self.tt.is_pseudo_legal(state, k1) && !k1.is_capture() {
                             return Some(k1);
                        }
                    }
                    if let Some(k2) = self.killers[1] {
                        if Some(k2) != self.tt_move && Some(k2) != self.killers[0] && self.tt.is_pseudo_legal(state, k2) && !k2.is_capture() {
                             return Some(k2);
                        }
                    }
                    if let Some(cm) = self.counter_move {
                         if Some(cm) != self.tt_move && Some(cm) != self.killers[0] && Some(cm) != self.killers[1] && self.tt.is_pseudo_legal(state, cm) && !cm.is_capture() {
                             return Some(cm);
                         }
                    }
                }
                MovePickerStage::GenerateQuiets => {
                    let start = self.move_count;
                    self.generate_moves_append(state, GenType::Quiets);
                    self.score_quiets(state, data, start);
                    self.stage = MovePickerStage::YieldRemaining;
                }
                MovePickerStage::YieldRemaining => {
                    while self.move_index < self.move_count {
                        let mv = match self.pick_best_move() {
                             Some(m) => m,
                             None => break,
                        };

                        if Some(mv) == self.tt_move
                           || (!self.captures_only && (Some(mv) == self.killers[0] || Some(mv) == self.killers[1] || Some(mv) == self.counter_move))
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

        // Optimization: if we are in YieldRemaining, we might have mixed scores (negatives from bad captures, positives from quiets).
        // Standard sort is fine.

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

            // Base WINNING_CAPTURE_BONUS
            let mut score = WINNING_CAPTURE_BONUS;

            if victim < 12 {
                 score += mvv_lva[victim % 6][attacker % 6];
            } else {
                 score += 100; // En Passant or similar?
            }

            if attacker < 12 && victim < 12 {
                score += data.capture_history[attacker][mv.target() as usize][victim % 6] / 16;
            }
            self.move_scores[i] = score;
        }
    }

    fn score_quiets(&mut self, state: &GameState, data: &SearchData, start_idx: usize) {
        let us = state.side_to_move;
        let enemy_pawns = state.bitboards[if us == WHITE { p } else { P }];
        let enemy_pawn_attacks = bitboard::pawn_attacks(enemy_pawns, 1 - us);

        let jitter_base = if let Some(tid) = self.thread_id {
            if tid > 0 {
                // Pseudo-random seed from hash and thread id
                state.hash.wrapping_add(tid as u64)
            } else { 0 }
        } else { 0 };

        for i in start_idx..self.move_count {
            let mv = self.move_list[i];
            let from = mv.source() as usize;
            let to = mv.target() as usize;

            let mut score = data.history[from][to];
            if let Some(c_idx) = self.cont_index {
                let piece = get_piece_type_safe(state, mv.source());
                if piece < 12 {
                    score += data.cont_history[c_idx][piece][to];
                }
            }

            // Threat Logic
            // Bonus for moving away from pawn attack
            if enemy_pawn_attacks.get_bit(from as u8) {
                score += 500;
            }
            // Penalty for moving into pawn attack
            if enemy_pawn_attacks.get_bit(to as u8) {
                score -= 1000;
            }

            if jitter_base != 0 {
                let noise = (jitter_base.wrapping_add(mv.0 as u64) & 0x3FF) as i32;
                score += noise;
            }

            self.move_scores[i] = score;
        }

        // --- TACTICAL TAGGING AND REFINEMENT ---
        if let Some(params) = self.params {
            // Check if we should use precise tagging
            // We check gating conditions for the node (Root/PV) here.
            // Individual move checks (quiet rank) are done below.
            // We assume is_root/is_pv is passed or derived.
            // For now, we only enable this logic if we have params, implying we care.

            // We want to process top K moves.
            // Since we haven't sorted yet, we need to sort or find top K.
            // Given the list size is small (up to 218), full sort is fast.
            // Or we can just iterate and find the best ones.
            // Let's do a partial sort or full sort of the `start_idx..move_count` range.
            // But `pick_best_move` does selection sort.
            // If we modify scores now, we might affect ordering.
            // Let's sort the quiet moves by their current scores to identify the candidates.

            let mut indices: SmallVec<[usize; 64]> = SmallVec::new();
            for i in start_idx..self.move_count {
                indices.push(i);
            }

            // Sort indices based on scores (descending)
            indices.sort_by(|&idx_a, &idx_b| self.move_scores[idx_b].cmp(&self.move_scores[idx_a]));

            let top_k = params.tactical_topk_quiets;
            let mut precise_count = 0;
            let mut approx_count = 0;

            for (rank, &idx) in indices.iter().enumerate() {
                let mv = self.move_list[idx];

                // Gating Check
                // Note: We don't have `mv_is_check` here easily without running `gives_check`.
                // `gives_check` is moderately expensive.
                // However, `tag_pin` relies on `occ_after`.
                // Let's use the helper `should_use_precise_tagging`.
                // We pass `mv_is_check=false` for now, or check it if we want.
                // Or we can rely on `quiet_rank`.

                if threat::should_use_precise_tagging(
                    self.is_pv_node,
                    self.is_pv_node,
                    false, // We don't know if it gives check yet
                    false, // Quiet move
                    Some(rank),
                    top_k
                ) {
                    precise_count += 1;

                    let us_white = state.side_to_move == WHITE;

                    // Determine the piece type ON THE TARGET SQUARE (handling promotion)
                    let (piece_on_target, raw_type) = if let Some(promo) = mv.promotion() {
                        // Promotion
                        let p_idx = if us_white { promo } else { promo + 6 };
                        (p_idx, promo)
                    } else {
                        // Normal move
                        let src_piece = get_piece_type_safe(state, mv.source());
                        (src_piece, src_piece % 6)
                    };

                    let is_rook = raw_type == 3;
                    let is_bishop = raw_type == 2;
                    let is_queen = raw_type == 4;

                    let occ_after = threat::occ_after_basic(state.occupancies[BOTH].0, mv.source(), mv.target());

                    let board_at = |sq: u8| {
                        // We need the board AFTER the move.
                        // But `state.board` is before.
                        // `occ_after` handles occupancy.
                        // The piece at `sq` is:
                        // - if sq == to: the moving piece (possibly promoted)
                        // - if sq == from: NO_PIECE (already handled by occ)
                        // - else: state.board[sq]

                        if sq == mv.target() {
                            piece_on_target as u8
                        } else {
                            state.board[sq as usize]
                        }
                    };

                    // 1. PIN / SKEWER
                    let tag = threat::tag_pin_or_skewer_precise(
                        board_at,
                        occ_after,
                        us_white,
                        is_rook,
                        is_bishop,
                        is_queen,
                        mv.target()
                    );

                    if tag.contains(MoveTag::PIN) {
                        self.move_scores[idx] += params.bonus_pin;
                    }
                    if tag.contains(MoveTag::SKEWER) {
                        self.move_scores[idx] += params.bonus_skewer;
                    }

                    // 2. DISCOVERED ATTACK
                    // Need enemy king ring.
                    // We can compute it or get it from `ThreatInfo` if we had it.
                    // Since we don't pass `ThreatInfo` to `MovePicker` easily (it's heavy),
                    // we can recompute ring (cheap bitboards) or skip it.
                    // The function `is_discovered_attack_precise` takes `king_ring_mask`.
                    // Let's compute it quickly.
                    let enemy = 1 - state.side_to_move;
                    let k_sq = state.bitboards[if enemy == WHITE { K } else { k }].get_lsb_index() as u8;
                    let king_ring = bitboard::get_king_attacks(k_sq).0;

                    if threat::is_discovered_attack_precise(
                        board_at,
                        occ_after,
                        us_white,
                        mv.source(),
                        king_ring
                    ) {
                        self.move_scores[idx] += params.bonus_discovered;
                    }

                } else {
                    approx_count += 1;
                }
            }

            // Stats Update
            data.threat_stats.precise_calls.fetch_add(precise_count, Ordering::Relaxed);
            data.threat_stats.approx_calls.fetch_add(approx_count, Ordering::Relaxed);
        }
    }
}

pub struct SearchInfo<'a> {
    pub data: &'a mut SearchData,
    pub static_evals: [i32; MAX_PLY + 1],
    pub nodes: u64,
    pub global_nodes: Option<&'a AtomicU64>,
    pub seldepth: u8,
    pub limits: Limits,
    pub stop_signal: Arc<AtomicBool>,
    pub stopped: bool,
    pub tt: &'a TranspositionTable,
    pub main_thread: bool,
    pub params: &'a SearchParameters,
    pub thread_id: Option<usize>,
    pub path: SearchPath, // Optimized
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
    ) -> Self {
        Self {
            data,
            static_evals: [0; MAX_PLY + 1],
            nodes: 0,
            global_nodes,
            seldepth: 0,
            limits,
            stop_signal: stop,
            stopped: false,
            tt,
            main_thread: main,
            params,
            thread_id,
            path: SearchPath::new(),
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

fn update_continuation_history(
    info: &mut SearchInfo,
    mv: Move,
    prev_move: Option<Move>,
    state: &GameState,
    bonus: i32,
) {
    if let Some(pm) = prev_move {
        let p_piece = get_moved_piece(state, pm);
        let p_to = pm.target() as usize;
        let idx = p_piece * 64 + p_to;

        let c_piece = get_piece_type_safe(state, mv.source());
        let c_to = mv.target() as usize;

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
        let piece = get_moved_piece(state, mv);
        let to = mv.target() as usize;
        let entry = &info.data.correction_history.0[piece][to];

        let scaled_diff = diff.clamp(-512, 512);
        let weight = (depth as i32).min(16);

        // Update formula: Move towards diff
        // We need to load and store atomically
        let current_val = entry.load(Ordering::Relaxed) as i32;
        let new_val = current_val + (scaled_diff - current_val) * weight / 64;
        let clamped_val = new_val.clamp(-16000, 16000) as i16;

        entry.store(clamped_val, Ordering::Relaxed);
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

        let delta = 975;
        use crate::state::{q, Q};
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

    let mut picker = MovePicker::new(info.data, info.tt, state, ply, None, None, true, info.thread_id, Some(info.params), false);

    let mut legal_moves_found = 0;

    while let Some(mv) = picker.next_move(state, info.data) {
        if !in_check {
            if !mv.is_capture() && mv.promotion().is_none() {
                continue;
            }
        }

        // SEE Gating for QSearch
        if ENABLE_SEE_GATE_QS && mv.is_capture() && mv.promotion().is_none() && !in_check {
             if !gives_check_fast(state, mv) {
                 if !see_ge(state, mv, 0) {
                     continue;
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

        let score = -quiescence(state, -beta, -alpha, info, ply + 1);

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

pub fn search(
    state: &GameState,
    limits: Limits,
    tt: &TranspositionTable,
    stop_signal: Arc<AtomicBool>,
    main_thread: bool,
    history: &[u64],
    search_data: &mut SearchData,
    params: &SearchParameters,
    global_nodes: Option<&AtomicU64>,
    thread_id: Option<usize>,
) -> (i32, Option<Move>) {
    let mut best_move: Option<Move> = None;
    let mut ponder_move = None;

    let max_depth = match limits {
        Limits::FixedDepth(d) => d,
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
    ));

    // Copy history to stack-based SearchPath
    info.path.load_from(history);

    let mut last_score = 0;

    let _nodes_at_root = 0;

    let mut root_state = *state;

    // --- THREAD DISTRIBUTION LOGIC START ---
    // Option 1: Same depth for all threads + move ordering jitter
    let mut depth = 1;

    while depth <= max_depth {
        info.seldepth = 0;
        let mut alpha = -INFINITY;
        let mut beta = INFINITY;

        if ENABLE_ASPIRATION && depth >= 5 && main_thread {
            alpha = last_score - ASP_WINDOW_CP;
            beta = last_score + ASP_WINDOW_CP;
        }

        let mut score;
        let mut delta = ASP_WIDEN_1;

        loop {
            if alpha < -3000 {
                alpha = -INFINITY;
            }
            if beta > 3000 {
                beta = INFINITY;
            }

            score = negamax(
                &mut root_state, depth, alpha, beta, &mut info, 0, true, None, None, None, false,
            );
            if info.stopped {
                break;
            }

            if score <= alpha {
                if let Limits::FixedTime(ref mut tm) = info.limits {
                    tm.report_aspiration_fail(2); // FLAG_ALPHA (Fail Low)
                }
                beta = (alpha + beta) / 2;
                alpha = (-INFINITY).max(alpha - delta);
                if delta == ASP_WIDEN_1 {
                     delta = ASP_WIDEN_2;
                } else {
                     delta = INFINITY;
                }
            } else if score >= beta {
                beta = (INFINITY).min(beta + delta);
                if delta == ASP_WIDEN_1 {
                     delta = ASP_WIDEN_2;
                } else {
                     delta = INFINITY;
                }
            } else {
                break;
            }
        }

        last_score = score;

        if info.stopped {
            break;
        }

        let mut forced_margin = None;
        if let Limits::FixedTime(ref mut tm) = info.limits {
            if main_thread {
                let best_move_found = tt.get_move(state.hash, state, thread_id).unwrap_or_default();
                tm.report_completed_depth(depth as i32, score, best_move_found);

                forced_margin = tm.check_for_forced_move(depth as i32);

                if tm.check_soft_limit() {
                    info.stopped = true;
                    info.stop_signal.store(true, Ordering::Relaxed);
                }
            }
        }

        if let Some(margin) = forced_margin {
             let best_move_found = tt.get_move(state.hash, state, thread_id).unwrap_or_default();
             let is_forced = is_forced_move(&mut root_state, margin, &mut info, best_move_found, score, depth);
             if is_forced {
                 if let Limits::FixedTime(ref mut tm) = info.limits {
                     tm.report_forced_move(depth as i32);
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
            let mut score_str = if score > MATE_SCORE {
                format!("mate {}", (MATE_VALUE - score + 1) / 2)
            } else if score < -MATE_SCORE {
                format!("mate -{}", (MATE_VALUE + score + 1) / 2)
            } else {
                format!("cp {}", score)
            };

            if crate::uci::UCI_SHOW_WDL.load(Ordering::Relaxed) {
                if score.abs() < MATE_SCORE {
                    let (w, d, l) = to_wdl(score);
                    score_str.push_str(&format!(" wdl {} {} {}", w, d, l));
                }
            }

            let mut pv_line = String::new();
            if let Some(mv) = tt.get_move(state.hash, state, thread_id) {
                if info.tt.is_pseudo_legal(state, mv) {
                    best_move = Some(mv);
                    let (line, p_move) = get_pv_line(state, tt, depth, thread_id);
                    pv_line = line;
                    ponder_move = p_move;
                }
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
    // --- THREAD DISTRIBUTION LOGIC END ---

    let mut final_move = best_move;
    let mut generator = movegen::MoveGenerator::new();
    generator.generate_moves(state);
    let mut legal_moves: SmallVec<[Move; 64]> = SmallVec::new();
    for i in 0..generator.list.count {
        let mv = generator.list.moves[i];
        let mut next_state = *state;
        next_state.make_move_inplace(mv, &mut None);
        let our_king = if state.side_to_move == WHITE { K } else { k };
        let king_sq = next_state.bitboards[our_king].get_lsb_index() as u8;
        if !movegen::is_square_attacked(&next_state, king_sq, next_state.side_to_move) {
            legal_moves.push(mv);
        }
    }

    if let Some(bm) = final_move {
        let mut found = false;
        for &lm in &legal_moves {
            if bm.source() == lm.source() && bm.target() == lm.target() && bm.promotion() == lm.promotion() {
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
                    format_move_uci(pm, state)
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
    state: &mut GameState,
    depth: u8,
    mut alpha: i32,
    beta: i32,
    info: &mut SearchInfo,
    ply: usize,
    is_pv: bool,
    // Removed path arg, uses info.path
    prev_move: Option<Move>,
    prev_prev_move: Option<Move>,
    excluded_move: Option<Move>,
    was_sacrifice: bool,
) -> i32 {
    if state.halfmove_clock >= 100 {
        return 0;
    }

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
        return eval::evaluate(state, &Some(&info.data.accumulators), alpha, beta);
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

    let in_check = is_check(state, state.side_to_move);

    let mut new_depth = depth;
    if in_check {
        new_depth = new_depth.saturating_add(1);
    }

    if new_depth == 0 {
        return quiescence(state, alpha, beta, info, ply);
    }

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

    if let Some((score, d, flag, mv)) = info.tt.probe_data(state.hash, state, info.thread_id) {
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

    // Internal Iterative Reduction (IIR)
    if ENABLE_IIR
        && ply > 0
        && !is_pv
        && !in_check
        && excluded_move.is_none()
        && tt_move.is_none()
        && new_depth >= 6
    {
        new_depth -= 1;
    }

    let mut raw_eval = -INFINITY;
    let static_eval = if in_check {
        -INFINITY
    } else {
        raw_eval = eval::evaluate(state, &Some(&info.data.accumulators), alpha, beta);
        let mut correction = 0;
        if let Some(pm) = prev_move {
            let piece = get_moved_piece(state, pm);
            // Change: Access atomic correction history
            correction = info.data.correction_history.0[piece][pm.target() as usize].load(Ordering::Relaxed) as i32;
        }
        let eval = raw_eval + correction;
        info.static_evals[ply] = eval;
        eval
    };

    let improving = ply >= 2 && !in_check && static_eval >= info.static_evals[ply - 2];

    if !is_pv && !in_check && excluded_move.is_none() && new_depth <= 3 {
        let razor_margin = info.params.razoring_base + (new_depth as i32 * info.params.razoring_multiplier);
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
        && static_eval - (info.params.rfp_margin * new_depth as i32) >= beta
    {
        return static_eval;
    }

    if ENABLE_NULL_MOVE
        && new_depth >= NULL_MIN_DEPTH
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
            let reduction_depth = info.params.nmp_base + new_depth as i32 / info.params.nmp_divisor;
            let unmake_info = state.make_null_move_inplace();

            let reduced_depth = new_depth.saturating_sub(reduction_depth as u8);
            let score = -negamax(
                state,
                reduced_depth,
                -beta,
                -beta + 1,
                info,
                ply + 1,
                false,
                None,
                None,
                None,
                false,
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
        && excluded_move.is_none()
        && beta.abs() < MATE_SCORE
    {
        let prob_beta = beta + 200;
        let prob_depth = new_depth - 4;

        let prob_score = -negamax(
            state,
            prob_depth,
            -prob_beta,
            -prob_beta + 1,
            info,
            ply + 1,
            false,
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
            prev_move,
            prev_prev_move,
            None,
            was_sacrifice,
        );
        if let Some(mv) = info.tt.get_move(state.hash, state, info.thread_id) {
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
        let margin = 2 * new_depth as i32;
        let singular_beta = tt_score.saturating_sub(margin);
        let reduced_depth = (new_depth - 1) / 2;

        let score = negamax(
            state,
            reduced_depth,
            singular_beta - 1,
            singular_beta,
            info,
            ply,
            false,
            prev_move,
            prev_prev_move,
            tt_move,
            was_sacrifice,
        );

        if score < singular_beta {
            extension = 1;
            if !is_pv && score < singular_beta - (new_depth as i32 * 2) {
                extension = 2;
            }
        } else if singular_beta >= beta {
            return singular_beta;
        }
    }

    let mut picker = MovePicker::new(info.data, info.tt, state, ply, tt_move, prev_move, false, info.thread_id, Some(info.params), is_pv);

    let mut max_score = -INFINITY;
    let mut best_move = None;
    let mut moves_searched = 0;
    let mut quiets_checked = 0;

    info.path.push(state.hash);

    let pinned = if in_check {
        Bitboard(0)
    } else {
        movegen::get_pinned_mask(state, state.side_to_move)
    };

    while let Some(mv) = picker.next_move(state, info.data) {
        if Some(mv) == excluded_move {
            continue;
        }

        let is_quiet = !mv.is_capture() && mv.promotion().is_none();

        // Futility Pruning
        if ENABLE_LMP && new_depth < 5 && !in_check && !is_pv && is_quiet {
            let futility_margin = FUTILITY_MARGIN_PER_PLY * new_depth as i32;
            if static_eval + futility_margin <= alpha {
                quiets_checked += 1;
                continue;
            }
        }

        // History Pruning
        if new_depth < 8 && !in_check && !is_pv && is_quiet {
            let history = info.data.history[mv.source() as usize][mv.target() as usize];
            if history < -4000 * (new_depth as i32) {
                quiets_checked += 1;
                continue;
            }
        }

        // Late Move Pruning
        if ENABLE_LMP && !is_pv && !in_check && new_depth <= LMP_DEPTH_MAX && is_quiet {
            if quiets_checked >= info.params.lmp_table[new_depth as usize] {
                continue;
            }
        }

        // SEE Gating for Captures (Main Search)
        if ENABLE_SEE_GATE_MAIN && mv.is_capture() && moves_searched >= 6 && new_depth <= 8 && !in_check {
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

        let unmake_info = state.make_move_inplace(mv, &mut Some(&mut info.data.accumulators));

        let our_side = state.side_to_move; // This is the side that just moved (opponent of current state side)
        // Correct logic: state.side_to_move is flipped in make_move_inplace.
        // So we want to check if the side that *just moved* (1 - state.side_to_move) is in check.
        // Wait, make_move_inplace flips side.
        // So state.side_to_move is now the opponent.
        // We want to check if 'mover' is in check.
        let mover = 1 - our_side;
        let mover_king_type = if mover == WHITE { K } else { k };
        let king_sq = state.bitboards[mover_king_type].get_lsb_index() as u8;

        // Optimization: If not previously in check, and piece was not pinned, and not King move, and not En Passant...
        // We can assume legality.
        // We need 'from' square.
        let from = mv.source();
        let piece = state.board[mv.target() as usize] as usize; // Piece is now at target
        // piece type can be retrieved from target.
        // King type is 5 or 11.

        let is_king = piece == K || piece == k;
        let is_ep = unmake_info.en_passant != 64 && (piece == P || piece == p) && mv.target() == unmake_info.en_passant;
        // Wait, unmake_info.en_passant is the OLD EP square.
        // Logic: if move is EP capture.
        // state.en_passant was cleared/changed.
        // We can check if move is EP by looking at the move struct logic or reconstructed.
        // Simpler: if mv.is_capture() and target is empty? No, target has piece now.
        // In unmake_info, we know if it was EP? No.
        // But we know from movegen logic: if it's EP, it's tricky.
        // Let's rely on `mv.is_capture()` and `piece == P`.
        // Actually, safer to just check `is_ep` properly or assume full check for EP.
        // `is_move_pseudo_legal` logic for EP: target empty + capture flag.
        // Here we already made the move.
        // Let's just check if it was EP.
        // In `make_move_inplace`: `if (piece_type == P || piece_type == p) && target == old_en_passant` -> EP.
        // We have `unmake_info.en_passant` (old).
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
                extended_depth - 1,
                -beta,
                -alpha,
                info,
                ply + 1,
                true,
                Some(mv),
                prev_move,
                None,
                false,
            );
        } else {
            let gives_check = is_in_check(state);
            let mut reduction = 0;

            if ENABLE_LMR
                && new_depth >= LMR_MIN_DEPTH
                && moves_searched >= LMR_MIN_MOVE_INDEX
                && is_quiet
                && !gives_check
                && !in_check
                && (ply >= MAX_PLY
                    || (info.data.killers[ply][0] != Some(mv)
                        && info.data.killers[ply][1] != Some(mv)))
            {
                let d_idx = new_depth.min(63) as usize;
                let m_idx = moves_searched.min(63) as usize;
                let mut lmr_r = info.params.lmr_table[d_idx][m_idx] as i32;

                let history = info.data.history[mv.source() as usize][mv.target() as usize];
                lmr_r -= history / 8192;
                if !improving {
                    lmr_r += 1;
                }

                reduction = lmr_r.max(0) as u8;
            }

            let d = new_depth.saturating_sub(1 + reduction);
            score = -negamax(
                state,
                d,
                -alpha - 1,
                -alpha,
                info,
                ply + 1,
                false,
                Some(mv),
                prev_move,
                None,
                false,
            );

            if score > alpha && reduction > 0 {
                score = -negamax(
                    state,
                    new_depth - 1,
                    -alpha - 1,
                    -alpha,
                    info,
                    ply + 1,
                    false,
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
                    state,
                    extended_depth - 1,
                    -beta,
                    -alpha,
                    info,
                    ply + 1,
                    true,
                    Some(mv),
                    prev_move,
                    None,
                    false,
                );
            }
        }

        state.unmake_move(mv, unmake_info, &mut Some(&mut info.data.accumulators));

        // Validation
        if cfg!(debug_assertions) || (info.nodes & 0xFFFF) == 0 {
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
                if is_quiet {
                    let from = mv.source() as usize;
                    let to = mv.target() as usize;
                    let bonus = (new_depth as i32) * (new_depth as i32);

                    update_history(&mut info.data.history[from][to], bonus);

                    if let Some(pm) = prev_move {
                        let p_piece = get_moved_piece(state, pm);
                        let p_to = pm.target() as usize;
                        let idx = p_piece * 64 + p_to;
                        let c_piece = get_piece_type_safe(state, mv.source());
                        let c_to = mv.target() as usize;
                        update_history(&mut info.data.cont_history[idx][c_piece][c_to], bonus);

                        info.data.counter_moves[p_piece][pm.target() as usize] = Some(mv);
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
            if is_quiet {
                let from = mv.source() as usize;
                let to = mv.target() as usize;
                let bonus = (new_depth as i32) * (new_depth as i32);
                update_history(&mut info.data.history[from][to], -bonus);

                if let Some(pm) = prev_move {
                    let p_piece = get_moved_piece(state, pm);
                    let p_to = pm.target() as usize;
                    let idx = p_piece * 64 + p_to;
                    let c_piece = get_piece_type_safe(state, mv.source());
                    let c_to = mv.target() as usize;
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

    info.path.pop();

    if moves_searched == 0 {
        if in_check {
            return -MATE_VALUE + (ply as i32);
        } else {
            return 0;
        }
    }

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
            .store(state.hash, max_score, best_move, new_depth, flag, info.thread_id);
    }
    max_score
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

fn is_forced_move(
    state: &mut GameState,
    margin: i32,
    info: &mut SearchInfo,
    best_move: Move,
    value: i32,
    depth: u8
) -> bool {
    let r_beta = (value - margin).max(-MATE_SCORE + 1);
    let r_depth = (depth.saturating_sub(1)) / 2;

    // Check if other moves fail low
    let score = negamax(
        state,
        r_depth,
        r_beta - 1,
        r_beta,
        info,
        0,
        false,
        None, None,
        Some(best_move), // Exclude the best move
        false
    );

    score < r_beta
}

fn see_ge(state: &GameState, mv: Move, threshold: i32) -> bool {
    see(state, mv) >= threshold
}
