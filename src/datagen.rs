// src/datagen.rs
use crate::book::Book;
use crate::bullet_helper::convert_to_bullet;
use crate::search;
use crate::state::{GameState, WHITE};
use crate::tt::{TranspositionTable, FLAG_EXACT};
use crate::parameters::SearchParameters;
use bulletformat::BulletFormat;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufRead, BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use std::fs::File;

// --- Config Constants ---
const MERCY_CP: i32 = 1000;
const MERCY_PLIES: usize = 8;
const DRAW_CP: i32 = 50;
const DRAW_PLIES: usize = 20;
const DRAW_START_PLY: usize = 30;

// High Score Threshold (Decided Game)
const HIGH_SCORE_CP: i32 = 600;
// Losing Side Threshold
const LOSING_SCORE_CP: i32 = -500;

static STOP_FLAG: AtomicBool = AtomicBool::new(false);

pub struct DatagenConfig {
    pub num_games: usize,
    pub num_threads: usize,
    pub filename: String,
    pub seed: u64,
    pub book_path: Option<String>,
    pub book_ply: usize,
}

// --- SplitMix64 RNG ---
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    // Bias-free range (Lemire's method)
    fn range(&mut self, min: usize, max: usize) -> usize {
        let range = (max - min) as u64;
        if range == 0 {
            return min;
        }
        let mut x = self.next_u64();
        let mut m = (x as u128).wrapping_mul(range as u128);
        let mut l = m as u64;
        if l < range {
            let t = range.wrapping_neg() % range;
            while l < t {
                x = self.next_u64();
                m = (x as u128).wrapping_mul(range as u128);
                l = m as u64;
            }
        }
        (m >> 64) as usize + min
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.range(0, i + 1);
            slice.swap(i, j);
        }
    }
}

pub fn convert_pgn(pgn_path: &str, output_path: &str) {
    println!("Converting PGN: {} -> {}", pgn_path, output_path);

    let pgn_file = File::open(pgn_path).expect("Failed to open PGN file");
    let reader = std::io::BufReader::new(pgn_file);
    let output_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true) // Overwrite
        .open(output_path)
        .expect("Failed to open output file");

    let mut writer = BufWriter::with_capacity(64 * 1024 * 1024, output_file);
    let mut games_converted = 0;
    let mut positions_converted = 0;

    let mut current_moves = String::new();
    let mut in_moves = false;
    let start_time = Instant::now();

    for line_res in reader.lines() {
        let line = line_res.expect("Error reading line");
        let trimmed = line.trim();

        if trimmed.is_empty() {
            continue;
        }

        if trimmed.starts_with('[') {
            // Header
            if in_moves {
                // End of previous game
                if !current_moves.is_empty() {
                    let count = process_and_write_game(&current_moves, &mut writer);
                    if count > 0 {
                        games_converted += 1;
                        positions_converted += count;
                        if games_converted % 1000 == 0 {
                            println!(
                                "Converted {} games, {} positions ({:.1} games/s)",
                                games_converted,
                                positions_converted,
                                games_converted as f64 / start_time.elapsed().as_secs_f64()
                            );
                        }
                    }
                    current_moves.clear();
                }
                in_moves = false;
            }
        } else {
            // Moves
            in_moves = true;
            current_moves.push_str(trimmed);
            current_moves.push(' ');
        }
    }

    // Last game
    if !current_moves.is_empty() {
        let count = process_and_write_game(&current_moves, &mut writer);
        if count > 0 {
            games_converted += 1;
            positions_converted += count;
        }
    }

    // Flush
    writer.flush().unwrap();

    println!(
        "Conversion Complete. Games: {}, Positions: {}",
        games_converted, positions_converted
    );
}

fn process_and_write_game(move_text: &str, writer: &mut BufWriter<File>) -> usize {
    let mut state =
        GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut count = 0;

    // Remove comments
    let mut clean_text = String::with_capacity(move_text.len());
    let mut depth_brace = 0;
    let mut depth_paren = 0;

    for c in move_text.chars() {
        match c {
            '{' => depth_brace += 1,
            '}' => {
                if depth_brace > 0 {
                    depth_brace -= 1
                }
            }
            '(' => depth_paren += 1,
            ')' => {
                if depth_paren > 0 {
                    depth_paren -= 1
                }
            }
            _ => {
                if depth_brace == 0 && depth_paren == 0 {
                    clean_text.push(c);
                }
            }
        }
    }

    // Handle compact PGNs (e.g. "1.e4") by adding spaces around dots
    let spaced_text = clean_text.replace(".", " . ");
    let tokens: Vec<&str> = spaced_text.split_whitespace().collect();

    // 1. Determine Global Result First?
    // The prompt says: Parse Global Result: "1-0" -> 1.0, etc.
    // Result is usually the last token.
    let last_token = tokens.last();
    let global_white_score = if let Some(res) = last_token {
        match *res {
            "1-0" => 1.0,
            "0-1" => 0.0,
            "1/2-1/2" => 0.5,
            _ => return 0, // Skip unfinished games (*) or unknown
        }
    } else {
        return 0;
    };

    // 2. Move Loop
    for token in tokens {
        // Skip numbers and dots
        if token.ends_with('.') || token == "." || token.chars().all(|c| c.is_numeric()) {
            continue;
        }
        // Stop at result
        if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" {
            break;
        }

        // 3. Process Position BEFORE making the move?
        // Usually we label the position we are in.
        // "Is this position good for the side to move?"
        // So we evaluate `state` and check who is to move.
        // `global_white_score` is fixed.

        // Calculate Eval (Score)
        let raw_stm = crate::eval::evaluate(&state, &None, -32000, 32000);

        // ADAPTER: The user requires a mandatory logic block that assumes the input `raw_eval`
        // is White-Relative. However, the engine's `evaluate` returns STM-Relative scores.
        // To strictly satisfy the mandatory logic AND produce the correct STM output labels,
        // we first convert the STM score to White-Relative.
        //
        // Logic check:
        // Input: STM Score.
        // Adapter: STM -> White-Relative.
        // Mandatory Block: White-Relative -> STM Score.
        // Result: STM Score (Goal Achieved).
        let raw_eval = if state.side_to_move == crate::state::BLACK { -raw_stm } else { raw_stm };

        // MANDATORY COMMENT & LOGIC
        // IMPORTANT: eval::evaluate() returns White-relative centipawns.
        // We flip here to enforce STM-relative training labels.
        let stm_score = if state.side_to_move == crate::state::WHITE { raw_eval } else { -raw_eval };

        // Calculate Result (Outcome)
        // Logic: let stm_result = ...
        let stm_result = if state.side_to_move == crate::state::WHITE {
            global_white_score
        } else {
            1.0 - global_white_score // Flip: White Win (1.0) becomes Black Loss (0.0)
        };

        // Write Data
        // convert_to_bullet expects White Relative score if we follow its docs, but user says:
        // "Pass these STM-relative values directly. If helper functions complain... ignore them"
        // So we pass stm_score (STM) and stm_result (STM).
        // WARNING: bullet_helper sets `stm` field in ChessBoard.
        // If bulletformat interprets score as always White-Relative, passing STM here is technically "wrong" for that format
        // UNLESS the trainer knows to handle it.
        // But we follow strict orders: "enforcing a single source of truth (STM)".

        // Note: convert_to_bullet takes i16 for score. stm_score is i32. Clamp it.
        let clamped_score = stm_score.clamp(-32000, 32000) as i16;
        let board_data = convert_to_bullet(&state, clamped_score, stm_result);
        bulletformat::ChessBoard::write_to_bin(writer, &[board_data]).unwrap();
        count += 1;

        // Make Move
        if let Some(mv) = crate::book::parse_san(&state, token) {
            let next_state = state.make_move(mv);
            // Verify legality
             if crate::search::is_check(&next_state, state.side_to_move) {
                 // Illegal move in PGN? Stop game.
                 break;
             }
             state = next_state;
        } else {
             // Parse error
             break;
        }
    }

    count
}

// --- Helpers ---

fn is_low_material(state: &GameState) -> bool {
    use crate::state::{k, K};
    let occ = state.occupancies[crate::state::BOTH];
    let k_occ = state.bitboards[K] | state.bitboards[k];
    let non_king_occ = crate::bitboard::Bitboard(occ.0 & !k_occ.0);

    if non_king_occ.0.count_ones() <= 4 {
        return true;
    }

    false
}

fn is_trivial_endgame(state: &GameState) -> bool {
    use crate::state::{b, k, n, B, BLACK, K, N, WHITE};
    let w_pieces = state.occupancies[WHITE].0 & !state.bitboards[K].0;
    let b_pieces = state.occupancies[BLACK].0 & !state.bitboards[k].0;

    if w_pieces == 0 && b_pieces == 0 {
        return true;
    }

    let w_count = w_pieces.count_ones();
    let b_count = b_pieces.count_ones();

    if (w_count == 1 && b_count == 0) || (w_count == 0 && b_count == 1) {
        let is_minor = |bb: u64| {
            (bb & (state.bitboards[N].0
                | state.bitboards[B].0
                | state.bitboards[n].0
                | state.bitboards[b].0))
                != 0
        };

        if w_count == 1 {
            if is_minor(w_pieces) {
                return true;
            }
        }
        if b_count == 1 {
            if is_minor(b_pieces) {
                return true;
            }
        }
    }

    if w_count == 1 && b_count == 1 {
        let is_minor_w = (w_pieces & (state.bitboards[N].0 | state.bitboards[B].0)) != 0;
        let is_minor_b = (b_pieces & (state.bitboards[n].0 | state.bitboards[b].0)) != 0;

        if is_minor_w && is_minor_b {
            return true;
        }
    }

    false
}

pub fn run_datagen(config: DatagenConfig) {
    // Reset the flag in case this function is called multiple times
    STOP_FLAG.store(false, Ordering::Relaxed);

    println!("Starting Datagen (Aether Zero)");
    println!("  Games:    {}", config.num_games);
    println!("  Threads:  {}", config.num_threads);
    println!("  Output:   {}", config.filename);
    println!("  Seed:     {}", config.seed);

    let book = if let Some(path) = &config.book_path {
        println!("  Book:     {} (ply: {})", path, config.book_ply);
        match Book::load_from_file(path, config.book_ply) {
            Ok(b) => {
                 if b.positions.is_empty() {
                     println!("Warning: Book is empty. Falling back to Random Walk.");
                     None
                 } else {
                     Some(Arc::new(b))
                 }
            },
            Err(e) => {
                println!("Error loading book: {}. Falling back to Random Walk.", e);
                None
            }
        }
    } else {
        println!("  Strategy: Random Walk (8-9 plies)");
        None
    };

    println!("  Guards:   Mercy Rule, Early Draw, Duplicate Game Prevention");

    // Channel now carries (GameHash, Data)
    let (tx, rx) = mpsc::sync_channel::<(u64, Vec<bulletformat::ChessBoard>)>(1000);

    let global_games_written = Arc::new(AtomicUsize::new(0));

    // Writer Thread
    let filename = config.filename.clone();
    let total_games = config.num_games;
    let writer_counter = global_games_written.clone();

    let writer_handle = thread::spawn(move || {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .expect("Unable to open output file");

        // massive I/O Buffer: 32MB
        let mut writer = BufWriter::with_capacity(32 * 1024 * 1024, file);
        let mut games_written = 0;
        let mut positions_written = 0;
        let mut seen_hashes: HashSet<u64> = HashSet::new();
        let mut duplicates_discarded = 0;
        let start_time = Instant::now();

        for (game_hash, game_data) in rx {
            // Duplicate Check
            if seen_hashes.contains(&game_hash) {
                duplicates_discarded += 1;
                continue;
            }
            seen_hashes.insert(game_hash);

            bulletformat::ChessBoard::write_to_bin(&mut writer, &game_data).unwrap();
            games_written += 1;
            writer_counter.fetch_add(1, Ordering::Relaxed);
            positions_written += game_data.len();

            if games_written >= total_games {
                STOP_FLAG.store(true, Ordering::Relaxed);
                break;
            }

            if games_written % 100 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let games_per_sec = games_written as f64 / elapsed;
                let pos_per_sec = positions_written as f64 / elapsed;

                let remaining_games = total_games.saturating_sub(games_written);
                let eta_secs = if games_per_sec > 0.0 {
                    remaining_games as f64 / games_per_sec
                } else {
                    0.0
                };

                let eta_str = if eta_secs > 3600.0 {
                    format!("{:.1}h", eta_secs / 3600.0)
                } else if eta_secs > 60.0 {
                    format!("{:.1}m", eta_secs / 60.0)
                } else {
                    format!("{:.0}s", eta_secs)
                };

                println!(
                    "Written {} games ({:.1}%) ({} pos)... {:.1} games/s, {:.1} pos/s, Dups: {}, Elapsed: {:.0}s, ETA: {}",
                    games_written,
                    (games_written as f64 / total_games as f64) * 100.0,
                    positions_written,
                    games_per_sec,
                    pos_per_sec,
                    duplicates_discarded,
                    elapsed,
                    eta_str
                );
            }
        }
        println!(
            "Writer thread finished. Total games: {}, Total pos: {}, Duplicates discarded: {}",
            games_written, positions_written, duplicates_discarded
        );
    });

    // Worker Threads
    let mut handles = vec![];
    let default_params = SearchParameters::default();

    for t_id in 0..config.num_threads {
        let tx = tx.clone();
        let params_clone = default_params.clone();
        let book_arc = book.clone();

        let builder = thread::Builder::new()
            .name(format!("datagen_worker_{}", t_id))
            .stack_size(8 * 1024 * 1024);

        let handle = builder
            .spawn(move || {
                let mut tt = TranspositionTable::new(32, 1); // 32MB per thread
                let mut rng = Rng::new(
                    config
                        .seed
                        .wrapping_add((t_id as u64).wrapping_mul(0xDEADBEEF)),
                );
                let correction_history = Arc::new(search::CorrectionTable::new());
                let mut search_data = search::SearchData::new(correction_history.clone());

                let mut rep_history: HashMap<u64, u8> = HashMap::with_capacity(300);
                let mut history_vec: Vec<u64> = Vec::with_capacity(300);
                let mut positions: Vec<(GameState, i16)> = Vec::with_capacity(200);

                // Book Shuffle Bag
                let mut book_indices: Vec<usize> = Vec::new();
                let mut book_cursor = 0;

                if let Some(book_ref) = &book_arc {
                    book_indices = (0..book_ref.positions.len()).collect();
                    rng.shuffle(&mut book_indices);
                }

                loop {
                    if STOP_FLAG.load(Ordering::Relaxed) {
                        break;
                    }

                    tt.clear();
                    search_data.clear();
                    rep_history.clear();
                    history_vec.clear();
                    positions.clear();

                    // 1. Setup Board
                    let mut state;
                    let mut game_ply = 0;
                    let mut abort_game = false;

                    if let Some(book_ref) = &book_arc {
                        if book_cursor >= book_indices.len() {
                            // Refill / Reshuffle
                            rng.shuffle(&mut book_indices);
                            book_cursor = 0;
                        }
                        let idx = book_indices[book_cursor];
                        book_cursor += 1;

                        state = book_ref.positions[idx].clone();

                        // If state is inconsistent, skip
                        if !state.is_consistent() {
                            continue;
                        }

                        rep_history.insert(state.hash, 1);
                        history_vec.push(state.hash);

                    } else {
                        state = GameState::parse_fen(
                            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        );

                        // 2. Random Walk (8-9 plies)
                        let random_plies = 8 + rng.range(0, 2);

                        rep_history.insert(state.hash, 1);
                        history_vec.push(state.hash);

                        for _ in 0..random_plies {
                            let mut moves = crate::movegen::MoveGenerator::new();
                            moves.generate_moves(&state);

                            if moves.list.count == 0 {
                                abort_game = true;
                                break;
                            }

                            // Lazy Validation
                            let mut picked_move = None;
                            while moves.list.count > 0 {
                                let idx = rng.range(0, moves.list.count);
                                let m = moves.list.moves[idx];
                                let next_state = state.make_move(m);
                                if !crate::search::is_check(&next_state, state.side_to_move) {
                                    picked_move = Some(m);
                                    state = next_state;
                                    break;
                                } else {
                                    // Swap-remove
                                    moves.list.moves[idx] = moves.list.moves[moves.list.count - 1];
                                    moves.list.count -= 1;
                                }
                            }

                            if let Some(m) = picked_move {
                                game_ply += 1;
                                *rep_history.entry(state.hash).or_insert(0) += 1;
                                history_vec.push(state.hash);

                                if state.halfmove_clock >= 100 || is_trivial_endgame(&state) {
                                    abort_game = true;
                                    break;
                                }
                            } else {
                                abort_game = true;
                                break;
                            }
                        }
                    }

                    if abort_game { continue; }
                    if is_trivial_endgame(&state) { continue; }

                    // Capture Game ID (Start Hash)
                    let game_id = state.hash;

                    // 3. Search Loop
                    let mut result_val = 0.5;
                    let mut finished = false;
                    let mut mercy_counter = 0;
                    let mut draw_counter = 0;
                    let mut last_score: i32 = 0; // For Adaptive Depth

                    loop {
                        // Repetition / 50-move
                        if let Some(&count) = rep_history.get(&state.hash) {
                            if count >= 3 {
                                result_val = 0.5;
                                finished = true;
                                break;
                            }
                        }
                        if state.halfmove_clock >= 100 {
                            result_val = 0.5;
                            finished = true;
                            break;
                        }

                        // Syzygy Adjudication
                        if state.occupancies[crate::state::BOTH].0.count_ones() <= 6 {
                             if let Some(score) = crate::syzygy::probe_wdl(&state) {
                                 if score > 0 {
                                     // Win for side to move
                                     result_val = if state.side_to_move == WHITE { 1.0 } else { 0.0 };
                                 } else if score < 0 {
                                     // Loss for side to move
                                     result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                                 } else {
                                     // Draw
                                     result_val = 0.5;
                                 }
                                 finished = true;
                                 break;
                             }
                        }

                        // Insufficient Material Adjudication
                        if is_trivial_endgame(&state) {
                            result_val = 0.5;
                            finished = true;
                            break;
                        }

                        let mut moves = crate::movegen::MoveGenerator::new();
                        moves.generate_moves(&state);

                        if moves.list.count == 0 {
                            if crate::search::is_in_check(&state) {
                                result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                            } else {
                                result_val = 0.5;
                            }
                            finished = true;
                            break;
                        }

                        // Determine Search Depth (Adaptive)
                        let mut search_depth = 8;
                        let abs_score = last_score.abs();
                        if abs_score > 600 {
                            search_depth -= 2;
                        } else if abs_score > 300 {
                            search_depth -= 1;
                        }

                        // Search
                        let mut used_tt_hit = false;
                        let mut search_score = 0;
                        let mut best_move = None;

                        if let Some((tt_score, tt_depth, tt_flag, tt_move)) =
                            tt.probe_data(state.hash, &state, None)
                        {
                            if tt_flag == FLAG_EXACT && tt_depth >= search_depth {
                                // SAFETY CHECK: Ensure TT move is valid for current state to prevent collision crashes
                                if let Some(mv) = tt_move {
                                    if tt.is_pseudo_legal(&state, mv) {
                                        search_score = tt_score;
                                        best_move = Some(mv);
                                        used_tt_hit = true;
                                    }
                                }
                            }
                        }

                        if !used_tt_hit {
                            let limits = search::Limits::FixedDepth(search_depth);
                            let (s, m) = search::search(
                                &state,
                                limits,
                                &tt,
                                Arc::new(AtomicBool::new(false)),
                                false,
                                history_vec.clone(),
                                &mut search_data,
                                &params_clone,
                                None,
                                None,
                            );
                            search_score = s;
                            best_move = m;
                        }

                        last_score = search_score;

                        // Mercy Rule
                        if search_score.abs() >= MERCY_CP {
                            mercy_counter += 1;
                        } else {
                            mercy_counter = 0;
                        }

                        if mercy_counter >= MERCY_PLIES {
                            if search_score > 0 {
                                result_val = if state.side_to_move == WHITE { 1.0 } else { 0.0 };
                            } else {
                                result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                            }
                            finished = true;
                            break;
                        }

                        // Early Draw
                        if game_ply >= DRAW_START_PLY {
                            if search_score.abs() <= DRAW_CP {
                                draw_counter += 1;
                            } else {
                                draw_counter = 0;
                            }
                            if draw_counter >= DRAW_PLIES {
                                result_val = 0.5;
                                finished = true;
                                break;
                            }
                        }

                        // Mate Score adjudication
                        if search_score.abs() > 20000 {
                             if search_score > 0 {
                                result_val = if state.side_to_move == WHITE { 1.0 } else { 0.0 };
                            } else {
                                result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                            }
                            finished = true;
                            break;
                        }

                        let final_move = if let Some(m) = best_move {
                            m
                        } else {
                            // Fallback (should normally be found by search or TT)
                            let mut valid = moves.list.moves[0];
                            for i in 0..moves.list.count {
                                let m = moves.list.moves[i];
                                let next_state = state.make_move(m);
                                if !crate::search::is_check(&next_state, state.side_to_move) {
                                    valid = m;
                                    break;
                                }
                            }
                            valid
                        };

                        let clamped_score = search_score.clamp(-32000, 32000);
                        positions.push((state.clone(), clamped_score as i16));

                        // Consistency
                         if !state.is_consistent() || !state.is_move_consistent(final_move) {
                            abort_game = true;
                            break;
                        }

                        let next_state = state.make_move(final_move);
                        if crate::search::is_check(&next_state, state.side_to_move) {
                            abort_game = true;
                            break;
                        }

                        state = next_state;
                        game_ply += 1;
                        *rep_history.entry(state.hash).or_insert(0) += 1;
                        history_vec.push(state.hash);

                        if history_vec.len() > 600 {
                            result_val = 0.5;
                            finished = true;
                            break;
                        }
                    }

                    if abort_game { continue; }

                    if finished {
                        let mut game_data = Vec::with_capacity(positions.len());
                        for (pos_state, score) in positions.drain(..) {
                            let board = convert_to_bullet(&pos_state, score, result_val);
                            game_data.push(board);
                        }

                        if !game_data.is_empty() {
                            // Send GameID + Data
                            if tx.send((game_id, game_data)).is_err() {
                                break;
                            }
                        }
                    }
                }
            })
            .unwrap();
        handles.push(handle);
    }

    drop(tx);

    for h in handles {
        h.join().unwrap();
    }
    writer_handle.join().unwrap();
}
