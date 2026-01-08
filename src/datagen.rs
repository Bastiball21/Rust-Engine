// src/datagen.rs
use crate::book::Book;
use crate::bullet_helper::convert_to_bullet;
use crate::search;
use crate::state::{GameState, WHITE};
use crate::tt::{TranspositionTable, FLAG_EXACT};
use crate::parameters::SearchParameters;
use crate::uci::UCI_CHESS960;
use crate::chess960::generate_chess960_position;
use bulletformat::BulletFormat;
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::io::{BufRead, BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use std::fs::File;

// --- Config Constants ---
const MERCY_CP: i32 = 1000;
const MERCY_PLIES: usize = 8;
const WIN_CP: i32 = 700;
const WIN_STABLE_PLIES: usize = 6;
const DRAW_CP: i32 = 50;
const DRAW_PLIES: usize = 20;
const DRAW_START_PLY: usize = 30;
const MAX_PLIES: usize = 200;
const OPENING_SKIP_PLIES: usize = 10;

static STOP_FLAG: AtomicBool = AtomicBool::new(false);

pub struct DatagenConfig {
    pub num_games: usize,
    pub num_threads: usize,
    pub filename: String,
    pub seed: u64,
    pub book_path: Option<String>,
    pub book_ply: usize,
    pub chess960: bool,
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

    // For rolling hash
    fn splitmix(&mut self, v: u64) -> u64 {
        let mut z = v.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
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

    // 1. Determine Global Result First
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

        // Calculate Eval (Score) - White Relative for BulletFormat
        let raw_stm = crate::eval::evaluate(&state, None, None, -32000, 32000);
        let score_white = if state.side_to_move == WHITE { raw_stm } else { -raw_stm };
        let mut clamped_score = score_white.clamp(-32000, 32000) as i16;

        // Mate Score Capping
        const MATE_THRESHOLD: i16 = 20000;
        const MATE_CAP: i16 = 3000;
        if clamped_score.abs() >= MATE_THRESHOLD {
            clamped_score = if clamped_score > 0 { MATE_CAP } else { -MATE_CAP };
        }

        let board_data = convert_to_bullet(&state, clamped_score, global_white_score);
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

    // No pawns logic for datagen
    // If no pawns and only minors, likely drawn or very long
    use crate::state::{P, p};
    let pawns = state.bitboards[P] | state.bitboards[p];
    if pawns.0 == 0 {
         // K vs K, KB vs K, KN vs K, etc already covered below but let's be aggressive
         // If total pieces <= 3 (Kings + 1), it's a draw unless it's a Queen/Rook?
         // User requested: "if no pawns and only minors remain => draw"
         let w_majors = state.bitboards[crate::state::R] | state.bitboards[crate::state::Q];
         let b_majors = state.bitboards[crate::state::r] | state.bitboards[crate::state::q];
         if w_majors.0 == 0 && b_majors.0 == 0 {
             return true;
         }
    }

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

    // --- NEW: Set Global 960 Flag ---
    UCI_CHESS960.store(config.chess960, Ordering::Relaxed);
    // --------------------------------

    println!("Starting Datagen (Aether Zero)");
    println!("  Games:    {}", config.num_games);
    println!("  Threads:  {}", config.num_threads);
    println!("  Output:   {}", config.filename);
    println!("  Seed:     {}", config.seed);
    if config.chess960 {
        println!("  Mode:     Chess960");
    }

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

    println!("  Guards:   Mercy Rule ({}/{}), Stable Win ({}/{}), Max Plies {}, Opening Skip {}",
             MERCY_CP, MERCY_PLIES, WIN_CP, WIN_STABLE_PLIES, MAX_PLIES, OPENING_SKIP_PLIES);

    // Channel now carries (GameHash, Data)
    let (tx, rx) = mpsc::sync_channel::<(u64, Vec<bulletformat::ChessBoard>)>(1000);

    let global_games_written = Arc::new(AtomicUsize::new(0));
    let global_nodes = Arc::new(AtomicU64::new(0));

    // Writer Thread
    let filename = config.filename.clone();
    let total_games = config.num_games;
    let writer_counter = global_games_written.clone();
    let writer_nodes = global_nodes.clone();

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

                let total_nodes = writer_nodes.load(Ordering::Relaxed);
                let nps = if elapsed > 0.0 {
                    total_nodes as f64 / elapsed
                } else {
                    0.0
                };
                let avg_nodes_pos = if positions_written > 0 {
                    total_nodes as f64 / positions_written as f64
                } else {
                    0.0
                };

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
                    "Written {} games ({:.1}%) ({} pos)... {:.1} games/s, {:.1} pos/s, NPS: {:.1}k, Nodes/Pos: {:.0}, Dups: {}, Elapsed: {:.0}s, ETA: {}",
                    games_written,
                    (games_written as f64 / total_games as f64) * 100.0,
                    positions_written,
                    games_per_sec,
                    pos_per_sec,
                    nps / 1000.0,
                    avg_nodes_pos,
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
        let global_nodes_clone = global_nodes.clone();

        let builder = thread::Builder::new()
            .name(format!("datagen_worker_{}", t_id))
            .stack_size(8 * 1024 * 1024);

        let handle = builder
            .spawn(move || {
                let mut tt = TranspositionTable::new(32, 1); // 32MB per thread
                // Seed per thread to ensure unique streams
                let mut rng = Rng::new(
                    config
                        .seed
                        .wrapping_add((t_id as u64).wrapping_mul(0xDEADBEEF)),
                );

                let local_stop = Arc::new(AtomicBool::new(false));

                let mut search_data = search::SearchData::new();

                let mut rep_history: HashMap<u64, u8> = HashMap::with_capacity(300);
                let mut history_vec: Vec<u64> = Vec::with_capacity(300);
                let mut positions: Vec<(GameState, i16)> = Vec::with_capacity(300);

                // Book Shuffle Bag
                let mut book_indices: Vec<usize> = Vec::new();
                let mut book_cursor = 0;

                if let Some(book_ref) = &book_arc {
                    // Shard: Only take indices belonging to this thread
                    book_indices = (0..book_ref.positions.len())
                        .filter(|&i| i % config.num_threads == t_id)
                        .collect();

                    // Shuffle using the custom RNG method
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
                    let mut game_rolling_hash;

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
                        // Init rolling hash from seed ^ start_hash
                        game_rolling_hash = rng.splitmix(config.seed ^ state.hash);

                    } else {
                        if config.chess960 {
                            // Generate random Chess960 position (0..960)
                            let idx = rng.range(0, 960) as u16;
                            state = generate_chess960_position(idx);
                        } else {
                            state = GameState::parse_fen(
                                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                            );
                        }
                        game_rolling_hash = rng.splitmix(config.seed ^ state.hash);

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

                                // Reject losing captures (SEE < 0)
                                if m.is_capture() && !search::see_ge(&state, m, 0) {
                                     // Swap-remove
                                    moves.list.moves[idx] = moves.list.moves[moves.list.count - 1];
                                    moves.list.count -= 1;
                                    continue;
                                }

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

                                // Update rolling hash
                                let mixed_input = game_rolling_hash ^ state.hash.rotate_left(1) ^ (m.0 as u64);
                                game_rolling_hash = rng.splitmix(mixed_input);

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

                    // 3. Search Loop
                    let mut result_val = 0.5;
                    let mut finished = false;
                    let mut mercy_counter = 0;
                    let mut win_stable_counter = 0;
                    let mut draw_counter = 0;

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

                        // Determine Search Limit
                        // FixedNodes(50_000) for consistent quality
                        let limits = search::Limits::FixedNodes(50_000);

                        // Search
                        let mut used_tt_hit = false;
                        let mut search_score = 0;
                        let mut best_move = None;

                        if let Some((tt_score, tt_depth, tt_flag, tt_move)) =
                            tt.probe_data(state.hash, &state, None)
                        {
                            if tt_flag == FLAG_EXACT && tt_depth >= 6 {
                                // SAFETY CHECK: Ensure TT move is valid for current state to prevent collision crashes.
                                // We check both pseudo-legality (geometry/rules) and consistency (capture flag matches board state).
                                if let Some(mv) = tt_move {
                                    // FORCE UPDATE: Explicitly check pseudo legality AND consistency
                                    if tt.is_pseudo_legal(&state, mv) && state.is_move_consistent(mv) {
                                        search_score = tt_score;
                                        best_move = Some(mv);
                                        used_tt_hit = true;
                                    }
                                }
                            }
                        }

                        if !used_tt_hit {
                            // Generate a random ID (ensure non-zero using | 1)
                            let random_id = (rng.next_u64() as usize) | 1;
                            let mut stack = [search::StackEntry::default(); search::STACK_SIZE];

                            let (s, m) = search::search(
                                &state,
                                limits,
                                &tt,
                                local_stop.clone(),
                                false,
                                &history_vec,
                                &mut search_data,
                                &mut stack,
                                &params_clone,
                                search::SearchMode::Datagen,
                                Some(&global_nodes_clone),
                                Some(random_id),
                            );
                            search_score = s;
                            best_move = m;
                        }

                        // Mercy Rule (Extreme stomp)
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

                        // Stable Win Adjudication
                        if search_score.abs() >= WIN_CP {
                            win_stable_counter += 1;
                        } else {
                            win_stable_counter = 0;
                        }
                        if win_stable_counter >= WIN_STABLE_PLIES {
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

                        let mut clamped_score = search_score.clamp(-32000, 32000);

                        // Mate Score Capping
                        const MATE_THRESHOLD: i32 = 20000;
                        const MATE_CAP: i32 = 3000;
                        if clamped_score.abs() >= MATE_THRESHOLD {
                            clamped_score = if clamped_score > 0 { MATE_CAP } else { -MATE_CAP };
                        }

                        // --- Data Sampling Logic ---
                        let mut should_keep = if game_ply < OPENING_SKIP_PLIES {
                            false
                        } else {
                             // "always keep if abs(score_white) <= 200"
                             // "50% keep if 200 < abs <= 600"
                             // "25% keep if abs > 600"
                             // search_score is STM. But absolute value is same as white-relative absolute.
                             let abs = clamped_score.abs();
                             if abs <= 200 {
                                 true
                             } else if abs <= 600 {
                                 rng.range(0, 100) < 50
                             } else {
                                 rng.range(0, 100) < 25
                             }
                        };

                        // Skip one legal move (unless check)
                        if moves.list.count == 1 && !crate::search::is_in_check(&state) {
                            should_keep = false;
                        }

                        if should_keep {
                            // We store the STM-relative score here.
                            // We will convert it to White-Relative before writing.
                            positions.push((state.clone(), clamped_score as i16));
                        }

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

                        // Update Rolling Hash
                        let mixed_input = game_rolling_hash ^ state.hash.rotate_left(1) ^ (final_move.0 as u64);
                        game_rolling_hash = rng.splitmix(mixed_input);

                        // MAX PLIES Hard Stop
                        if history_vec.len() > MAX_PLIES {
                            result_val = 0.5;
                            finished = true;
                            break;
                        }
                    }

                    if abort_game { continue; }

                    if finished {
                        let mut game_data = Vec::with_capacity(positions.len());
                        for (pos_state, score_stm) in positions.drain(..) {
                            // Fix: Convert STM score to White Relative
                            let score_white = if pos_state.side_to_move == WHITE { score_stm } else { -score_stm };

                            let board = convert_to_bullet(&pos_state, score_white, result_val);
                            game_data.push(board);
                        }

                        if !game_data.is_empty() {
                            // Send GameHash (Rolling) + Data
                            if tx.send((game_rolling_hash, game_data)).is_err() {
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
