// src/datagen.rs
use crate::book::Book;
use crate::bullet_helper::convert_to_bullet;
use crate::search;
use crate::state::{b, k, n, p, q, r, GameState, B, BLACK, K, N, P, Q, R, WHITE};
use crate::time::{TimeControl, TimeManager};
use crate::tt::{TranspositionTable, FLAG_EXACT};
use crate::parameters::SearchParameters;
use bulletformat::BulletFormat;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::sync::mpsc;
use std::sync::{atomic::AtomicBool, Arc};
use std::thread;
use std::time::Instant;

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

pub struct DatagenConfig {
    pub num_games: usize,
    pub num_threads: usize,
    pub depth: u8,
    pub filename: String,
    pub book_path: Option<String>,
    pub book_ply: usize,
    pub seed: u64,
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
}

// --- Helpers ---

fn is_low_material(state: &GameState) -> bool {
    // Count all pieces except kings
    // Iterate over piece types 0..4 (P, N, B, R, Q) and 6..10
    // Actually indices are:
    // 0=P, 1=N, 2=B, 3=R, 4=Q, 5=K
    // 6=p, 7=n, 8=b, 9=r, 10=q, 11=k

    // We can just popcnt the occupancies excluding kings.
    // However, we need to know if it's "low material".
    // Let's count non-pawn non-king pieces as major/minor.
    // Pawns also matter.

    // Simple Heuristic: If piece count (excluding Kings) <= 3, it's low material.
    // Or if only Pawns and Kings remain.

    // Let's stick to total piece count (excluding kings).
    let occ = state.occupancies[crate::state::BOTH];
    let k_occ = state.bitboards[K] | state.bitboards[k];
    let non_king_occ = crate::bitboard::Bitboard(occ.0 & !k_occ.0);

    // Using count_ones() which is popcnt
    if non_king_occ.0.count_ones() <= 4 {
        return true;
    }

    false
}

fn is_trivial_endgame(state: &GameState) -> bool {
    let w_pieces = state.occupancies[WHITE].0 & !state.bitboards[K].0;
    let b_pieces = state.occupancies[BLACK].0 & !state.bitboards[k].0;

    // K vs K
    if w_pieces == 0 && b_pieces == 0 {
        return true;
    }

    let w_count = w_pieces.count_ones();
    let b_count = b_pieces.count_ones();

    // K + X vs K (where X is 1 piece)
    if (w_count == 1 && b_count == 0) || (w_count == 0 && b_count == 1) {
        // Check if the piece is a pawn. If pawn, NOT trivial (unless K vs KP is trivial? No, could be win).
        // Check for KN vs K or KB vs K.
        // Piece types: N=1, B=2. n=7, b=8.
        let is_minor = |bb: u64| {
            // Check if the bit matches N or B bitboards
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

    // KB vs KB (Minor vs Minor) - Drawish if bishops same color? Or just low value.
    if w_count == 1 && b_count == 1 {
        // Check if both are minors
        let is_minor_w = (w_pieces & (state.bitboards[N].0 | state.bitboards[B].0)) != 0;
        let is_minor_b = (b_pieces & (state.bitboards[n].0 | state.bitboards[b].0)) != 0;

        if is_minor_w && is_minor_b {
            // KB vs KB, KN vs KN, KB vs KN...
            return true;
        }
    }

    false
}

pub fn run_datagen(config: DatagenConfig) {
    println!("Starting Datagen (Optimized)");
    println!("  Games:    {}", config.num_games);
    println!("  Threads:  {}", config.num_threads);
    println!("  Depth:    {}", config.depth);
    println!("  Output:   {}", config.filename);
    println!("  Seed:     {}", config.seed);

    // Load Book
    let book = if let Some(ref path) = config.book_path {
        println!("  Book:     {} (Ply: {})", path, config.book_ply);
        match Book::load_from_file(path, config.book_ply) {
            Ok(bk) => {
                if bk.positions.is_empty() {
                    println!("WARNING: Book loaded but contains 0 positions. Using startpos.");
                    None
                } else {
                    Some(Arc::new(bk))
                }
            }
            Err(e) => {
                println!("ERROR: Failed to load book: {}", e);
                return;
            }
        }
    } else {
        println!("  Book:     None (Using Random Walk)");
        None
    };

    println!("  Features: Mercy Rule, Early Draw, Dynamic Nodes, Allocation Reuse");

    let (tx, rx) = mpsc::sync_channel::<Vec<bulletformat::ChessBoard>>(1000);

    // Writer Thread
    let filename = config.filename.clone();
    let total_games = config.num_games;
    let writer_handle = thread::spawn(move || {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .expect("Unable to open output file");

        let mut writer = BufWriter::new(file);
        let mut games_written = 0;
        let mut positions_written = 0;
        let start_time = Instant::now();

        for game_data in rx {
            bulletformat::ChessBoard::write_to_bin(&mut writer, &game_data).unwrap();
            games_written += 1;
            positions_written += game_data.len();

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

                // Format ETA
                let eta_str = if eta_secs > 3600.0 {
                    format!("{:.1}h", eta_secs / 3600.0)
                } else if eta_secs > 60.0 {
                    format!("{:.1}m", eta_secs / 60.0)
                } else {
                    format!("{:.0}s", eta_secs)
                };

                println!(
                    "Written {} games ({:.1}%) ({} pos)... {:.1} games/s, {:.1} pos/s, Elapsed: {:.0}s, ETA: {}",
                    games_written,
                    (games_written as f64 / total_games as f64) * 100.0,
                    positions_written,
                    games_per_sec,
                    pos_per_sec,
                    elapsed,
                    eta_str
                );
            }
        }
        println!(
            "Writer thread finished. Total games: {}, Total pos: {}",
            games_written, positions_written
        );
    });

    // Worker Threads
    let mut handles = vec![];
    let games_per_thread = config.num_games / config.num_threads;
    let remainder = config.num_games % config.num_threads;

    let default_params = SearchParameters::default();

    for t_id in 0..config.num_threads {
        let tx = tx.clone();
        let my_games = games_per_thread + if t_id < remainder { 1 } else { 0 };
        let depth_config = config.depth;
        let book_arc = book.clone();
        let params_clone = default_params.clone(); // Pass default params to datagen

        let builder = thread::Builder::new()
            .name(format!("datagen_worker_{}", t_id))
            .stack_size(8 * 1024 * 1024);

        let handle = builder
            .spawn(move || {
                let mut tt = TranspositionTable::new(32); // 32MB per thread
                let mut rng = Rng::new(
                    config
                        .seed
                        .wrapping_add((t_id as u64).wrapping_mul(0xDEADBEEF)),
                );
                let mut search_data = search::SearchData::new();

                // Reusable allocations
                let mut rep_history: HashMap<u64, u8> = HashMap::with_capacity(300);
                let mut history_vec: Vec<u64> = Vec::with_capacity(300);
                let mut positions: Vec<(GameState, i16)> = Vec::with_capacity(200);

                for _ in 0..my_games {
                    // Clear reusable structures
                    tt.clear();
                    search_data.clear();
                    rep_history.clear();
                    history_vec.clear();
                    positions.clear();

                    // Select Start Position
                    let mut state;
                    let mut game_ply = 0;

                    if let Some(ref bk) = book_arc {
                        let idx = rng.range(0, bk.positions.len());
                        state = bk.positions[idx];
                        // Book positions are assumed to be "ply 0" of the generation sequence.
                    } else {
                        state = GameState::parse_fen(
                            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        );
                    }

                    let mut result_val = 0.5;
                    let mut finished = false;

                    // Adjudication Counters
                    let mut mercy_counter = 0;
                    let mut draw_counter = 0;

                    rep_history.insert(state.hash, 1);
                    history_vec.push(state.hash);

                    // --- 1. Random Move Phase ---
                    // Skip if using book (as per plan), otherwise 8-9 plies.
                    let mut abort_game = false;

                    if book_arc.is_none() {
                        let random_plies = 8 + rng.range(0, 2); // 8 or 9

                        for _ in 0..random_plies {
                            let mut moves = crate::movegen::MoveGenerator::new();
                            moves.generate_moves(&state);

                            if moves.list.count == 0 {
                                abort_game = true;
                                break;
                            }

                            // Collect all legal moves first
                            let mut legal_moves = Vec::with_capacity(64);

                            for i in 0..moves.list.count {
                                let m = moves.list.moves[i];
                                let next_state = state.make_move(m);
                                if !crate::search::is_check(&next_state, state.side_to_move) {
                                    legal_moves.push(m);
                                }
                            }

                            if legal_moves.is_empty() {
                                abort_game = true;
                                break;
                            }

                            // Pick one uniformly
                            let idx = rng.range(0, legal_moves.len());
                            let m = legal_moves[idx];

                            state = state.make_move(m);
                            game_ply += 1;

                            *rep_history.entry(state.hash).or_insert(0) += 1;
                            history_vec.push(state.hash);

                            if state.halfmove_clock >= 100 {
                                abort_game = true;
                                break;
                            }
                            if is_trivial_endgame(&state) {
                                abort_game = true;
                                break;
                            }
                        }
                    }

                    if abort_game {
                        continue;
                    }

                    if is_trivial_endgame(&state) {
                        continue;
                    }

                    // --- 2. Real Search Loop ---
                    loop {
                        // Repetition Check
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

                        let mut moves = crate::movegen::MoveGenerator::new();
                        moves.generate_moves(&state);

                        if moves.list.count == 0 {
                            if crate::search::is_in_check(&state) {
                                result_val = if state.side_to_move == WHITE {
                                    0.0
                                } else {
                                    1.0
                                };
                            } else {
                                result_val = 0.5;
                            }
                            finished = true;
                            break;
                        }

                        // Trivial Endgame Check
                        if is_trivial_endgame(&state) {
                            result_val = 0.5;
                            finished = true;
                            break;
                        }

                        // Dynamic Node Limits
                        let mut min_nodes = 15_000;
                        let mut max_nodes = 40_000;

                        let mut is_losing = false;
                        let mut is_winning_or_decided = false;

                        if let Some((score, _, _, _)) = tt.probe_data(state.hash, &state) {
                            if score <= LOSING_SCORE_CP {
                                is_losing = true;
                            }
                            if score.abs() >= HIGH_SCORE_CP {
                                is_winning_or_decided = true;
                            }
                        }

                        if is_losing {
                            min_nodes = 1000;
                            max_nodes = 2000;
                        } else if is_winning_or_decided {
                            min_nodes = 2000;
                            max_nodes = 5000;
                        } else if game_ply < 20 {
                            min_nodes = 2000;
                            max_nodes = 5000;
                        } else if is_low_material(&state) {
                            min_nodes = 2000;
                            max_nodes = 8000;
                        }

                        let node_limit = rng.range(min_nodes, max_nodes) as u64;

                        // TT Probe Logic for fast exit
                        let mut used_tt_hit = false;
                        let mut search_score = 0;
                        let mut best_move = None;

                        if let Some((tt_score, tt_depth, tt_flag, tt_move)) =
                            tt.probe_data(state.hash, &state)
                        {
                            if tt_flag == FLAG_EXACT && tt_depth >= depth_config {
                                search_score = tt_score;
                                best_move = tt_move;
                                used_tt_hit = true;
                            }
                        }

                        if !used_tt_hit {
                            // We use FixedNodes logic from Limits to enforce node limit
                            let limits = search::Limits::FixedNodes(node_limit);
                            let (s, m) = search::search(
                                &state,
                                limits,
                                &tt,
                                Arc::new(AtomicBool::new(false)),
                                false,
                                history_vec.clone(),
                                &mut search_data,
                                &params_clone,
                            );
                            search_score = s;
                            best_move = m;
                        }

                        // --- Adjudication Logic ---

                        // 1. Mercy Rule
                        if search_score.abs() >= MERCY_CP {
                            mercy_counter += 1;
                        } else {
                            mercy_counter = 0;
                        }

                        if mercy_counter >= MERCY_PLIES {
                            if search_score > 0 {
                                result_val = if state.side_to_move == WHITE {
                                    1.0
                                } else {
                                    0.0
                                };
                            } else {
                                result_val = if state.side_to_move == WHITE {
                                    0.0
                                } else {
                                    1.0
                                };
                            }
                            finished = true;
                            break;
                        }

                        // 2. Early Draw Adjudication
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

                        // Mate Check
                        if search_score.abs() > 20000 {
                            if search_score > 0 {
                                result_val = if state.side_to_move == WHITE {
                                    1.0
                                } else {
                                    0.0
                                };
                            } else {
                                result_val = if state.side_to_move == WHITE {
                                    0.0
                                } else {
                                    1.0
                                };
                            }
                            finished = true;
                            break;
                        }

                        let final_move = if let Some(m) = best_move {
                            m
                        } else {
                            // Fallback: Pick first legal.
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

                    if abort_game {
                        continue;
                    }

                    if finished {
                        let mut game_data = Vec::with_capacity(positions.len());
                        for (pos_state, score) in positions.drain(..) {
                            let board = convert_to_bullet(&pos_state, score, result_val);
                            game_data.push(board);
                        }

                        if !game_data.is_empty() {
                            tx.send(game_data).unwrap();
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
