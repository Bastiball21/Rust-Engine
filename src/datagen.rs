// src/datagen.rs
use crate::search;
use crate::state::{GameState, WHITE, BLACK};
use crate::time::{TimeControl, TimeManager};
use crate::tt::{TranspositionTable, FLAG_EXACT};
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::sync::{atomic::AtomicBool, Arc};
use std::thread;
use std::sync::mpsc;
use std::time::Instant;
use crate::bullet_helper::convert_to_bullet;
use bulletformat::BulletFormat;
use std::collections::HashMap;

pub struct DatagenConfig {
    pub num_games: usize,
    pub num_threads: usize,
    pub depth: u8,
    pub filename: String,
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
        if range == 0 { return min; }
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

pub fn run_datagen(config: DatagenConfig) {
    println!("Starting Datagen (High Quality Mode)");
    println!("  Games:    {}", config.num_games);
    println!("  Threads:  {}", config.num_threads);
    println!("  Depth:    {}", config.depth);
    println!("  Output:   {}", config.filename);
    println!("  Strategy: Node Limit (20k-50k), SplitMix64, TT Probe, Sync Channel");

    // Use a Bounded Channel to prevent memory explosion
    // 1000 games buffer should be plenty (approx 200MB if games are long, usually much less)
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

        // Iterate over the receiver. This blocks if empty, and terminates when all senders drop.
        for game_data in rx {
            bulletformat::ChessBoard::write_to_bin(&mut writer, &game_data).unwrap();
            games_written += 1;
            positions_written += game_data.len();

            if games_written % 50 == 0 {
                let elapsed = start_time.elapsed().as_secs_f64();
                let games_per_sec = games_written as f64 / elapsed;
                let pos_per_sec = positions_written as f64 / elapsed;

                let remaining_games = total_games.saturating_sub(games_written);
                let eta_secs = if games_per_sec > 0.0 {
                    remaining_games as f64 / games_per_sec
                } else {
                    0.0
                };
                let eta = std::time::Duration::from_secs_f64(eta_secs);

                println!(
                    "Written {} games ({:.1}%) ({} pos)... {:.1} games/s, {:.1} pos/s, Elapsed: {:.0}s, ETA: {:.0}s",
                    games_written,
                    (games_written as f64 / total_games as f64) * 100.0,
                    positions_written,
                    games_per_sec,
                    pos_per_sec,
                    elapsed,
                    eta.as_secs()
                );
            }
        }
        println!("Writer thread finished. Total games: {}, Total pos: {}", games_written, positions_written);
    });

    // Worker Threads
    let mut handles = vec![];
    let games_per_thread = config.num_games / config.num_threads;
    let remainder = config.num_games % config.num_threads;

    for t_id in 0..config.num_threads {
        let tx = tx.clone();
        let my_games = games_per_thread + if t_id < remainder { 1 } else { 0 };
        let depth_config = config.depth;

        let builder = thread::Builder::new()
            .name(format!("datagen_worker_{}", t_id))
            .stack_size(8 * 1024 * 1024);

        let handle = builder.spawn(move || {
            let mut tt = TranspositionTable::new(32); // 32MB per thread
            // Seed RNG with thread ID for diversity
            let mut rng = Rng::new(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64 ^ (t_id as u64).wrapping_mul(0xDEADBEEF));
            let mut search_data = search::SearchData::new();

            for _ in 0..my_games {
                search_data.clear();
                // Start from standard position
                let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                let mut positions: Vec<(GameState, i16)> = Vec::with_capacity(200);
                let mut result_val = 0.5;
                let mut finished = false;

                // Repetition History (using HashMap for O(1) check)
                let mut rep_history: HashMap<u64, u8> = HashMap::with_capacity(300);
                rep_history.insert(state.hash, 1);

                // For 3-fold repetition check, we need to know the count.
                let mut history_vec: Vec<u64> = Vec::with_capacity(300);
                history_vec.push(state.hash);


                // --- 1. Random Move Phase (8-9 plies) ---
                let random_plies = 8 + rng.range(0, 2); // 8 or 9
                let mut abort_game = false;

                for _ in 0..random_plies {
                    let mut moves = crate::movegen::MoveGenerator::new();
                    moves.generate_moves(&state);

                    if moves.list.count == 0 {
                        abort_game = true;
                        break;
                    }

                    // Las Vegas Style: Try random move, check legality.
                    // Fallback to filtering if luck fails.
                    let mut chosen_move = None;

                    // Try 16 times to pick a legal move randomly
                    for _ in 0..16 {
                        let idx = rng.range(0, moves.list.count);
                        let m = moves.list.moves[idx];
                        let next_state = state.make_move(m);
                        if !crate::search::is_check(&next_state, state.side_to_move) {
                            chosen_move = Some(m);
                            break;
                        }
                    }

                    if chosen_move.is_none() {
                        // Unlucky. Filter all valid moves.
                        let mut valid_moves = Vec::new();
                        for i in 0..moves.list.count {
                            let m = moves.list.moves[i];
                            let next_state = state.make_move(m);
                            if !crate::search::is_check(&next_state, state.side_to_move) {
                                valid_moves.push(m);
                            }
                        }
                        if valid_moves.is_empty() {
                            abort_game = true;
                            break;
                        }
                        chosen_move = Some(valid_moves[rng.range(0, valid_moves.len())]);
                    }

                    let m = chosen_move.unwrap();
                    state = state.make_move(m);

                    *rep_history.entry(state.hash).or_insert(0) += 1;
                    history_vec.push(state.hash);

                    if state.halfmove_clock >= 100 { abort_game = true; break; }
                }

                if abort_game {
                    tt.clear();
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
                            result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                        } else {
                            result_val = 0.5;
                        }
                        finished = true;
                        break;
                    }

                    // Node Limit Selection (20k - 50k)
                    // We can add some jitter to depth if requested, but node limit is dominant.
                    let node_limit = 20_000 + (rng.next_u64() % 30_000);

                    // TT Probe Logic: Check if we have an EXACT hit deep enough
                    let mut used_tt_hit = false;
                    let mut search_score = 0;
                    let mut best_move = None;

                    if let Some((tt_score, tt_depth, tt_flag, tt_move)) = tt.probe_data(state.hash) {
                         // If we have an exact score at >= target depth, skip search!
                         if tt_flag == FLAG_EXACT && tt_depth >= depth_config {
                             search_score = tt_score;
                             best_move = tt_move;
                             used_tt_hit = true;
                         }
                    }

                    if !used_tt_hit {
                        // Perform Search
                        // We use TimeManager::new with Infinite, relying on Node Limit to stop.
                        let tm = TimeManager::new(TimeControl::Infinite, state.side_to_move, 0);

                        // We do not pass `history_vec` to search here. Search uses history for cycle detection.
                        // Our `history_vec` is valid.
                        let (s, m) = search::search(
                            &state,
                            tm,
                            &tt,
                            Arc::new(AtomicBool::new(false)),
                            depth_config,
                            false, // not main thread (suppress output)
                            history_vec.clone(),
                            &mut search_data,
                            Some(node_limit),
                        );
                        search_score = s;
                        best_move = m;
                    }

                    // Score Validation & Clamping
                    // Check for Mate Scores. If found, game over.
                    if search_score.abs() > 20000 {
                         if search_score > 0 {
                             result_val = if state.side_to_move == WHITE { 1.0 } else { 0.0 };
                         } else {
                             result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                         }
                         finished = true;
                         break;
                    }

                    // Ensure we have a move
                    let final_move = if let Some(m) = best_move {
                        m
                    } else {
                        // Search failed or returned nothing? (Shouldn't happen with valid nodes)
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

                    // Store Position
                    // Score is strictly clamped to ensure safe casting
                    let clamped_score = search_score.clamp(-32000, 32000);

                    // Datagen stores White Relative Score
                    let white_score = if state.side_to_move == WHITE {
                        clamped_score
                    } else {
                        -clamped_score
                    };

                    positions.push((state.clone(), white_score as i16));

                    // Make Move
                    let next_state = state.make_move(final_move);

                    // Legality Sanity Check (should be redundant if search works)
                    if crate::search::is_check(&next_state, state.side_to_move) {
                        // Illegal move. Abort.
                        abort_game = true;
                        break;
                    }

                    state = next_state;
                    *rep_history.entry(state.hash).or_insert(0) += 1;
                    history_vec.push(state.hash);

                    if history_vec.len() > 600 {
                        result_val = 0.5;
                        finished = true;
                        break;
                    }
                }

                if abort_game {
                    tt.clear();
                    continue;
                }

                if finished {
                    let mut game_data = Vec::with_capacity(positions.len());
                    for (pos_state, score) in positions {
                        // Safe: score is clamped. Result is 0.0, 0.5, or 1.0.
                        let board = convert_to_bullet(&pos_state, score, result_val);
                        game_data.push(board);
                    }

                    if !game_data.is_empty() {
                         // This blocks if channel is full
                         tx.send(game_data).unwrap();
                    }
                }

                tt.clear();
            }
        }).unwrap();
        handles.push(handle);
    }

    // Close the sender on the main thread so the writer terminates when workers are done
    drop(tx);

    for h in handles {
        h.join().unwrap();
    }
    writer_handle.join().unwrap();
}
