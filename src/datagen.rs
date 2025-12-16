// src/datagen.rs
use crate::search;
use crate::state::{GameState, WHITE, BLACK};
use crate::time::{TimeControl, TimeManager};
use crate::tt::TranspositionTable;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::sync::{atomic::AtomicBool, Arc};
use std::thread;
use std::sync::mpsc;
use std::time::Instant;
use crate::bullet_helper::convert_to_bullet;
use bulletformat::BulletFormat;

pub struct DatagenConfig {
    pub num_games: usize,
    pub num_threads: usize,
    pub depth: u8,
    pub filename: String,
}

// Simple Xorshift RNG for internal use
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn range(&mut self, min: usize, max: usize) -> usize {
        let range = max - min;
        if range == 0 { return min; }
        (self.next_u64() as usize % range) + min
    }
}

pub fn run_datagen(config: DatagenConfig) {
    println!("Starting Datagen with:");
    println!("  Games:   {}", config.num_games);
    println!("  Threads: {}", config.num_threads);
    println!("  Depth:   {}", config.depth);
    println!("  Output:  {}", config.filename);

    let (tx, rx) = mpsc::channel::<Vec<bulletformat::ChessBoard>>();

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

            if games_written % 10 == 0 {
                let elapsed = start_time.elapsed();
                let elapsed_secs = elapsed.as_secs_f64();
                let games_per_sec = games_written as f64 / elapsed_secs;
                let pos_per_sec = positions_written as f64 / elapsed_secs;

                let remaining_games = total_games.saturating_sub(games_written);
                let eta_secs = if games_per_sec > 0.0 {
                    remaining_games as f64 / games_per_sec
                } else {
                    0.0
                };
                let eta = std::time::Duration::from_secs_f64(eta_secs);

                println!(
                    "Written {} games ({:.2}%) ({} pos)... {:.2} games/s, {:.2} pos/s, Elapsed: {:.0}s, ETA: {:.0}s",
                    games_written,
                    (games_written as f64 / total_games as f64) * 100.0,
                    positions_written,
                    games_per_sec,
                    pos_per_sec,
                    elapsed_secs,
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
        let depth = config.depth;

        let handle = thread::spawn(move || {
            let mut tt = TranspositionTable::new(16);
            let stop = Arc::new(AtomicBool::new(false));
            let mut rng = Rng::new(123456789 + t_id as u64);

            for _ in 0..my_games {
                let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                let mut positions: Vec<(GameState, i16)> = Vec::new();
                let mut result_val = 0.5;
                let mut finished = false;

                // Repetition History (Hash stack)
                let mut history: Vec<u64> = Vec::with_capacity(300);
                history.push(state.hash);

                // --- 1. Random Mover (Diversity) ---
                let random_plies = 8 + rng.range(0, 4); // 8 to 11 plies
                let mut abort_game = false;

                for _ in 0..random_plies {
                    let mut moves = crate::movegen::MoveGenerator::new();
                    moves.generate_moves(&state);

                    if moves.list.count == 0 {
                        abort_game = true;
                        break;
                    }

                    // Pick random valid move
                    let mut valid_moves = Vec::new();
                    for i in 0..moves.list.count {
                        let m = moves.list.moves[i];
                        // Basic legality check (does not leave king in check)
                        let next_state = state.make_move(m);
                        // We need to check if we left our king in check.
                        // Since make_move switches side, we check if `next_state` has the *previous* side (us) in check.
                        // `is_in_check` checks the side to move.
                        // So `is_in_check(&next_state)` checks if the *opponent* is in check.
                        // We need `is_square_attacked` for our king.
                        // The engine's `is_in_check` checks `state.side_to_move`.
                        // Wait, `make_move` updates `side_to_move`.
                        // If we are WHITE, make_move sets side to BLACK.
                        // If we verify legality, we need to ensure WHITE king is not attacked.
                        // The safest way is to use `movegen`'s legal generation if available, or manual check.
                        // For speed here, let's just trust `generate_moves` + pseudo-legal filter?
                        // `generate_moves` generates pseudo-legal.
                        // We must verify legality.

                        // Let's use `search::is_in_check` on the *original* side? No.
                        // `state` is before move.
                        // `next_state` is after move.
                        // Check if King of (next_state.side_to_move ^ 1) is attacked.

                        let us = state.side_to_move;
                        // In `next_state`, `side_to_move` is `them`.
                        // We check if `us` King is attacked.
                        if !crate::search::is_check(&next_state, us) {
                             valid_moves.push(m);
                        }
                    }

                    if valid_moves.is_empty() {
                        abort_game = true;
                        break;
                    }

                    let rand_idx = rng.range(0, valid_moves.len());
                    let chosen_move = valid_moves[rand_idx];
                    state = state.make_move(chosen_move);
                    history.push(state.hash);

                    if state.halfmove_clock >= 100 { abort_game = true; break; }
                }

                if abort_game {
                    tt.clear();
                    continue;
                }

                // --- 2. Real Search Loop ---
                let mut prev_score: i32 = 0; // Relative to side to move
                let mut first_move = true;

                loop {
                    // Check draw by repetition
                    // Check if hash appears twice in history (excluding current)
                    let mut rep_count = 0;
                    for &h in &history {
                        if h == state.hash { rep_count += 1; }
                    }
                    if rep_count >= 3 {
                         result_val = 0.5; finished = true; break;
                    }
                    if state.halfmove_clock >= 100 {
                         result_val = 0.5; finished = true; break;
                    }

                    let mut moves = crate::movegen::MoveGenerator::new();
                    moves.generate_moves(&state);

                    // Game Over Check
                    if moves.list.count == 0 {
                        if crate::search::is_in_check(&state) {
                            result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                        } else {
                            result_val = 0.5;
                        }
                        finished = true;
                        break;
                    }

                    // SEARCH
                    // Hard depth limit.
                    let tm = TimeManager::new(TimeControl::MoveTime(1000), state.side_to_move, 0);
                    // MoveTime is high to ensure depth is reached, but we rely on depth limit.

                    search::search(&state, tm, &tt, stop.clone(), depth, false, vec![]);

                    let (tt_score, _, _, best_move_opt) =
                        tt.probe_data(state.hash).unwrap_or((0, 0, 0, None));

                    // Adjudication
                    if tt_score.abs() > 20000 {
                        // Mate score
                        if tt_score > 0 {
                            result_val = if state.side_to_move == WHITE { 1.0 } else { 0.0 };
                        } else {
                            result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                        }
                        finished = true;
                        break;
                    }

                    // Blunder Check (Stability)
                    // `tt_score` is relative to side to move.
                    // `prev_score` was relative to previous side to move.
                    // If we made a move, `tt_score` (for us now) should be roughly `-prev_score`.
                    // Wait, `tt_score` is from the perspective of `state.side_to_move`.
                    // After we made a move, it's the opponent's turn.
                    // So `tt_score` is opponent's advantage.
                    // If we played well, `tt_score` should be `-prev_score`.
                    // If `tt_score` is much HIGHER than `-prev_score` (opponent is winning more), we blundered.
                    // Example: Prev (Us) = +100. Best move -> Opponent sees -100.
                    // If Opponent sees +200, we blundered (swing of 300).

                    // Scores are stored relative.
                    // Check: `tt_score` vs `-prev_score`.
                    // Diff = `tt_score - (-prev_score)` = `tt_score + prev_score`.
                    // If Diff > 300, it means Opponent is doing 300cp better than expected.

                    // if !first_move {
                    //     let diff = tt_score + prev_score; // e.g. -100 + 100 = 0 (Good). +200 + 100 = 300 (Bad).
                    //     if diff > 1000 {
                    //         // Blunder detected or unstable.
                    //         // Discard game.
                    //         abort_game = true;
                    //         break;
                    //     }
                    // }

                    let best_move = if let Some(m) = best_move_opt {
                        m
                    } else {
                        // Search failed to return move? Should not happen if legal moves exist.
                        // Fallback to first legal move
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

                    // Save Data (only if quiet? or all?)
                    // "No illegal, duplicated, or trivially drawn"
                    // We handle duplication via hash check above (roughly).
                    // We handle illegal via movegen.
                    // We will save valid positions.

                    // White Relative Score for file
                    let white_score = if state.side_to_move == WHITE { tt_score } else { -tt_score };

                    // Don't save if in check? (Noise) - User didn't specify, but often good to skip checks.
                    // We'll save everything for now to maximize data.
                    positions.push((state.clone(), white_score as i16));

                    // Make Move
                    // We need to verify legality of best_move (search usually returns legal, but...)
                    let next_state = state.make_move(best_move);
                    if crate::search::is_check(&next_state, state.side_to_move) {
                         // Search returned illegal move? Bug or TTHit bad?
                         // Abort game.
                         abort_game = true;
                         break;
                    }

                    state = next_state;
                    history.push(state.hash);
                    prev_score = tt_score;
                    first_move = false;

                    if history.len() > 400 {
                         // Too long
                         result_val = 0.5; finished = true; break;
                    }
                }

                if abort_game {
                    tt.clear();
                    continue;
                }

                if finished {
                    let mut game_data = Vec::with_capacity(positions.len());
                    for (pos_state, score) in positions {
                        let board = convert_to_bullet(&pos_state, score, result_val);
                        game_data.push(board);
                    }

                    if !game_data.is_empty() {
                         tx.send(game_data).unwrap();
                    }
                }

                tt.clear();
            }
        });
        handles.push(handle);
    }
    drop(tx);
    for h in handles { h.join().unwrap(); }
    writer_handle.join().unwrap();
}
