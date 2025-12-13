// src/datagen.rs
use crate::search;
use crate::state::{GameState, WHITE};
use crate::time::{TimeControl, TimeManager};
use crate::tt::TranspositionTable;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::sync::{atomic::AtomicBool, Arc};
use std::thread;
use std::sync::mpsc;
use crate::bullet_helper::convert_to_bullet;
use bulletformat::BulletFormat;

pub struct DatagenConfig {
    pub num_games: usize,
    pub num_threads: usize,
    pub depth: u8,
    pub filename: String,
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
    let writer_handle = thread::spawn(move || {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(filename)
            .expect("Unable to open output file");

        let mut writer = BufWriter::new(file);
        let mut games_written = 0;

        for game_data in rx {
            bulletformat::ChessBoard::write_to_bin(&mut writer, &game_data).unwrap();
            games_written += 1;
            if games_written % 10 == 0 {
                println!("Written {} games...", games_written);
            }
        }
        println!("Writer thread finished. Total games written: {}", games_written);
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
            // Each thread gets its own TT
            let mut tt = TranspositionTable::new(16); // 16MB per thread is reasonable
            let stop = Arc::new(AtomicBool::new(false));

            for i in 0..my_games {
                let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                let mut positions: Vec<(GameState, i16)> = Vec::new();
                let mut result_val = 0.5;
                let mut finished = false;

                loop {
                    let mut moves = crate::movegen::MoveGenerator::new();
                    moves.generate_moves(&state);

                    // 1. Check Game Over
                    if moves.list.count == 0 {
                        if crate::search::is_in_check(&state) {
                            result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                        } else {
                            result_val = 0.5;
                        }
                        finished = true;
                        break;
                    }
                    if state.halfmove_clock >= 100 {
                        result_val = 0.5;
                        finished = true;
                        break;
                    }

                    // 2. Set Depth (Random opening vs Real search)
                    let current_depth = if state.fullmove_number < 8 { 1 } else { depth };

                    // For Depth 1 (randomization), we might want even more randomness?
                    // The original code used Depth 1.

                    let tm = TimeManager::new(TimeControl::MoveTime(50), state.side_to_move, 0);
                    // search::search expects a stop boolean.
                    search::search(&state, tm, &tt, stop.clone(), current_depth, false, vec![]);

                    // 3. Get Score & Best Move
                    let (tt_score, _, _, best_move_opt) =
                        tt.probe_data(state.hash).unwrap_or((0, 0, 0, None));

                    let best_move = if let Some(m) = best_move_opt {
                        m
                    } else if moves.list.count > 0 {
                         moves.list.moves[0]
                    } else {
                        // Should not happen if count > 0 checked above
                        break;
                    };

                    // --- ADJUDICATION ---
                    if tt_score.abs() > 20000 {
                        if tt_score > 0 {
                            result_val = if state.side_to_move == WHITE { 1.0 } else { 0.0 };
                        } else {
                            result_val = if state.side_to_move == WHITE { 0.0 } else { 1.0 };
                        }
                        finished = true;
                        break;
                    }

                    // Save Data: FEN | SCORE | RESULT
                    // White relative score for storage?
                    // Original code:
                    // let white_relative_score = if state.side_to_move == WHITE { tt_score } else { -tt_score };

                    let white_relative_score = if state.side_to_move == WHITE {
                        tt_score
                    } else {
                        -tt_score
                    };

                    // Only save positions after opening phase to ensure quality/diversity
                    if state.fullmove_number > 8 {
                        positions.push((state.clone(), white_relative_score as i16));
                    }

                    state = state.make_move(best_move);
                }

                if finished {
                    // Buffer for this game
                    let mut game_data = Vec::with_capacity(positions.len());
                    for (pos_state, score) in positions {
                        let board = convert_to_bullet(&pos_state, score, result_val);
                        game_data.push(board);
                    }

                    // Send to writer
                    if !game_data.is_empty() {
                         tx.send(game_data).unwrap();
                    }
                }

                tt.clear();
            }
        });
        handles.push(handle);
    }

    // Close sender in main thread (not needed, but good practice to drop if we held one)
    drop(tx);

    for h in handles {
        h.join().unwrap();
    }
    writer_handle.join().unwrap();
}
