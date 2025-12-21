// src/tuning.rs
use crate::eval::{self, Trace};
use crate::state::GameState;
use std::fs::File;
use std::io::{self, BufRead, Write};
#[cfg(feature = "tuning")]
use std::sync::atomic::Ordering;

const K_FACTOR: f64 = 1.13;
const LEARNING_RATE: f64 = 1.0;
const EPOCHS: usize = 2000;
const BATCH_SIZE: usize = 8192;

struct TunerEntry {
    state: GameState,
    result: f64,
    trace: Trace,
}

enum ParamType {
    MaterialMG(usize),
    MaterialEG(usize),
    PsqtMG(usize, usize),
    PsqtEG(usize, usize),
}

struct Parameter {
    val: f64,
    ptype: ParamType,
}

// Stub function when tuning is not enabled to satisfy linker if called
#[cfg(not(feature = "tuning"))]
pub fn run_tuning() {
    println!("Tuning not enabled. Build with --features tuning");
}

#[cfg(feature = "tuning")]
pub fn run_tuning() {
    println!("--- AETHER TUNER ---");
    let mut params = init_parameters();
    println!("Parameters to tune: {}", params.len());

    println!("Loading dataset 'quiet-labeled.epd'...");
    let mut entries = load_entries("quiet-labeled.epd");
    if entries.is_empty() {
        println!("No positions found. Please download 'quiet-labeled.epd' and place it in root.");
        return;
    }
    println!("Loaded {} positions.", entries.len());

    println!("Pre-computing evaluation traces...");
    for entry in &mut entries {
        let _ = eval::trace_evaluate(&entry.state, &mut entry.trace);
    }

    let mut best_error = 999.0;

    for epoch in 1..=EPOCHS {
        let mut total_error = 0.0;
        let mut gradients = vec![0.0; params.len()];
        let mut count = 0;

        for entry in &entries {
            let mut mg_score = entry.trace.fixed_mg as f64;
            let mut eg_score = entry.trace.fixed_eg as f64;
            let mut phase = 0;

            for &(idx, count) in &entry.trace.terms {
                if idx < params.len() {
                    if idx < 6 {
                        mg_score += params[idx].val * count as f64;
                    } else if idx < 12 {
                        eg_score += params[idx].val * count as f64;
                    } else {
                        let relative = idx - 12;
                        let block_offset = relative % 128;
                        if block_offset < 64 {
                            mg_score += params[idx].val * count as f64;
                        } else {
                            eg_score += params[idx].val * count as f64;
                        }
                    }
                }
            }

            for p in 0..6 {
                let c =
                    (entry.state.bitboards[p] | entry.state.bitboards[p + 6]).count_bits() as i32;
                phase += c * eval::PHASE_WEIGHTS[p];
            }
            phase = phase.clamp(0, 24);

            let score = (mg_score * phase as f64 + eg_score * (24 - phase) as f64) / 24.0;

            let sigmoid = 1.0 / (1.0 + 10.0f64.powf(-K_FACTOR * score / 400.0));
            let error = entry.result - sigmoid;
            total_error += error * error;

            let gradient_term = (error) * sigmoid * (1.0 - sigmoid);

            for &(idx, count) in &entry.trace.terms {
                let mg_weight = phase as f64 / 24.0;
                let eg_weight = (24 - phase) as f64 / 24.0;
                let local_grad = gradient_term * count as f64;

                if idx < 6 {
                    gradients[idx] += local_grad * mg_weight;
                } else if idx < 12 {
                    gradients[idx] += local_grad * eg_weight;
                } else {
                    let relative = idx - 12;
                    let block_offset = relative % 128;
                    if block_offset < 64 {
                        gradients[idx] += local_grad * mg_weight;
                    } else {
                        gradients[idx] += local_grad * eg_weight;
                    }
                }
            }

            count += 1;
            if count % BATCH_SIZE == 0 {
                for i in 0..params.len() {
                    params[i].val += gradients[i] * LEARNING_RATE / BATCH_SIZE as f64;
                    gradients[i] = 0.0;
                }
            }
        }

        let mse = total_error / entries.len() as f64;
        if mse < best_error {
            best_error = mse;
            println!("Epoch {} Best MSE: {:.6}", epoch, mse);
            save_parameters(&params);
        } else {
            println!("Epoch {} MSE: {:.6}", epoch, mse);
        }
    }
}

#[cfg(feature = "tuning")]
fn init_parameters() -> Vec<Parameter> {
    let mut params = Vec::new();
    // Use Relaxed loading from Atomics
    for i in 0..6 {
        params.push(Parameter {
            val: eval::MG_VALS[i].load(Ordering::Relaxed) as f64,
            ptype: ParamType::MaterialMG(i),
        });
    }
    for i in 0..6 {
        params.push(Parameter {
            val: eval::EG_VALS[i].load(Ordering::Relaxed) as f64,
            ptype: ParamType::MaterialEG(i),
        });
    }

    // Helper using generic types to avoid specifying exact atomic type, relying on inference or explicit cast at call site?
    // Actually, eval::* tables are strictly defined.
    // The previous error was: expected `&[AtomicI32; 64]`, found `&[i32; 64]`.
    // This is because I imported eval::*, and MG_PAWN_TABLE is conditionally defined.
    // When tuning is ENABLED, MG_PAWN_TABLE is [AtomicI32; 64].
    // The previous check failed because `cargo check` ran WITHOUT features enabled?
    // Ah, `cargo check` by default uses default features. I didn't enable `tuning` by default.
    // So `eval.rs` defined `i32` arrays.
    // But `tuning.rs` code (which I removed cfg guards from initially or had partial guards) tried to use `AtomicI32` specific logic?
    // Wait, the error `no method named store found for type i32` confirms that `eval::*` were `i32` arrays.
    // And `init_parameters` (and `save_parameters`) were trying to call `.store` or pass them to closures expecting `AtomicI32`.
    // So the fix is to wrap the entire body of `init_parameters` and `save_parameters` (and `run_tuning` logic) in `#[cfg(feature = "tuning")]`.
    // Or make them generic?
    // Since tuning is a specific mode, conditional compilation is cleaner.

    let mut add_table = |mg_table: &[std::sync::atomic::AtomicI32; 64],
                         eg_table: &[std::sync::atomic::AtomicI32; 64],
                         piece_idx: usize| {
        for sq in 0..64 {
            params.push(Parameter {
                val: mg_table[sq].load(Ordering::Relaxed) as f64,
                ptype: ParamType::PsqtMG(piece_idx, sq),
            });
        }
        for sq in 0..64 {
            params.push(Parameter {
                val: eg_table[sq].load(Ordering::Relaxed) as f64,
                ptype: ParamType::PsqtEG(piece_idx, sq),
            });
        }
    };

    add_table(&eval::MG_PAWN_TABLE, &eval::EG_PAWN_TABLE, 0);
    add_table(&eval::MG_KNIGHT_TABLE, &eval::EG_KNIGHT_TABLE, 1);
    add_table(&eval::MG_BISHOP_TABLE, &eval::EG_BISHOP_TABLE, 2);
    add_table(&eval::MG_ROOK_TABLE, &eval::EG_ROOK_TABLE, 3);
    add_table(&eval::MG_QUEEN_TABLE, &eval::EG_QUEEN_TABLE, 4);
    add_table(&eval::MG_KING_TABLE, &eval::EG_KING_TABLE, 5);
    params
}

#[cfg(feature = "tuning")]
fn load_entries(path: &str) -> Vec<TunerEntry> {
    let mut entries = Vec::new();
    if let Ok(file) = File::open(path) {
        for line in io::BufReader::new(file).lines().flatten() {
            let parts: Vec<&str> = line.split(" c").collect();
            if parts.len() < 2 {
                continue;
            }
            let fen = parts[0];
            let result_part = parts[1];

            let result = if result_part.contains("\"1.0\"") {
                1.0
            } else if result_part.contains("\"0.5\"") {
                0.5
            } else {
                0.0
            };

            entries.push(TunerEntry {
                state: GameState::parse_fen(fen),
                result,
                trace: Trace::new(),
            });
        }
    }
    entries
}

#[cfg(feature = "tuning")]
fn save_parameters(params: &[Parameter]) {
    for p in params {
        let int_val = p.val as i32;
        match p.ptype {
            ParamType::MaterialMG(i) => eval::MG_VALS[i].store(int_val, Ordering::Relaxed),
            ParamType::MaterialEG(i) => eval::EG_VALS[i].store(int_val, Ordering::Relaxed),
            ParamType::PsqtMG(piece, sq) => match piece {
                0 => eval::MG_PAWN_TABLE[sq].store(int_val, Ordering::Relaxed),
                1 => eval::MG_KNIGHT_TABLE[sq].store(int_val, Ordering::Relaxed),
                2 => eval::MG_BISHOP_TABLE[sq].store(int_val, Ordering::Relaxed),
                3 => eval::MG_ROOK_TABLE[sq].store(int_val, Ordering::Relaxed),
                4 => eval::MG_QUEEN_TABLE[sq].store(int_val, Ordering::Relaxed),
                5 => eval::MG_KING_TABLE[sq].store(int_val, Ordering::Relaxed),
                _ => {}
            },
            ParamType::PsqtEG(piece, sq) => match piece {
                0 => eval::EG_PAWN_TABLE[sq].store(int_val, Ordering::Relaxed),
                1 => eval::EG_KNIGHT_TABLE[sq].store(int_val, Ordering::Relaxed),
                2 => eval::EG_BISHOP_TABLE[sq].store(int_val, Ordering::Relaxed),
                3 => eval::EG_ROOK_TABLE[sq].store(int_val, Ordering::Relaxed),
                4 => eval::EG_QUEEN_TABLE[sq].store(int_val, Ordering::Relaxed),
                5 => eval::EG_KING_TABLE[sq].store(int_val, Ordering::Relaxed),
                _ => {}
            },
        }
    }

    let mut file = File::create("tuned_params.txt").unwrap();
    writeln!(file, "Tuned Values Saved.").unwrap();
}
