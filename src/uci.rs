use crate::movegen::{self, MoveGenerator};
use crate::search;
use crate::state::{k, GameState, Move, K};
use crate::time::{TimeControl, TimeManager};
use crate::tt::TranspositionTable;
use crate::parameters::SearchParameters;
use std::io::{self, BufRead};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use std::thread;

use crate::syzygy;

// Global UCI Option
pub static UCI_CHESS960: AtomicBool = AtomicBool::new(false);

pub fn parse_move_wrapper(state: &GameState, move_str: &str) -> Option<Move> {
    parse_move(state, move_str)
}

pub fn uci_loop() {
    let stdin = io::stdin();
    let mut buffer = String::new();

    // Default 64MB Hash
    let mut tt = Arc::new(TranspositionTable::new(64));

    // Default Parameters
    let default_params = Arc::new(SearchParameters::default());

    let mut game_state =
        GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut game_history: Vec<u64> = Vec::new();
    game_history.push(game_state.hash);

    let mut num_threads = 1;
    let mut move_overhead = 10;

    if std::path::Path::new("nn-aether.nnue").exists() {
        crate::nnue::init_nnue("nn-aether.nnue");
        game_state.refresh_accumulator();
    }

    let stop_signal = Arc::new(AtomicBool::new(false));
    let mut search_threads: Vec<thread::JoinHandle<()>> = Vec::new();
    let global_nodes = Arc::new(AtomicU64::new(0));

    loop {
        buffer.clear();
        match stdin.lock().read_line(&mut buffer) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading stdin: {}", e);
                break;
            }
        }

        let cmd = buffer.trim();
        if cmd.is_empty() {
            continue;
        }

        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let command = parts[0];

        // Basic logging
        log::debug!("UCI Input: {}", cmd);

        match command {
            "uci" => {
                println!("id name Aether 1.0.0");
                println!("id author Basti Dangca");
                println!("option name Hash type spin default 64 min 1 max 1024");
                println!("option name Threads type spin default 1 min 1 max 64");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name Move Overhead type spin default 10 min 0 max 5000");
                println!("option name EvalFile type string default <empty>");
                println!("option name UCI_Chess960 type check default false");
                println!("uciok");
            }
            "isready" => println!("readyok"),
            "ucinewgame" => {
                if let Some(tt_mut) = Arc::get_mut(&mut tt) {
                    tt_mut.clear();
                } else {
                    tt = Arc::new(TranspositionTable::new(64));
                }

                game_state = GameState::parse_fen(
                    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                );
                game_history.clear();
                game_history.push(game_state.hash);
            }
            "position" => {
                game_history = handle_position(&mut game_state, &parts);
            }
            "go" => {
                stop_signal.store(true, Ordering::Relaxed);
                for h in search_threads.drain(..) {
                    h.join().unwrap();
                }

                stop_signal.store(false, Ordering::Relaxed);

                let limits = parse_go(game_state.side_to_move, &parts, move_overhead);

                global_nodes.store(0, Ordering::Relaxed);

                log::info!("Starting search with {} threads", num_threads);

                for i in 0..num_threads {
                    let mut safe_state = game_state;
                    safe_state.refresh_accumulator();

                    let stop_clone = stop_signal.clone();
                    let limits_clone = limits.clone();
                    let history_clone = game_history.clone();
                    let is_main = i == 0;
                    let tt_clone = tt.clone();
                    let params_clone = default_params.clone();
                    let global_nodes_clone = global_nodes.clone();

                    let builder = thread::Builder::new()
                        .name(format!("search_worker_{}", i))
                        .stack_size(8 * 1024 * 1024);

                    let handle = builder
                        .spawn(move || {
                            let mut search_data = search::SearchData::new();
                            search::search(
                                &safe_state,
                                limits_clone,
                                &tt_clone,
                                stop_clone,
                                is_main,
                                history_clone,
                                &mut search_data,
                                &params_clone,
                                Some(&global_nodes_clone),
                            );
                        });

                    if let Ok(h) = handle {
                        search_threads.push(h);
                    } else {
                        eprintln!("Failed to spawn search thread {}", i);
                    }
                }
            }
            "stop" => {
                stop_signal.store(true, Ordering::Relaxed);
                for h in search_threads.drain(..) {
                    h.join().unwrap();
                }
            }
            "setoption" => {
                if parts.len() >= 5 && parts[1] == "name" {
                    let mut value_idx = 0;
                    for i in 2..parts.len() {
                        if parts[i] == "value" {
                            value_idx = i;
                            break;
                        }
                    }

                    if value_idx > 2 {
                        let name_parts = &parts[2..value_idx];
                        let name = name_parts.join(" ");
                        let value = parts.get(value_idx + 1).unwrap_or(&"");

                        if name.eq_ignore_ascii_case("Hash") {
                            if let Ok(mb) = value.parse::<usize>() {
                                let clamped_mb = mb.clamp(1, 1024);
                                stop_signal.store(true, Ordering::Relaxed);
                                for h in search_threads.drain(..) {
                                    h.join().unwrap();
                                }
                                tt = Arc::new(TranspositionTable::new(clamped_mb));
                            }
                        } else if name.eq_ignore_ascii_case("Threads") {
                            if let Ok(t) = value.parse::<usize>() {
                                num_threads = t.clamp(1, 64);
                            }
                        } else if name.eq_ignore_ascii_case("Move Overhead") {
                            if let Ok(ov) = value.parse::<u128>() {
                                move_overhead = ov;
                            }
                        } else if name.eq_ignore_ascii_case("EvalFile") {
                            crate::nnue::init_nnue(value);
                            game_state.refresh_accumulator();
                        } else if name.eq_ignore_ascii_case("UCI_Chess960") {
                            let val = value.parse::<bool>().unwrap_or(false);
                            UCI_CHESS960.store(val, Ordering::Relaxed);
                        } else if name.eq_ignore_ascii_case("SyzygyPath") {
                            syzygy::init_tablebase(value);
                        }
                    }
                }
            }
            "quit" => {
                stop_signal.store(true, Ordering::Relaxed);
                for h in search_threads.drain(..) {
                    h.join().unwrap();
                }
                break;
            }
            _ => {}
        }
    }
}

fn handle_position(state: &mut GameState, parts: &[&str]) -> Vec<u64> {
    let mut move_index = 0;
    let mut history = Vec::new();

    if parts[1] == "startpos" {
        *state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        if parts.len() > 2 && parts[2] == "moves" {
            move_index = 3;
        }
    } else if parts[1] == "fen" {
        let mut fen = String::new();
        let mut i = 2;
        while i < parts.len() && parts[i] != "moves" {
            fen.push_str(parts[i]);
            fen.push(' ');
            i += 1;
        }
        *state = GameState::parse_fen(&fen);
        if i < parts.len() && parts[i] == "moves" {
            move_index = i + 1;
        }
    }

    history.push(state.hash);

    if move_index > 0 && move_index < parts.len() {
        for i in move_index..parts.len() {
            let move_str = parts[i];
            let parsed_move = parse_move(state, move_str);
            if let Some(mv) = parsed_move {
                *state = state.make_move(mv);
                history.push(state.hash);

                // Reset history on capture or pawn move (50-move rule reset)
                if mv.is_capture()
                    || (state.bitboards[crate::state::P].get_bit(mv.target())
                        || state.bitboards[crate::state::p].get_bit(mv.target()))
                {
                    history.clear();
                    history.push(state.hash);
                }
            }
        }
    }
    history
}

pub fn parse_move(state: &GameState, move_str: &str) -> Option<Move> {
    let mut generator = crate::movegen::MoveGenerator::new();
    generator.generate_moves(state);
    let src_str = &move_str[0..2];
    let tgt_str = &move_str[2..4];
    let src = square_from_str(src_str);
    let tgt = square_from_str(tgt_str);

    // COMPATIBILITY: Handle e1g1 as e1h1 if UCI_Chess960 is false (Standard Chess)
    // AND checks for castling validity.
    // Actually, simply: If input is e1g1 and we find e1h1 in legal moves, match it?
    // Be careful not to match e1g1 if e1g1 is actually valid (e.g. empty square move).
    // In standard chess, e1g1 is EITHER castling OR normal king move to g1.
    // If it's castling, the generator produces e1h1 (King takes Rook).
    // So if user sends e1g1, and generator has e1h1, and e1g1 is NOT in generator (because g1 empty/attacked/whatever), then we might map.
    // BUT safest way:
    // If standard chess (UCI_Chess960 == false), and we see e1g1:
    // 1. Check if e1g1 is in generated moves.
    // 2. Check if e1h1 (Kingside Castle) is in generated moves.
    // If e1h1 is there, and e1g1 matches the "Concept" of castling, map it.

    let promo = if move_str.len() > 4 {
        match move_str.chars().nth(4).unwrap() {
            'q' => Some(4),
            'r' => Some(3),
            'b' => Some(2),
            'n' => Some(1),
            _ => None,
        }
    } else {
        None
    };

    // First pass: Try exact match
    for i in 0..generator.list.count {
        let mv = generator.list.moves[i];
        if mv.source() == src && mv.target() == tgt {
            if let Some(p) = mv.promotion() {
                if let Some(user_p) = promo {
                    if p == user_p {
                        return Some(mv);
                    }
                }
            } else {
                if promo.is_none() {
                    return Some(mv);
                }
            }
        }
    }

    // Second pass: Standard Chess Castling Compatibility
    // Map e1g1 -> e1h1, e1c1 -> e1a1, e8g8 -> e8h8, e8c8 -> e8a8
    // ONLY if UCI_Chess960 is FALSE.
    if !UCI_CHESS960.load(Ordering::Relaxed) {
        let is_white = state.side_to_move == crate::state::WHITE;
        let king_sq = if is_white { 4 } else { 60 }; // e1 / e8

        if src == king_sq {
            let kingside_tgt = if is_white { 6 } else { 62 }; // g1 / g8
            let queenside_tgt = if is_white { 2 } else { 58 }; // c1 / c8

            let mut expected_rook_sq = 64;

            if tgt == kingside_tgt {
                expected_rook_sq = if is_white { 7 } else { 63 }; // h1 / h8
            } else if tgt == queenside_tgt {
                expected_rook_sq = if is_white { 0 } else { 56 }; // a1 / a8
            }

            if expected_rook_sq != 64 {
                // Check if King->Rook move exists in generator
                for i in 0..generator.list.count {
                    let mv = generator.list.moves[i];
                    if mv.source() == src && mv.target() == expected_rook_sq {
                        return Some(mv);
                    }
                }
            }
        }
    }

    None
}

fn square_from_str(s: &str) -> u8 {
    let bytes = s.as_bytes();
    let file = bytes[0] - b'a';
    let rank = bytes[1] - b'1';
    rank * 8 + file
}

fn parse_go(side: usize, parts: &[&str], overhead: u128) -> search::Limits {
    let mut depth: Option<u8> = None;
    let mut wtime: Option<u128> = None;
    let mut btime: Option<u128> = None;
    let mut winc: Option<u128> = None;
    let mut binc: Option<u128> = None;
    let mut movestogo: Option<u32> = None;
    let mut infinite = false;
    let mut movetime: Option<u128> = None;
    let mut nodes: Option<u64> = None;

    let mut i = 1;
    while i < parts.len() {
        match parts[i] {
            "depth" => {
                depth = parts[i + 1].parse().ok();
                i += 1;
            }
            "wtime" => {
                wtime = Some(parts[i + 1].parse().unwrap_or(0));
                i += 1;
            }
            "btime" => {
                btime = Some(parts[i + 1].parse().unwrap_or(0));
                i += 1;
            }
            "winc" => {
                winc = Some(parts[i + 1].parse().unwrap_or(0));
                i += 1;
            }
            "binc" => {
                binc = Some(parts[i + 1].parse().unwrap_or(0));
                i += 1;
            }
            "movestogo" => {
                movestogo = Some(parts[i + 1].parse().unwrap_or(30));
                i += 1;
            }
            "movetime" => {
                movetime = Some(parts[i + 1].parse().unwrap_or(1000));
                i += 1;
            }
            "infinite" => {
                infinite = true;
            }
            "nodes" => {
                nodes = Some(parts[i + 1].parse().unwrap_or(u64::MAX));
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }

    if infinite {
        return search::Limits::Infinite;
    }

    if let Some(n) = nodes {
        return search::Limits::FixedNodes(n);
    }

    if movetime.is_some() || wtime.is_some() || btime.is_some() {
        let tc = if let Some(mt) = movetime {
            TimeControl::MoveTime(mt)
        } else {
            TimeControl::GameTime {
                wtime: wtime.unwrap_or(0),
                btime: btime.unwrap_or(0),
                winc: winc.unwrap_or(0),
                binc: binc.unwrap_or(0),
                moves_to_go: movestogo,
            }
        };
        return search::Limits::FixedTime(TimeManager::new(tc, side, overhead));
    }

    if let Some(d) = depth {
        return search::Limits::FixedDepth(d);
    }

    search::Limits::Infinite
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_go_defaults() {
        let parts = vec!["go"];
        let limits = parse_go(0, &parts, 10);
        match limits {
            crate::search::Limits::Infinite => assert!(true),
            _ => assert!(false, "Expected Infinite"),
        }
    }

    #[test]
    fn test_parse_go_depth() {
        let parts = vec!["go", "depth", "10"];
        let limits = parse_go(0, &parts, 10);
        match limits {
            crate::search::Limits::FixedDepth(d) => assert_eq!(d, 10),
            _ => assert!(false, "Expected FixedDepth(10)"),
        }
    }

    #[test]
    fn test_parse_go_time() {
        let parts = vec!["go", "wtime", "1000", "btime", "2000"];
        let limits = parse_go(0, &parts, 10);
        match limits {
            crate::search::Limits::FixedTime(tm) => {
                 // Hard to check internal state of TimeManager without getters,
                 // but checking we got FixedTime is sufficient for basic robustness.
                 assert!(true);
            }
            _ => assert!(false, "Expected FixedTime"),
        }
    }

    #[test]
    fn test_square_from_str() {
        assert_eq!(square_from_str("a1"), 0);
        assert_eq!(square_from_str("h1"), 7);
        assert_eq!(square_from_str("e1"), 4);
        assert_eq!(square_from_str("a8"), 56);
        assert_eq!(square_from_str("h8"), 63);
    }
}
