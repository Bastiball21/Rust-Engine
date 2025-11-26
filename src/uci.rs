use std::io::{self, BufRead};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use crate::state::{GameState, Move, WHITE, BLACK};
use crate::search;
use crate::tt::TranspositionTable;
use crate::movegen::{self, MoveGenerator};
use crate::time::{TimeManager, TimeControl};

pub fn uci_loop() {
    let stdin = io::stdin();
    let mut buffer = String::new();
    
    let mut tt = TranspositionTable::new(64); 
    let mut game_state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    
    let stop_signal = Arc::new(AtomicBool::new(false));
    let mut search_thread: Option<thread::JoinHandle<()>> = None;

    crate::zobrist::init_zobrist();
    crate::bitboard::init_magic_tables();
    crate::movegen::init_move_tables();

    loop {
        buffer.clear();
        match stdin.lock().read_line(&mut buffer) {
            Ok(0) => break,
            Ok(_) => {},
            Err(_) => break,
        }

        let cmd = buffer.trim();
        if cmd.is_empty() { continue; }
        
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        let command = parts[0];

        match command {
            "uci" => {
                println!("id name Flash");
                println!("id author bastiball");
                println!("option name Hash type spin default 64 min 1 max 1024");
                println!("uciok");
            },
            "isready" => println!("readyok"),
            "ucinewgame" => {
                game_state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                tt = TranspositionTable::new(64); 
            },
            "position" => {
                handle_position(&mut game_state, &parts);
            },
            "go" => {
                stop_signal.store(true, Ordering::Relaxed);
                if let Some(h) = search_thread.take() { h.join().unwrap(); }
                
                stop_signal.store(false, Ordering::Relaxed);

                let state_clone = game_state;
                let stop_clone = stop_signal.clone();
                
                let (tm, depth) = parse_go(game_state.side_to_move, &parts);
                
                let tt_ref: &'static mut TranspositionTable = unsafe { std::mem::transmute(&mut tt) };

                search_thread = Some(thread::spawn(move || {
                    // FIXED: Added 'depth' argument here
                    search::search(&state_clone, tm, tt_ref, stop_clone, depth);
                }));
            },
            "stop" => {
                stop_signal.store(true, Ordering::Relaxed);
                if let Some(h) = search_thread.take() { h.join().unwrap(); }
            },
            "setoption" => {
                if parts.len() > 4 && parts[1] == "name" && parts[2] == "Hash" && parts[3] == "value" {
                    if let Ok(mb) = parts[4].parse::<usize>() {
                        tt = TranspositionTable::new(mb);
                    }
                }
            },
            "quit" => {
                stop_signal.store(true, Ordering::Relaxed);
                if let Some(h) = search_thread.take() { h.join().unwrap(); }
                break;
            },
            _ => {}
        }
    }
}

fn handle_position(state: &mut GameState, parts: &[&str]) {
    let mut move_index = 0;
    
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

    if move_index > 0 && move_index < parts.len() {
        for i in move_index..parts.len() {
            let move_str = parts[i];
            let parsed_move = parse_move(state, move_str);
            if let Some(mv) = parsed_move {
                *state = state.make_move(mv);
            }
        }
    }
}

fn parse_move(state: &GameState, move_str: &str) -> Option<Move> {
    let mut generator = crate::movegen::MoveGenerator::new();
    generator.generate_moves(state);
    let src_str = &move_str[0..2];
    let tgt_str = &move_str[2..4];
    let src = square_from_str(src_str);
    let tgt = square_from_str(tgt_str);
    let promo = if move_str.len() > 4 {
        match move_str.chars().nth(4).unwrap() {
            'q' => Some(4), 'r' => Some(3), 'b' => Some(2), 'n' => Some(1), _ => None,
        }
    } else { None };

    for i in 0..generator.list.count {
        let mv = generator.list.moves[i];
        if mv.source == src && mv.target == tgt {
            if let Some(p) = mv.promotion {
                if let Some(user_p) = promo { return Some(mv); }
            } else {
                if promo.is_none() { return Some(mv); }
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

fn parse_go(side: usize, parts: &[&str]) -> (TimeManager, u8) {
    let mut depth = 64;
    let mut wtime: Option<u128> = None;
    let mut btime: Option<u128> = None;
    let mut winc: Option<u128> = None;
    let mut binc: Option<u128> = None;
    let mut movestogo: Option<u32> = None;
    let mut infinite = false;
    let mut movetime: Option<u128> = None;

    let mut i = 1;
    while i < parts.len() {
        match parts[i] {
            "depth" => { depth = parts[i+1].parse().unwrap_or(64); i += 1; },
            "wtime" => { wtime = Some(parts[i+1].parse().unwrap_or(0)); i += 1; },
            "btime" => { btime = Some(parts[i+1].parse().unwrap_or(0)); i += 1; },
            "winc" => { winc = Some(parts[i+1].parse().unwrap_or(0)); i += 1; },
            "binc" => { binc = Some(parts[i+1].parse().unwrap_or(0)); i += 1; },
            "movestogo" => { movestogo = Some(parts[i+1].parse().unwrap_or(30)); i += 1; },
            "movetime" => { movetime = Some(parts[i+1].parse().unwrap_or(1000)); i += 1; },
            "infinite" => { infinite = true; },
            _ => {}
        }
        i += 1;
    }

    let tc = if infinite {
        TimeControl::Infinite
    } else if let Some(mt) = movetime {
        TimeControl::MoveTime(mt)
    } else if wtime.is_some() || btime.is_some() {
        TimeControl::GameTime {
            wtime: wtime.unwrap_or(0),
            btime: btime.unwrap_or(0),
            winc: winc.unwrap_or(0),
            binc: binc.unwrap_or(0),
            moves_to_go: movestogo,
        }
    } else {
        TimeControl::Infinite
    };

    (TimeManager::new(tc, side), depth)
}