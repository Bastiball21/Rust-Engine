use crate::movegen::MoveGenerator;
use crate::state::{r, GameState, Move, B, K, N, P, Q, R};
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

pub struct Book {
    pub positions: Vec<GameState>,
}

impl Book {
    pub fn new() -> Self {
        Book {
            positions: Vec::new(),
        }
    }

    pub fn load_from_file(path: &str, start_ply: usize) -> io::Result<Self> {
        if path.ends_with(".pgn") {
            Self::load_pgn(path, start_ply)
        } else {
            Self::load_epd(path)
        }
    }

    fn load_epd(path: &str) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut positions = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let fen = line.split(';').next().unwrap_or(&line).trim(); // Handle EPD ops
            if !fen.is_empty() {
                positions.push(GameState::parse_fen(fen));
            }
        }
        println!("Loaded {} positions from EPD/FEN", positions.len());
        Ok(Book { positions })
    }

    fn load_pgn(path: &str, start_ply: usize) -> io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut positions = Vec::new();

        let mut pgn_content = String::new();
        for line in reader.lines() {
            let line = line?;
            // Simple logic: Accumulate lines. If line starts with "1. ", it's moves.
            // But PGNs can be multiline.
            // Robust approach: Read entire file (or chunk), split by [Event ...
            // Since we need to stream large files potentially, let's process game by game.
            // A game ends with result (1-0, 0-1, 1/2-1/2, *)
            pgn_content.push_str(&line);
            pgn_content.push('\n');
        }

        // Split by Event tag is common, but weak.
        // Let's rely on the fact that moves are in a block.
        // We will do a simple pass: Extract text between headers?
        // Or just use a simple state machine:
        // 1. Reading Headers (starts with [)
        // 2. Reading Moves (starts with non-[)

        // Actually, many PGNs have headers then empty line then moves.
        // Let's iterate lines again properly.

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut in_moves = false;
        let mut current_moves = String::new();

        for line in reader.lines() {
            let line = line?;
            let trimmed = line.trim();

            if trimmed.is_empty() {
                continue;
            }

            if trimmed.starts_with('[') {
                if in_moves {
                    // Previous game finished
                    if !current_moves.is_empty() {
                        if let Some(pos) = process_pgn_game(&current_moves, start_ply) {
                            positions.push(pos);
                        }
                        current_moves.clear();
                    }
                    in_moves = false;
                }
            } else {
                in_moves = true;
                current_moves.push_str(trimmed);
                current_moves.push(' ');
            }
        }

        // Last game
        if !current_moves.is_empty() {
            if let Some(pos) = process_pgn_game(&current_moves, start_ply) {
                positions.push(pos);
            }
        }

        println!("Loaded {} positions from PGN", positions.len());
        Ok(Book { positions })
    }
}

fn process_pgn_game(move_text: &str, stop_ply: usize) -> Option<GameState> {
    let mut state =
        GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    // Remove comments { ... } and ( ... )
    // Simple regex replacement is hard without regex crate.
    // Manual state machine to strip comments.
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

    // Split by whitespace and ignore move numbers "1."
    let tokens: Vec<&str> = clean_text.split_whitespace().collect();
    let mut ply = 0;

    for token in tokens {
        if ply >= stop_ply {
            break;
        }

        // Skip move numbers (e.g. "1.", "1...")
        if token.ends_with('.') {
            continue;
        }

        // Result?
        if token == "1-0" || token == "0-1" || token == "1/2-1/2" || token == "*" {
            break;
        }

        // Parse SAN
        if let Some(mv) = parse_san(&state, token) {
            state = state.make_move(mv);
            ply += 1;
        } else {
            // Failed to parse move (maybe recursive variation end? or bad PGN)
            // println!("Failed to parse move: {}", token);
            // If we fail, do we abort the game or just return current state?
            // If we haven't reached stop_ply, it's a short game or error.
            // If short game, return final position?
            // Let's break.
            break;
        }
    }

    // Only return if we made at least some moves? Or if we reached stop_ply?
    // User said "Select a configurable ply range".
    // If game is shorter than ply range, we can return the end position.
    Some(state)
}

fn parse_san(state: &GameState, san: &str) -> Option<Move> {
    // Clean SAN (remove +, #, etc)
    let san = san.trim_matches(|c| c == '+' || c == '#' || c == '!' || c == '?');

    if san == "O-O" || san == "0-0" {
        return find_castling_move(state, true);
    }
    if san == "O-O-O" || san == "0-0-0" {
        return find_castling_move(state, false);
    }

    let mut generator = MoveGenerator::new();
    generator.generate_moves(state);

    // 1. Piece Type
    let first_char = san.chars().next()?;
    let piece_type = match first_char {
        'N' => N,
        'B' => B,
        'R' => R,
        'Q' => Q,
        'K' => K,
        _ => P, // Pawn moves don't start with P usually, but check for files a-h
    };

    // If piece type is P, first char is file.
    let mut file_constraint: Option<u8> = None;
    let mut rank_constraint: Option<u8> = None;
    let target_sq: u8;
    let mut promotion: Option<usize> = None;

    // Remaining string to parse for disambiguation and target
    let mut remainder = if piece_type == P { san } else { &san[1..] };

    // Promotion? (ends with =Q, Q, etc)
    if let Some(idx) = remainder.find('=') {
        let promo_char = remainder.chars().nth(idx + 1)?;
        promotion = match promo_char {
            'N' => Some(1),
            'B' => Some(2),
            'R' => Some(3),
            'Q' => Some(4),
            _ => None,
        };
        remainder = &remainder[..idx];
    } else {
        // implicit promotion? e.g. a8Q
        // Check last char
        let last = remainder.chars().last()?;
        if "NBRQ".contains(last) {
            promotion = match last {
                'N' => Some(1),
                'B' => Some(2),
                'R' => Some(3),
                'Q' => Some(4),
                _ => None,
            };
            remainder = &remainder[..remainder.len() - 1];
        }
    }

    // Target Square: Last 2 chars of remainder should be file/rank
    if remainder.len() < 2 {
        return None;
    }
    let tgt_str = &remainder[remainder.len() - 2..];
    if let (Some(f), Some(rnk)) = (
        parse_file(tgt_str.chars().nth(0)?),
        parse_rank(tgt_str.chars().nth(1)?),
    ) {
        target_sq = rnk * 8 + f;
    } else {
        return None;
    }

    // Disambiguation: Check chars between start and target
    // e.g. Nbd7 -> remainder "bd7", target "d7", disam "b"
    // e.g. R1e3 -> remainder "1e3", target "e3", disam "1"
    let disam_len = remainder.len() - 2;
    if disam_len > 0 {
        let disam = &remainder[..disam_len];
        // x is capture, ignore
        let disam = disam.replace("x", "");

        for c in disam.chars() {
            if let Some(f) = parse_file(c) {
                file_constraint = Some(f);
            } else if let Some(rnk) = parse_rank(c) {
                rank_constraint = Some(rnk);
            }
        }
    }

    // If Pawn, and capture (x), we usually have file constraint (exd5)
    // If not explicit, the first char of original san was file.
    if piece_type == P && san.contains('x') && file_constraint.is_none() {
        if let Some(f) = parse_file(san.chars().next()?) {
            file_constraint = Some(f);
        }
    }

    // Matching
    let mut matched_move = None;

    for i in 0..generator.list.count {
        let m = generator.list.moves[i];

        // 1. Match Target
        if m.target != target_sq {
            continue;
        }

        // 2. Match Piece Type (Moving Piece)
        let moved_piece = get_piece_at(state, m.source)?;
        let moved_type = moved_piece % 6;
        if moved_type != piece_type {
            continue;
        }

        // 3. Match Promotion
        if m.promotion != promotion {
            continue;
        }

        // 4. Match Constraints
        if let Some(f) = file_constraint {
            if (m.source % 8) != f {
                continue;
            }
        }
        if let Some(rnk) = rank_constraint {
            if (m.source / 8) != rnk {
                continue;
            }
        }

        // Found a candidate.
        // In valid SAN, there should be only one.
        // If we find multiple, it's ambiguous (shouldn't happen with valid SAN + full disam).
        // We take the first valid one.

        // Additional Check: Legality? MoveGenerator generates pseudo-legal?
        // Code says "generate_moves". Datagen checks `!is_check` later.
        // We should verify legality to be safe.
        let next = state.make_move(m);
        if crate::search::is_check(&next, state.side_to_move) {
            continue;
        }

        matched_move = Some(m);
        break;
    }

    matched_move
}

fn get_piece_at(state: &GameState, sq: u8) -> Option<usize> {
    for p in 0..12 {
        if state.bitboards[p].get_bit(sq) {
            return Some(p);
        }
    }
    None
}

fn parse_file(c: char) -> Option<u8> {
    if c >= 'a' && c <= 'h' {
        Some(c as u8 - b'a')
    } else {
        None
    }
}

fn parse_rank(c: char) -> Option<u8> {
    if c >= '1' && c <= '8' {
        Some(c as u8 - b'1')
    } else {
        None
    }
}

fn find_castling_move(state: &GameState, kingside: bool) -> Option<Move> {
    // Generate moves and look for King moves > 1 square dist
    let mut generator = MoveGenerator::new();
    generator.generate_moves(state);

    let k_start = if state.side_to_move == crate::state::WHITE {
        4
    } else {
        60
    };

    for i in 0..generator.list.count {
        let m = generator.list.moves[i];
        if m.source == k_start {
            // Is it castling?
            // In this engine, castling is KxR.
            // Check if target is a friendly rook?
            let captured = get_piece_at(state, m.target);
            if let Some(p) = captured {
                // p is friendly rook?
                // White R=3, Black r=9.
                let expected_rook = if state.side_to_move == crate::state::WHITE {
                    R
                } else {
                    r
                };
                if p == expected_rook {
                    // Check side
                    // Kingside: Rook file > King file
                    // Queenside: Rook file < King file
                    // But wait, KxR means m.target is Rook sq.
                    // Kingside Rook usually on H (7). Queenside on A (0).
                    // Or 960 positions.

                    let r_file = m.target % 8;
                    let k_file = m.source % 8;

                    let is_ks = r_file > k_file;

                    if is_ks == kingside {
                        let next = state.make_move(m);
                        if !crate::search::is_check(&next, state.side_to_move) {
                            return Some(m);
                        }
                    }
                }
            }
        }
    }
    None
}
