// src/tests_chess960.rs
use crate::state::{GameState, Move, WHITE, BLACK, R, r, K, k};
use crate::movegen::MoveGenerator;
use crate::uci::UCI_CHESS960;
use std::sync::atomic::Ordering;

#[test]
fn test_standard_startpos_regression() {
    crate::zobrist::init_zobrist();
    crate::bitboard::init_magic_tables();
    crate::movegen::init_move_tables();

    let state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    assert_eq!(state.castling_rook_files[WHITE][0], 7);
    assert_eq!(state.castling_rook_files[WHITE][1], 0);
    assert_eq!(state.castling_rook_files[BLACK][0], 7);
    assert_eq!(state.castling_rook_files[BLACK][1], 0);

    let mut mg = MoveGenerator::new();
    mg.generate_moves(&state);
    assert_eq!(mg.list.count, 20);
}

#[test]
fn test_chess960_fen_parsing() {
    crate::zobrist::init_zobrist();
    crate::bitboard::init_magic_tables();

    // FEN: rkr5/8/8/8/8/8/8/RKR5 w AC - 0 1
    // Rooks at A1(0), C1(2). King at B1(1).
    let state = GameState::parse_fen("rkr5/8/8/8/8/8/8/RKR5 w AC - 0 1");

    assert_eq!(state.castling_rook_files[WHITE][1], 0); // A -> Queenside
    assert_eq!(state.castling_rook_files[WHITE][0], 2); // C -> Kingside
    assert_eq!((state.castling_rights & 1) != 0, true);
    assert_eq!((state.castling_rights & 2) != 0, true);

    // Expect CQ because C(2) is Kingside (K), A(0) is Queenside (Q).
    // And standard FEN output usually uses Q for A-file.
    let fen_out = state.to_fen();
    assert!(fen_out.contains("CQ") || fen_out.contains("AC") || fen_out.contains("QC"));
}

#[test]
fn test_castling_move_generation_standard() {
    crate::zobrist::init_zobrist();
    crate::bitboard::init_magic_tables();
    crate::movegen::init_move_tables();

    let state = GameState::parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let mut mg = MoveGenerator::new();
    mg.generate_moves(&state);

    let mut has_h1 = false;
    let mut has_a1 = false;

    for i in 0..mg.list.count {
        let mv = mg.list.moves[i];
        if mv.source == 4 {
            if mv.target == 7 { has_h1 = true; }
            if mv.target == 0 { has_a1 = true; }
        }
    }

    assert!(has_h1, "White Kingside Castle (e1h1) missing");
    assert!(has_a1, "White Queenside Castle (e1a1) missing");
}

#[test]
fn test_castling_make_move() {
    crate::zobrist::init_zobrist();
    crate::bitboard::init_magic_tables();

    let state = GameState::parse_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let mv = Move { source: 4, target: 7, promotion: None, is_capture: false };
    let next_state = state.make_move(mv);

    assert!(next_state.bitboards[K].get_bit(6));
    assert!(!next_state.bitboards[K].get_bit(4));
    assert!(next_state.bitboards[R].get_bit(5));
    assert!(!next_state.bitboards[R].get_bit(7));
    assert_eq!(next_state.castling_rights & 3, 0);
}

#[test]
fn test_chess960_castling_make_move() {
    crate::zobrist::init_zobrist();
    crate::bitboard::init_magic_tables();
    crate::movegen::init_move_tables();

    // FEN: 8/8/8/8/8/8/8/1R1K2R1 w BG - 0 1
    // King at D1 (3). Rooks at B1 (1) and G1 (6).
    // Castling rights: BG.
    // Removed black pieces to avoid check.
    let state = GameState::parse_fen("8/8/8/8/8/8/8/1R1K2R1 w BG - 0 1");

    println!("DEBUG: Castling Rights: {}", state.castling_rights);
    println!("DEBUG: Rook Files: K={}, Q={}", state.castling_rook_files[WHITE][0], state.castling_rook_files[WHITE][1]);

    let mut mg = MoveGenerator::new();
    mg.generate_moves(&state);
    let mut found = false;
    for i in 0..mg.list.count {
        let mv = mg.list.moves[i];
        if mv.source == 3 && mv.target == 6 {
            found = true;
        }
    }
    assert!(found, "960 Castle move d1g1 not generated");

    let mv = Move { source: 3, target: 6, promotion: None, is_capture: false };
    let next_state = state.make_move(mv);

    assert!(next_state.bitboards[K].get_bit(6));
    assert!(next_state.bitboards[R].get_bit(5));
}
