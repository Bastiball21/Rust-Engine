
use aether::state::{GameState, Move, NO_PIECE};
use aether::movegen::{self, MoveGenerator, is_move_pseudo_legal};
use aether::zobrist;
use aether::bitboard;
use std::sync::Once;

static INIT: Once = Once::new();

fn init() {
    INIT.call_once(|| {
        zobrist::init_zobrist();
        bitboard::init_magic_tables();
        movegen::init_move_tables();
    });
}

#[test]
fn test_pseudo_legal_all_legal_moves() {
    init();
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // Start
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", // Kiwipete
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", // Endgame
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", // Complex
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", // Promotion
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", // Quiet
    ];

    for fen in fens {
        let state = GameState::parse_fen(fen);
        let mut mg = MoveGenerator::new();
        mg.generate_moves(&state);

        for i in 0..mg.list.count {
            let mv = mg.list.moves[i];
            // Assert all generated moves are pseudo-legal
            if !is_move_pseudo_legal(&state, mv) {
                 panic!("Generated move considered illegal: {:?} for FEN: {}", mv, fen);
            }
        }
    }
}

#[test]
fn test_pseudo_legal_random_garbage() {
    init();
    // Start position
    let state = GameState::new();

    let mut pseudo_legal_count = 0;
    let trials = 10000;

    // Generate random moves
    // Most should be illegal
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..trials {
        let from = rng.gen_range(0..64);
        let to = rng.gen_range(0..64);
        if from == to { continue; }

        let is_capture = rng.gen_bool(0.2); // 20% capture flag
        let promotion = if rng.gen_bool(0.05) { Some(4) } else { None }; // Occasional promotion

        let mv = Move::new(from, to, promotion, is_capture);

        if is_move_pseudo_legal(&state, mv) {
            // If it passes, verify it's actually in the move list
            let mut mg = MoveGenerator::new();
            mg.generate_moves(&state);
            let mut found = false;
            for i in 0..mg.list.count {
                let real_mv = mg.list.moves[i];
                // Note: Random move might miss correct promotion/capture flags that generator has
                // So we check basic from/to match and type plausibility
                if real_mv.source() == from && real_mv.target() == to {
                    // Rough check: If flags match enough
                    if real_mv.is_capture() == is_capture && real_mv.promotion() == promotion {
                        found = true;
                        break;
                    }
                }
            }

            if !found {
                // It passed pseudo-legal check but wasn't generated.
                // This is acceptable IF it's "pseudo-legal" but not fully legal (e.g. pinned),
                // OR if our flags were wrong in a way that is_pseudo_legal accepts but generator didn't produce.
                // However, is_pseudo_legal checks geometric validity.
                // Let's verify specifically "Ghost Piece" or "Teleport" cases aren't happening.

                let piece = state.board[from as usize] as usize;
                if piece == NO_PIECE {
                    panic!("is_pseudo_legal accepted move from empty square: {:?}", mv);
                }
            }
            pseudo_legal_count += 1;
        }
    }

    // Sanity check: Should be very low count for random moves on startpos
    println!("Random pseudo-legal moves found: {}/{}", pseudo_legal_count, trials);
    assert!(pseudo_legal_count < 100, "Too many random moves passed pseudo-legal check!");
}

#[test]
fn test_knight_moving_like_rook_fails() {
    init();
    let state = GameState::new(); // Start pos
    // White Knight on b1 (1). Rook-like move to b3 (17) (Occupied by pawn? No, b3 is empty)
    // b1 is N. b3 is empty.
    // N can move to a3, c3. NOT b3.

    let mv = Move::new(1, 17, None, false); // b1 -> b3
    assert!(!is_move_pseudo_legal(&state, mv), "Knight moving like rook should fail");
}

#[test]
fn test_bishop_jumping_over_pieces_fails() {
    init();
    let state = GameState::new(); // Start pos
    // White Bishop on c1 (2). Blocked by pawn on d2 (11).
    // Try moving to h6 (47).
    let mv = Move::new(2, 47, None, false);
    assert!(!is_move_pseudo_legal(&state, mv), "Bishop jumping over pawn should fail");
}

#[test]
fn test_capture_on_empty_square_fails() {
    init();
    let state = GameState::new();
    // e2 -> e4 capture (e4 is empty)
    let mv = Move::new(12, 28, None, true); // e2(12)->e4(28) with capture flag
    assert!(!is_move_pseudo_legal(&state, mv), "Capture on empty square should fail");
}

#[test]
fn test_castling_pseudo_legality() {
     init();
     // FEN with castling rights but path blocked
     let fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
     let state = GameState::parse_fen(fen);

     // White King e1(4) -> h1(7) (King-side, Castling on Rook)
     let mv = Move::new(4, 7, None, false); // King takes Rook representation
     // Path blocked? No, empty board in middle.
     // Rights? Yes.
     assert!(is_move_pseudo_legal(&state, mv), "Castling should be pseudo-legal here");

     // Blocked path
     let fen_blocked = "r3k2r/8/8/8/8/8/8/R3KB1R w KQkq - 0 1"; // f1 blocked by Bishop
     let state_blocked = GameState::parse_fen(fen_blocked);
     let mv_blocked = Move::new(4, 7, None, false);
     assert!(!is_move_pseudo_legal(&state_blocked, mv_blocked), "Blocked castling should fail");
}
