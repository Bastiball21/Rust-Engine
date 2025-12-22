use criterion::{black_box, criterion_group, criterion_main, Criterion};
use aether::state::{GameState, Move};
use aether::movegen::MoveGenerator;
use aether::tt::TranspositionTable;
use aether::{zobrist, bitboard, movegen, eval, threat};

fn init_globals() {
    zobrist::init_zobrist();
    bitboard::init_magic_tables();
    movegen::init_move_tables();
    eval::init_eval();
    threat::init_threat();
}

fn bench_make_move(c: &mut Criterion) {
    init_globals();
    let mut group = c.benchmark_group("make_move");
    let mut state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let mut gen = MoveGenerator::new();
    gen.generate_moves(&state);
    let mv = gen.list.moves[0]; // e2e4 usually

    group.bench_function("make_unmake_startpos", |b| b.iter(|| {
        let unmake = state.make_move_inplace(black_box(mv), &mut None);
        state.unmake_move(black_box(mv), unmake, &mut None);
    }));
    group.finish();
}

fn bench_tt(c: &mut Criterion) {
    init_globals();
    let mut group = c.benchmark_group("tt");
    let tt = TranspositionTable::new_default(16); // Default 1 shard
    let state = GameState::parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    group.bench_function("tt_probe_empty", |b| b.iter(|| {
        tt.probe_data(black_box(state.hash), &state, None);
    }));

    tt.store(state.hash, 100, None, 5, 1, None);
    group.bench_function("tt_probe_hit", |b| b.iter(|| {
        tt.probe_data(black_box(state.hash), &state, None);
    }));
    group.finish();
}

criterion_group!(benches, bench_make_move, bench_tt);
criterion_main!(benches);
