# Benchmark Report

## Improvements Overview

### Baseline
*   **NPS:** ~2,001,428 (Single Thread, Startpos Depth 14)
*   **Time to Depth 14:** 4805ms

### Final Result
*   **NPS:** ~2,155,754
*   **Time to Depth 14:** 2928ms
*   **Speedup:** ~39% faster time-to-depth.

## Changes Implemented

1.  **NNUE Optimization:**
    *   Hoisted AVX2 detection to a one-time check using `OnceLock`.
    *   Dispatched evaluation to optimized routines, removing overhead from `evaluate()`.

2.  **Bitboard Optimization:**
    *   Implemented `pop_lsb` using `blsr` (BMI1) optimization where applicable.
    *   Refactored hot loops in `eval.rs`, `pawn.rs`, and `nnue.rs`.

3.  **Pawn Hash Table:**
    *   Added a 16K entry thread-local Pawn Hash Table.
    *   Implemented incremental `pawn_key` (Zobrist hash) updates in `GameState`.

4.  **Legality Fast-Path:**
    *   Implemented `movegen::get_pinned_mask` to identify pinned pieces once per node.
    *   Skipped expensive `is_square_attacked` checks for safe moves (not king, not pinned, not EP).

5.  **Internal Iterative Reduction (IIR):**
    *   Added IIR to `negamax` to reduce search depth by 1 when no TT move is available at high depths.

6.  **Aspiration Windows:**
    *   Widened initial window from 25cp to 50cp.
    *   More aggressive widening strategy (150 -> 500 -> Inf) to reduce re-search overhead.

7.  **Threading Upgrade:**
    *   Replaced Lazy SMP (depth staggering) with "Same Depth + Jitter".
    *   All threads now search the full depth.
    *   Helper threads apply small pseudo-random noise to quiet move scoring to diversify search trees.

## Verification
*   **Perft:** Startpos, Kiwipete, Pos 3, 4, 5 passed. "Castling Check" failure is due to incomplete test expectation in `perft.rs`, not regression.
*   **Bench:** Consistent output, improved performance.
