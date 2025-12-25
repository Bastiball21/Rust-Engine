# Aether
My first chess engine

## Features
*   **Search**: Alpha-Beta with Principal Variation Search, Transposition Table, Iterative Deepening.
*   **Evaluation**: Hybrid 5-Layer NNUE (768 → 256 → 32 → 32 → 32 → 1) with Hand-Crafted Evaluation (HCE) fallback.
*   **Protocol**: Universal Chess Interface (UCI).
*   **Training**: CUDA-accelerated NNUE trainer using `bullet`.

## Building and Testing

### Prerequisites
*   Rust Toolchain (Stable)
*   CUDA Toolkit (Required for Trainer only)

### Running Tests
Run the standard test suite:
```bash
# Set stack size to prevent overflow in deep search tests
RUST_MIN_STACK=8388608 cargo test --release
```

Run Perft tests (Move Generation Verification):
```bash
cargo test --release --test perft_test
```
*Note: `perft` tests cover Start Position, Kiwipete, and Castling/En Passant edge cases.*

## Usage

### UCI Mode (Standard)
Run the engine in UCI mode:
```bash
cargo run --release
```
Then send UCI commands (e.g., `uci`, `isready`, `go depth 10`).

### Options
*   **Hash**: Size of the transposition table in MB (Range: 1-1024, Default: 64).
*   **Threads**: Number of search threads (Range: 1-64, Default: 1).
*   **EvalFile**: Path to the NNUE network file. The engine attempts to load `nn-aether.nnue` from the working directory by default.
*   **UCI_Chess960**: Enable Chess960 (Fischer Random) mode.
*   **SyzygyPath**: Path to Syzygy endgame tablebases.
*   **TTShards**: Number of Transposition Table shards (Range: 1-64, Default: 1).
*   **Move Overhead**: Time buffer in milliseconds to compensate for network/GUI latency (Range: 0-5000, Default: 10).

### Evaluation Fallback
If no NNUE network is loaded, Aether falls back to a Hand-Crafted Evaluation (HCE). This allows the engine to function without a network file, though playing strength will be significantly lower.

## Data Generation

Generate self-play games for training using the `datagen` command. This mode uses Pure Self-Play with random walk openings (Zero Knowledge/No Book).

**New Features:**
*   **Duplicate Game Prevention**: Ensures unique games.
*   **Fixed Depth**: Uses fixed search depth instead of node limits.
*   **Mercy Rule Adjudication**: Early termination for decisive games.

**Command:**
```bash
cargo run --release -- datagen <games> <threads> <depth> <filename> <seed>
```

**Example:**
```bash
# Generate 1000 games on 1 thread, depth 8, saving to data.bin with seed 12345
cargo run --release -- datagen 1000 1 8 data.bin 12345
```

## Training

To train the NNUE network, you need a training dataset (generated via the `datagen` command) and a CUDA-capable GPU. The trainer is configured for the Hybrid 5-Layer Architecture.

**Command:**
```bash
./train.sh [paths...]
```

**Note on CUDA:**
The trainer crate requires `nvcc` and CUDA libraries to build. If CUDA is not available, the trainer build will fail. The core engine (`aether`) does **not** require CUDA and runs on CPU.

**Arguments:**
*   `[paths...]`: Optional. File paths to `.bin` data files or directories containing them. Defaults to `../aether_data.bin`.

**Example:**
```bash
./train.sh my_data.bin
```
Checkpoints are saved to `trainer/checkpoints`.
