# Aether
My first chess engine

## Features
*   **Search**: Alpha-Beta with Principal Variation Search, Transposition Table, Iterative Deepening.
*   **Evaluation**: NNUE (Efficiently Updatable Neural Network) with Hand-Crafted Evaluation (HCE) fallback.
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

### Evaluation Fallback
If no NNUE network is loaded, Aether falls back to a Hand-Crafted Evaluation (HCE). This allows the engine to function without a network file, though playing strength will be significantly lower.

## Data Generation

Generate self-play games for training using the `datagen` command.

**Command:**
```bash
cargo run --release -- datagen <games> <threads> <depth> <filename> [book_path] [book_ply]
```

**Example:**
```bash
# Generate 1000 games on 1 thread, depth 6, saving to data.bin
cargo run --release -- datagen 1000 1 6 data.bin /path/to/book.epd 16
```

## Training

To train the NNUE network, you need a training dataset (generated via the `datagen` command) and a CUDA-capable GPU.

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
