# Aether
My first chess engine

## Data Generation

Generate self-play games for training using the `datagen` command.

**Command:**
```bash
cargo run --release -- datagen <games> <threads> <depth> <filename> [book_path] [book_ply]
```

**Arguments:**
*   `<games>`: Number of games to generate.
*   `<threads>`: Number of threads to use.
*   `<depth>`: Search depth per move.
*   `<filename>`: Output file for the training data (e.g., `aether_data.bin`).
*   `[book_path]`: (Optional) Path to an opening book (`.epd` or `.pgn`). Use `none` or `-` to skip.
*   `[book_ply]`: (Optional) Max ply (half-moves) to use from the book (e.g., 16 ply = 8 full moves).

**Example:**
```bash
# Generate 1000 games on 1 thread, depth 6, saving to data.bin, using an opening book up to 8 moves
cargo run --release -- datagen 1000 1 6 data.bin /path/to/book.epd 16
```

## Training

To train the NNUE network, you need a training dataset (generated via the `datagen` command).

**Command:**
```bash
./train.sh [paths...]
```

**Arguments:**
*   `[paths...]`: Optional. File paths to `.bin` data files or directories containing them. If omitted, defaults to `../aether_data.bin` (relative to the `trainer` directory).

**Examples:**
```bash
# Default behavior (uses aether_data.bin in root)
./train.sh

# Specific file
./train.sh my_data.bin

# Multiple files
./train.sh part1.bin part2.bin

# Directory (scans for .bin files)
./train.sh data_folder/
```

*Note: The trainer requires a CUDA-capable GPU and the CUDA Toolkit installed. Checkpoints are saved to `trainer/checkpoints`.*

## Playing
This engine supports the UCI protocol.

### Options
*   **Hash**: Size of the transposition table in MB (Default: 64)
*   **Threads**: Number of search threads (Default: 1)
*   **EvalFile**: Path to the NNUE network file (Default: empty). The engine will look for `nn-aether.nnue` in the working directory on startup if not specified.
