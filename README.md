# Aether

A Rust chess engine (UCI) with a Bullet-format NNUE pipeline + CUDA trainer.

## What’s in here

- **Engine**: alpha-beta search with TT + iterative deepening (UCI).
- **Eval**: NNUE (Bullet-format compatible). Falls back to HCE if no network is loaded.
- **Data**:
  - `datagen` mode: self-play / search-based position generation (writes `.bin` in BulletFormat).
  - `convert` mode: **PGN → BulletFormat** conversion for training.
- **Trainer**: `trainer/` uses `bullet_lib` (CUDA) to train and exports `nn-aether.nnue`.

> ⚠️ Windows note: if `git clone` fails with `error: invalid path '0.'`,
> the repo contains a file literally named `0.` which Windows dislikes. Delete/rename it in the repo (recommended),
> or clone inside **WSL**.

---

## Build (engine)

Fast local build:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Run as a UCI engine:
```bash
cargo run --release
```

Load a network (UCI option):
- `EvalFile` → path to your `nn-aether.nnue`

---

## PGN → training data (BulletFormat)

This converts games from a PGN into a Bullet-format binary dataset the trainer can read.

```bash
cargo run --release -- convert <input.pgn> <output.bin>
```

Example:
```bash
cargo run --release -- convert ./test_compact.pgn ./aether_data.bin
```

### What the converter writes

For each legal position in the PGN (until parse fails / game ends), it stores:
- **Score**: a static evaluation (centipawn-ish i16), converted to **White-relative**.
- **Result**: game result mapped to a float from White’s perspective:
  - `1-0` → `1.0`
  - `0-1` → `0.0`
  - `1/2-1/2` → `0.5`

This “perspective correction” is important: the network learns **“is this good for the side to move?”**
instead of accidentally learning “Black losing positions are good because White eventually won” 💀.

---

## Train (CUDA trainer)

The trainer lives in `trainer/` and reads Bullet-format `.bin` files.

### 1) Build/run the trainer
```bash
cd trainer
cargo run --release -- <path_to_dataset_or_folder>
```

- If you pass a **folder**, it will scan for `.bin` files inside.
- If you pass nothing, it defaults to `../aether_data.bin`.

Example:
```bash
cd trainer
cargo run --release -- ../aether_data.bin
```

### 2) Output network
At the end, the trainer writes:
- `nn-aether.nnue` (in `trainer/`)

Copy it to the engine folder (or point `EvalFile` to it):
```bash
cp trainer/nn-aether.nnue .
```

### 3) Keep engine + trainer architecture in sync

There is an optional feature flag for a bigger net:
- `nnue_512_64`

If you enable it, **enable it on both** the engine and the trainer, or the engine will reject the file
(different magic numbers are used for strict shape validation).

Engine:
```bash
cargo build --release --features nnue_512_64
```

Trainer:
```bash
cd trainer
cargo run --release --features nnue_512_64 -- ../aether_data.bin
```

---

## Tests

```bash
RUST_MIN_STACK=8388608 cargo test --release
cargo test --release --test perft_test
```

---

## License / credits

- Training/data format and trainer stack powered by `bulletformat` / `bullet_lib`.
