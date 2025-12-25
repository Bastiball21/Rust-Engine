#!/bin/bash
set -e

# Configuration
GAMES_PER_ITER=5000000 # 5 Million
THREADS=12
DEPTH=8
DATAGEN_FILE="aether_data.bin"
NETWORK_FILE="nn-aether.nnue"
CHECKPOINT_DIR="trainer/checkpoints"
GEN_COUNTER_FILE=".gen_counter"
SEED_BASE=1000

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[Aether Loop]${NC} $1"
}

# 0. Compile Everything
log "Compiling Engine and Trainer..."
cargo build --release --bin aether
cd trainer
cargo build --release
cd ..

# 1. Initialize Loop
if [ ! -f $GEN_COUNTER_FILE ]; then
    echo "0" > $GEN_COUNTER_FILE
fi

GEN=$(cat $GEN_COUNTER_FILE)
log "Starting from Generation $GEN"

while true; do
    log "${CYAN}--- Generation $GEN ---${NC}"

    # 2. Generate Data
    # If Gen 0, we use HCE (no network loaded, or empty file).
    # If Gen > 0, we ensure the best network is available as 'nn-aether.nnue'

    DATA_FILE="data_gen${GEN}.bin"
    SEED=$((SEED_BASE + GEN))

    log "Generating $GAMES_PER_ITER games into $DATA_FILE using seed $SEED..."

    # Run Datagen
    # ./target/release/aether datagen <games> <threads> <depth> <filename> <seed>
    # Note: Depth arg is now ignored/fixed in code (FixedDepth 8), but we pass a placeholder.
    ./target/release/aether datagen $GAMES_PER_ITER $THREADS 8 $DATA_FILE $SEED

    log "Generation Complete."

    # 3. Train
    # We train on ALL previous data or window?
    # Strategy: Train on Current Gen + Last Gen? Or All?
    # For now: Train on the file just generated (or accumulate).
    # Simpler: Pass the new file to the trainer. The trainer logic in main.rs accepts a list of files.
    # We can pass all data_gen*.bin files.

    log "Training Network..."

    # Collect all data files
    DATA_FILES=$(ls data_gen*.bin)

    # Run Trainer
    # Trainer outputs checkpoints to trainer/checkpoints
    # It auto-saves. We need to pick the best one (or last one) after it finishes.
    cd trainer
    cargo run --release -- $DATA_FILES
    cd ..

    # 4. Update Network
    # Find the latest checkpoint
    # Checkpoints are named: net-epoch-X.nnue or similar?
    # Bullet usually saves `aether-zero-epoch-X.nnue` or `step-X`
    # We need to identify the file.

    LATEST_NET=$(ls -t trainer/checkpoints/*.nnue | head -n 1)

    if [ -z "$LATEST_NET" ]; then
        log "ERROR: No network file found after training!"
        exit 1
    fi

    log "New best network: $LATEST_NET"
    cp "$LATEST_NET" "$NETWORK_FILE"

    log "Updated $NETWORK_FILE for next generation."

    # 5. Increment Generation
    GEN=$((GEN + 1))
    echo "$GEN" > $GEN_COUNTER_FILE

    log "Waiting 5 seconds before next loop..."
    sleep 5
done
