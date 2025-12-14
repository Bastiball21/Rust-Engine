#!/bin/bash

# Define the expected data file name
DATA_FILE="aether_data.bin"

# Check if the data file exists in the current directory (repo root)
if [ ! -f "$DATA_FILE" ]; then
    echo "Error: '$DATA_FILE' not found in the root directory."
    echo "Please ensure you have generated or placed the training data file named '$DATA_FILE' in the root of the repository."
    echo "Expected path: $(pwd)/$DATA_FILE"
    exit 1
fi

echo "Found '$DATA_FILE'. Starting training..."

# Navigate to the trainer directory
cd trainer || { echo "Error: Could not change directory to 'trainer'."; exit 1; }

# Run the trainer
# Using --release for performance is crucial for training
cargo run --release
