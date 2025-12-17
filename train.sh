#!/bin/bash

# Initialize args array
args=()

# Process arguments to handle paths relative to the current directory (root)
# because we will 'cd' into 'trainer' later.
for arg in "$@"; do
    if [ -f "$arg" ] || [ -d "$arg" ]; then
        # If the argument is a file or directory, resolve it to an absolute path
        # realpath should be available on most modern systems
        if command -v realpath >/dev/null 2>&1; then
             args+=("$(realpath "$arg")")
        else
             # Fallback if realpath is missing: assume relative and prepend $PWD
             # Note: This is simple and might not handle edge cases like ../ well without realpath
             # but it's better than nothing.
             if [[ "$arg" = /* ]]; then
                 args+=("$arg")
             else
                 args+=("$(pwd)/$arg")
             fi
        fi
    else
        # Pass non-path arguments as is
        args+=("$arg")
    fi
done

# Navigate to the trainer directory
cd trainer || { echo "Error: Could not change directory to 'trainer'."; exit 1; }

# Run the trainer, forwarding the processed arguments
# Using --release for performance is crucial for training
cargo run --release -- "${args[@]}"
