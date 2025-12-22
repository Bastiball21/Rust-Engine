#!/bin/bash
set -e

# Profiling script for Aether Engine

BINARY="./target/release/aether"
OUTPUT_SVG="profile.svg"

# Check for perf
if command -v perf >/dev/null 2>&1; then
    echo "Found perf. Running detailed profiling..."

    # Run the engine with perf
    # We pipe commands to the engine and record
    # Note: Requires permissions in sandbox. If this fails, we fall back.

    # Benchmark Command
    CMD="position startpos\ngo depth 16\nquit"

    echo -e "$CMD" | perf record -F 999 --call-graph dwarf -g -o perf.data $BINARY

    # Generate Flamegraph
    if command -v inferno-flamegraph >/dev/null 2>&1; then
        perf script | inferno-flamegraph > $OUTPUT_SVG
        echo "Generated $OUTPUT_SVG using inferno-flamegraph"
    elif [ -f "FlameGraph/flamegraph.pl" ]; then
        perf script | FlameGraph/stackcollapse-perf.pl | FlameGraph/flamegraph.pl > $OUTPUT_SVG
        echo "Generated $OUTPUT_SVG using FlameGraph scripts"
    else
        echo "FlameGraph tools not found. Dumping top 20 functions:"
        perf report --sort comm,dso,symbol --stdio | head -n 40
    fi

else
    echo "perf not found or permission denied."
    echo "Running fallback timing benchmark..."

    # Simple PIDStat or Time check
    echo "Running: go depth 16"

    start_time=$(date +%s%N)
    echo -e "position startpos\ngo depth 16\nquit" | $BINARY > bench.log
    end_time=$(date +%s%N)

    duration=$((end_time - start_time))
    echo "Duration: $((duration / 1000000)) ms"

    # Parse NPS
    grep "nps" bench.log | tail -n 1

    # Manual Hotspot Guessing (grep based on source)
    echo "Note: Without perf, check 'src/movegen.rs' and 'src/eval.rs' manually."
fi
