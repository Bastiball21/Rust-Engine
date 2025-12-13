# Aether
My first chess engine

## Data Generation
To generate training data for the NNUE, use the `datagen` command. This will produce a binary file `aether_data.bin` compatible with the `bullet` trainer.

```bash
cargo run --release -- datagen <number_of_games>
```

Example:
```bash
cargo run --release -- datagen 1000
```
