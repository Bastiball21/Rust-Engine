# Aether
My first chess engine

## Training

To train the NNUE network, you need to have a training dataset ready.

1.  **Prepare Data**: Place your generated training data file named `aether_data.bin` in the root directory of this repository.
    *   To generate data: `cargo run --release -- datagen <games> <threads> <depth> <filename>`
2.  **Run Training**: You can start the training process using the provided script:

    ```bash
    ./train.sh
    ```

    Alternatively, you can run the trainer manually:

    ```bash
    cd trainer
    cargo run --release
    ```

    *Note: The trainer requires a CUDA-capable GPU and the CUDA Toolkit installed.*

3.  **Checkpoints**: The training process will output network checkpoints to the `trainer/checkpoints` directory.
    *   The `trainer` crate uses `bullet_lib`. Ensure your environment is set up for it (CUDA).

## Playing
This engine supports the UCI protocol.

### Options
*   **Hash**: Size of the transposition table in MB (Default: 64)
*   **Threads**: Number of search threads (Default: 1)
*   **EvalFile**: Path to the NNUE network file (Default: empty). The engine will look for `nn-aether.nnue` in the working directory on startup if not specified.
