# Aether
My first chess engine

## Training

To train the NNUE network, you need to have a training dataset ready.

1.  **Prepare Data**: Place your generated training data file named `aether_data.bin` in the root directory of this repository.
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
