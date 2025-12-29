use bullet_lib::{
    game::inputs::ChessBucketsMirrored,
    nn::{
        optimiser::AdamW,
        NetworkBuilder, NetworkBuilderNode, BackendMarker,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};
use std::path::Path;
use std::io::Write;

fn main() {
    println!("Aether NNUE Trainer (Funnel Architecture)");

    // Basic runtime check for CUDA availability
    let has_cuda = std::process::Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !has_cuda {
        eprintln!("ERROR: CUDA is not detected on this system (nvidia-smi failed).");
        std::process::exit(1);
    }
    println!("CUDA detected. Initializing training...");

    // Hyperparameters
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);
    let superbatches = 40;

    // Architecture: 768 -> 512 -> 64 -> 32 -> 16 -> 1
    // L1: Accumulator (512)
    // L2: Dense (64)
    // L3: Dense (32)
    // L4: Dense (16)
    // OUT: Dense (1)
    let l1_size = 512;
    let l2_size = 64;
    let l3_size = 32;
    let l4_size = 16;

    // Parse command line arguments for dataset paths
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dataset_paths = Vec::new();

    if args.is_empty() {
        dataset_paths.push("../aether_data.bin".to_string());
    } else {
        for arg in args {
            let path = Path::new(&arg);
            if path.is_dir() {
                if let Ok(entries) = std::fs::read_dir(path) {
                    for entry in entries {
                        if let Ok(entry) = entry {
                            let path = entry.path();
                            if path.is_file() && path.extension().map_or(false, |ext| ext == "bin") {
                                if let Some(path_str) = path.to_str() {
                                    dataset_paths.push(path_str.to_string());
                                }
                            }
                        }
                    }
                }
            } else {
                dataset_paths.push(arg);
            }
        }
    }

    if dataset_paths.is_empty() {
        eprintln!("Error: No dataset files found.");
        std::process::exit(1);
    }

    println!("Using datasets:");
    for path in &dataset_paths {
        println!("  {}", path);
    }

    // Pure WDL
    let wdl_proportion = 1.0;

    // 32 Buckets (Standard Mirrored)
    let buckets = ChessBucketsMirrored::new(std::array::from_fn(|i| i));

    let mut trainer = ValueTrainerBuilder::default()
        .use_devices(vec![0])
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(buckets)
        .save_format(&[
            // Layer 0 (Accumulator): 768 -> 512 (SCReLU)
            SavedFormat::id("l0w").round().quantise::<i16>(255), // Weights
            SavedFormat::id("l0b").round().quantise::<i16>(255), // Bias

            // Layer 1: 512 -> 64 (ClippedReLU)
            // Input size 512 (STM only)
            SavedFormat::id("l1w").round().quantise::<i16>(64),
            SavedFormat::id("l1b").round().quantise::<i16>(64 * 127),

            // Layer 2: 64 -> 32 (ClippedReLU)
            SavedFormat::id("l2w").round().quantise::<i16>(64),
            SavedFormat::id("l2b").round().quantise::<i16>(64 * 127),

            // Layer 3: 32 -> 16 (ClippedReLU)
            SavedFormat::id("l3w").round().quantise::<i16>(64),
            SavedFormat::id("l3b").round().quantise::<i16>(64 * 127),

            // Layer 4: 16 -> 1 (Raw)
            SavedFormat::id("l4w").round().quantise::<i16>(64),
            SavedFormat::id("l4b").round().quantise::<i32>(255 * 64 * 127), // Output bias
        ])
        .loss_fn(|output: NetworkBuilderNode<BackendMarker>, target: NetworkBuilderNode<BackendMarker>| {
            output.sigmoid().squared_error(target)
        })
        .build(|builder: &NetworkBuilder<BackendMarker>, stm_inputs: NetworkBuilderNode<BackendMarker>, ntm_inputs: NetworkBuilderNode<BackendMarker>| {
            // Layer 0: 768 -> 512
            let l0 = builder.new_affine("l0", stm_inputs.annotated_node().shape.size(), l1_size);

            // Layer 1: 512 -> 64
            // Input is 512 (STM only).
            let l1 = builder.new_affine("l1", l1_size, l2_size);

            // Layer 2: 64 -> 32
            let l2 = builder.new_affine("l2", l2_size, l3_size);

            // Layer 3: 32 -> 16
            let l3 = builder.new_affine("l3", l3_size, l4_size);

            // Layer 4: 16 -> 1
            let l4 = builder.new_affine("l4", l4_size, 1);

            // Forward Pass
            // L0 Activation: SCReLU (SquarerClippedReLU)
            let stm0 = l0.forward(stm_inputs).screlu();
            let ntm0 = l0.forward(ntm_inputs).screlu();

            let combined = stm0 - ntm0;

            // Single Perspective: Use STM features directly for the rest of the network
            // Note: NTM features are unused in the forward pass, but 'l0' weights are shared/updated via STM usage.
            // effectively this learns to evaluate purely based on "My King's perspective" features.

            // L1: 512 -> 64 (ClippedReLU)
            let hidden_layer_1 = l1.forward(combined).crelu();

            // L2: 64 -> 32 (ClippedReLU)
            let hidden_layer_2 = l2.forward(hidden_layer_1).crelu();

            // L3: 32 -> 16 (ClippedReLU)
            let hidden_layer_3 = l3.forward(hidden_layer_2).crelu();

            // L4: 16 -> 1 (Linear)
            l4.forward(hidden_layer_3)
        });

    let net_id = "aether-funnel";
    let schedule = TrainingSchedule {
        net_id: net_id.to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 8728,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL {
            value: wdl_proportion,
        },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr,
            final_lr,
            final_superbatch: superbatches,
        },
        save_rate: 1,
    };

    let output_directory = "checkpoints";
    let settings = LocalSettings {
        threads: 2,
        test_set: None,
        output_directory,
        batch_queue_size: 32,
    };

    // Convert paths to string slices for the loader
    let path_slices: Vec<&str> = dataset_paths.iter().map(|s| s.as_str()).collect();
    let dataloader = DirectSequentialDataLoader::new(&path_slices);

    trainer.run(&schedule, &settings, &dataloader);

    // --- Post-Training Finalization (Magic Number Prepend) ---
    println!("Training complete. Finalizing network...");

    // Construct expected last checkpoint path
    // Bullet format: {output_directory}/{net_id}-{superbatch}.bin (usually)
    // But SavedFormat might save as .bin
    // Let's look for the file corresponding to the last superbatch.
    let checkpoint_path = format!("{}/{}-{}.bin", output_directory, net_id, superbatches);
    let final_path = "nn-aether.nnue";

    if let Ok(mut content) = std::fs::read(&checkpoint_path) {
        let magic: u32 = 0xAE74E201;
        let mut final_content = Vec::with_capacity(4 + content.len());
        final_content.extend_from_slice(&magic.to_le_bytes());
        final_content.append(&mut content);

        if let Ok(_) = std::fs::write(final_path, &final_content) {
            println!("Successfully saved finalized network to: {}", final_path);
        } else {
            eprintln!("Failed to write finalized network to: {}", final_path);
        }
    } else {
        println!("Could not find final checkpoint at: {}. Please manually prepend magic 0xAE74E201.", checkpoint_path);
    }
}
