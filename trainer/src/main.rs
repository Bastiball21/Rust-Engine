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

fn main() {
    println!("Aether NNUE Trainer (Zero Architecture)");

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

    // Architecture
    let l1_size = 256;
    let l2_size = 32;
    let l3_size = 32;
    let l4_size = 32;

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
            // Layer 0 (Accumulator): 768 -> 256 (SCReLU)
            SavedFormat::id("l0w").round().quantise::<i16>(255), // Weights
            SavedFormat::id("l0b").round().quantise::<i16>(255), // Bias

            // Layer 1: 512 -> 32 (ClippedReLU)
            // Note: 512 because we concat both sides (256 + 256)
            SavedFormat::id("l1w").round().quantise::<i16>(64),
            SavedFormat::id("l1b").round().quantise::<i16>(64 * 127), // Bias scales with activation limit

            // Layer 2: 32 -> 32 (ClippedReLU)
            SavedFormat::id("l2w").round().quantise::<i16>(64),
            SavedFormat::id("l2b").round().quantise::<i16>(64 * 127),

            // Layer 3: 32 -> 32 (ClippedReLU)
            SavedFormat::id("l3w").round().quantise::<i16>(64),
            SavedFormat::id("l3b").round().quantise::<i16>(64 * 127),

            // Layer 4: 32 -> 1 (Raw)
            SavedFormat::id("l4w").round().quantise::<i16>(64),
            SavedFormat::id("l4b").round().quantise::<i16>(255 * 64 * 127), // Output bias
        ])
        .loss_fn(|output: NetworkBuilderNode<BackendMarker>, target: NetworkBuilderNode<BackendMarker>| {
            output.sigmoid().squared_error(target)
        })
        .build(|builder: &NetworkBuilder<BackendMarker>, stm_inputs: NetworkBuilderNode<BackendMarker>, ntm_inputs: NetworkBuilderNode<BackendMarker>| {
            // Layer 0: 768 -> 256
            let l0 = builder.new_affine("l0", stm_inputs.annotated_node().shape.size(), l1_size);

            // Layer 1: 512 (256*2) -> 32
            let l1 = builder.new_affine("l1", 2 * l1_size, l2_size);

            // Layer 2: 32 -> 32
            let l2 = builder.new_affine("l2", l2_size, l3_size);

            // Layer 3: 32 -> 32
            let l3 = builder.new_affine("l3", l3_size, l4_size);

            // Layer 4: 32 -> 1
            let l4 = builder.new_affine("l4", l4_size, 1);

            // Forward Pass
            // L0 Activation: SCReLU (SquarerClippedReLU) -> Range [0, 255] roughly (if quantised) or [0, 1] in float world squared
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();

            // Concat perspectives: 256 + 256 = 512
            let hidden_layer_0 = stm_hidden.concat(ntm_hidden);

            // L1: 512 -> 32 (ClippedReLU)
            let hidden_layer_1 = l1.forward(hidden_layer_0).clamp_zero().clamp_max(1.0); // ClippedReLU [0, 1]

            // L2: 32 -> 32 (ClippedReLU)
            let hidden_layer_2 = l2.forward(hidden_layer_1).clamp_zero().clamp_max(1.0);

            // L3: 32 -> 32 (ClippedReLU)
            let hidden_layer_3 = l3.forward(hidden_layer_2).clamp_zero().clamp_max(1.0);

            // L4: 32 -> 1 (Linear)
            l4.forward(hidden_layer_3)
        });

    let schedule = TrainingSchedule {
        net_id: "aether-zero".to_string(),
        eval_scale: 400.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 6104,
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
        save_rate: 10,
    };

    let settings = LocalSettings {
        threads: 2,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 32,
    };

    // Convert paths to string slices for the loader
    let path_slices: Vec<&str> = dataset_paths.iter().map(|s| s.as_str()).collect();
    let dataloader = DirectSequentialDataLoader::new(&path_slices);

    trainer.run(&schedule, &settings, &dataloader);
}
