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

fn main() {
    // hyperparams to fiddle with
    let hl_size = 256;
    let dataset_path = "../aether_data.bin";
    let initial_lr = 0.001;
    let final_lr = 0.001 * 0.3f32.powi(5);
    let superbatches = 40;

    // Pure WDL
    let wdl_proportion = 1.0;

    // 32 Buckets (Standard Mirrored)
    // Maps each of the 32 squares (files 0-3) to a unique bucket.
    // Files 4-7 are mirrored automatically by ChessBucketsMirrored.
    let buckets = ChessBucketsMirrored::new(std::array::from_fn(|i| i));

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(buckets)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(255),
            SavedFormat::id("l0b").round().quantise::<i16>(255),
            SavedFormat::id("l1w").round().quantise::<i16>(64),
            SavedFormat::id("l1b").round().quantise::<i16>(255 * 64),
        ])
        .loss_fn(|output: NetworkBuilderNode<BackendMarker>, target: NetworkBuilderNode<BackendMarker>| {
            output.sigmoid().squared_error(target)
        })
        .build(|builder: &NetworkBuilder<BackendMarker>, stm_inputs: NetworkBuilderNode<BackendMarker>, ntm_inputs: NetworkBuilderNode<BackendMarker>| {
            let l0 = builder.new_affine("l0", stm_inputs.annotated_node().shape.size(), hl_size);
            let l1 = builder.new_affine("l1", 2 * hl_size, 1);

            // inference
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            let hidden_layer = stm_hidden.concat(ntm_hidden);
            l1.forward(hidden_layer)
        });

    let schedule = TrainingSchedule {
        net_id: "aether".to_string(),
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

    let dataloader = DirectSequentialDataLoader::new(&[dataset_path]);

    trainer.run(&schedule, &settings, &dataloader);
}
