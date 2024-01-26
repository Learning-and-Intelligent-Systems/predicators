
import itertools
for featurizer_sizes, feasibility_general_lr, feasibility_transformer_lr, embedding_concat, transformer_num_blocks in itertools.product(
        [[1024], [128, 128, 128], [1024, 1024, 1024]],
        [0.01, 0.001, 0.0001],
        [1.0, 0.01, 0.001, 0.0001],
        [True, False],
        [1, 2, 4]
    ):
    signature = (f"featurizer_sizes-{featurizer_sizes}_feasibility_general_lr-{feasibility_general_lr}_"
        f"feasibility_transformer_lr-{feasibility_transformer_lr}_embedding_concat-{embedding_concat}_"
        f"transformer_num_blocks-{transformer_num_blocks}").replace(" ","")
    flags = [
        "--env", "shelves2d",
        "--approach", "search_pruning",
        "--seed", "1",
        "--sesame_task_planner", "astar",
        "--horizon", "1000",
        "--option_model_terminate_on_repeat", "false",
        "--make_failure_videos",
        "--make_test_videos",
        "--video_fps", "4",
        "--strips_learner", "oracle",
        "--option_learner", "no_learning",
        "--sesame_max_samples_per_step", "10",
        "--timeout", "3000",
        "--num_test_tasks", "100",
        "--num_train_tasks", "4000",
        "--sesame_max_skeletons_optimized", "1",
        "--learning_rate", "0.0001",
        "--sampler_disable_classifier", "true",
        "--sampler_learning_regressor_model", "diffusion",
        "--diffusion_regressor_hid_sizes", "[512,512]",
        "--diffusion_regressor_max_itr", "0",
        "--diffusion_regressor_timesteps", "100",
        "--feasibility_learning_strategy", "ground_truth_data",
        "--feasibility_loss_output_file", f"mapreduce_out/{signature}"
        "--feasibility_featurizer_hid_sizes", str(featurizer_sizes[:-1]).replace(" ", ""),
        "--feasibility_feature_size", str(featurizer_sizes[-1]),
        "--feasibility_embedding_size", str(featurizer_sizes[-1] // 2),
        "--feasibility_embedding_concat", str(embedding_concat),
        "--feasibility_enc_num_layers", str(transformer_num_blocks),
        "--feasibility_dec_num_layers", str(transformer_num_blocks),
        "--feasibility_ffn_hid_size", str(featurizer_sizes[-1] * 4),
        "--feasibility_general_lr", str(feasibility_general_lr),
        "--feasibility_transformer_lr", str(feasibility_transformer_lr),
    ]

    out = f"mapreduce_in/{signature}.txt"
    print(" ".join(flags), file=open(out, "w"))