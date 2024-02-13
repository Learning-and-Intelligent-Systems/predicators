
import itertools
import numpy as np
import os
for num_datapoints, feasibility_itr, feasibility_width, max_samples_per_step in itertools.product(
        [100, 300, 1200, 2500],
        [3000, 4000, 6000],
        [8, 16, 32, 64, 128],
        [5, 10, 20]
    ):
    signature = (f"num_datapoints-{num_datapoints}_feasibility_itr-{feasibility_itr}_feasibility_width-{feasibility_width}_max_samples_per_step-{max_samples_per_step}").replace(" ","")
    flags = [
        "--env", "shelves2d",
        "--approach", "search_pruning",
        "--seed", "1",
        "--sesame_task_planner", "astar",
        "--horizon", "1000",
        "--option_model_terminate_on_repeat", "false",
        # "--make_failure_videos",
        # "--make_test_videos",
        # "--video_fps", "4",
        "--strips_learner", "oracle",
        "--option_learner", "no_learning",
        "--sesame_max_samples_per_step", str(max_samples_per_step),
        "--timeout", "150",
        "--num_test_tasks", "40",
        "--num_train_tasks", "8000",
        "--sesame_max_skeletons_optimized", "1",
        "--learning_rate", "0.0005",
        "--sampler_disable_classifier", "true",
        "--sampler_learning_regressor_model", "diffusion",
        "--diffusion_regressor_hid_sizes", "[512,512]",
        "--diffusion_regressor_max_itr", "10000",
        "--diffusion_regressor_timesteps", "50",
        "--feasibility_learning_strategy", "load_data",
        "--feasibility_feature_size", str(feasibility_width),
        "--feasibility_embedding_size", str(feasibility_width//2),
        "--feasibility_ffn_hid_size", str(feasibility_width*4),
        "--feasibility_featurizer_hid_sizes", f"[{feasibility_width},{feasibility_width}]",
        "--feasibility_max_itr", str(feasibility_itr),
        "--feasibility_num_negative_loaded_datapoints", str(num_datapoints),
        "--results_dir", f"mapreduce_out/{signature}",
        "--load_data",
    ]

    os.makedirs(f"mapreduce_out/{signature}", exist_ok=True)

    out = f"mapreduce_in/{signature}.txt"
    print(" ".join(flags), file=open(out, "w"))
