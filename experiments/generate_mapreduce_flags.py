
import itertools
import numpy as np
import os
for seed, in itertools.product(range(40)):
    signature = (f"seed-{seed}").replace(" ","")
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
        "--sesame_max_samples_per_step", "15",
        "--timeout", "240",
        "--num_test_tasks", "50",
        "--num_train_tasks", "8000",
        "--sesame_max_skeletons_optimized", "1",
        "--learning_rate", "0.0005",
        "--sampler_disable_classifier", "true",
        "--sampler_learning_regressor_model", "diffusion",
        "--diffusion_regressor_hid_sizes", "[512,512]",
        "--diffusion_regressor_max_itr", "10000",
        "--diffusion_regressor_timesteps", "50",
        "--feasibility_learning_strategy", "backtracking",
        "--use_torch_gpu", "true",
        "--results_dir", f"mapreduce_out/{signature}",
    ]

    os.makedirs(f"mapreduce_out/{signature}", exist_ok=True)

    out = f"mapreduce_in/{signature}.txt"
    print(" ".join(flags), file=open(out, "w"))
