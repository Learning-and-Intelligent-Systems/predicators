
import itertools
for num_train_tasks, learning_rate, diffusion_arch, max_itr in itertools.product(
        [500, 5000, 50000],
        [0.001, 0.0005, 0.0001, 0.00005],
        [[512, 512], [256, 256, 256], [1024, 1024], [2048, 2048]],
        [500, 5000, 50000]
    ):
    flags = [
        "--env", "shelves2d",
        "--approach", "nsrt_learning",
        "--seed", "0",
        "--sesame_task_planner", "astar",
        "--horizon", "1000",
        "--option_model_terminate_on_repeat", "false",
        "--make_failure_videos",
        "--make_test_videos",
        "--video_fps", "4",
        "--strips_learner", "oracle",
        "--option_learner", "no_learning",
        "--sesame_max_samples_per_step", "1000",
        "--timeout", "10000",
        "--num_test_tasks", "1",
        "--num_train_tasks", str(num_train_tasks),
        "--sesame_max_skeletons_optimized", "1",
        "--learning_rate", str(learning_rate),
        "--sampler_disable_classifier", "true",
        "--sampler_learning_regressor_model", "diffusion",
        "--diffusion_regressor_hid_sizes", str(diffusion_arch),
        "--diffusion_regressor_max_itr", str(max_itr),
        "--diffusion_regressor_timesteps", "100"
    ]
    escaped_flags = [flag.replace(" ", "\\ ") for flag in flags]

    out = f"mapreduce_in/num_train_tasks-{num_train_tasks}_learning_rate-{learning_rate}_diffusion_arch-{diffusion_arch}_max_itr-{max_itr}.pdf".replace(" ","")
    print(" ".join(escaped_flags), file=open(out, "w"))