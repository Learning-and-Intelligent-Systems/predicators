
import itertools
for num_train_tasks, learning_rate, diffusion_arch, max_itr in itertools.product(
        [200, 2000, 20000],
        [0.001, 0.0005, 0.0001, 0.00005],
        [[512, 512], [256, 256, 256], [1024, 1024], [2048, 2048]],
        [500, 5000, 50000]
    ):
    print(f'''--env shelves2d
--approach nsrt_learning
--seed 0
--sesame_task_planner astar
--horizon 1000
--option_model_terminate_on_repeat false
--make_failure_videos
--make_test_videos
--video_fps 4
--strips_learner oracle
--option_learner no_learning
--sesame_max_samples_per_step 100
--timeout 10000
--num_test_tasks 200
--num_train_tasks {num_train_tasks}
--sesame_max_skeletons_optimized 1
--learning_rate {learning_rate}
--sampler_disable_classifier true
--sampler_learning_regressor_model diffusion
--diffusion_regressor_hid_sizes {diffusion_arch}
--diffusion_regressor_max_itr {max_itr}
--diffusion_regressor_timesteps 100
'''.replace("\n", " "))