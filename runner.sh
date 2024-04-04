#!/bin/bash

#SBATCH --exclusive
#SBATCH --partition=xeon-g6-volta
#SBATCH --output=runner-jigsaw-1.out

# Setting up the environment
# source /etc/profile
# module load anaconda/2023a-pytorch

# Running the code
nvidia-smi
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 predicators/main.py --env jigsaw --approach search_pruning --seed 2 --sesame_task_planner "astar" --horizon 10000 --option_model_terminate_on_repeat false --make_failure_videos --make_test_videos --video_fps 4 --strips_learner oracle --option_learner no_learning --diffusion_regressor_timesteps 100 --sesame_max_samples_per_step 40 --timeout 90 --num_test_tasks 50 --num_train_tasks 1000 --sesame_max_skeletons_optimized 1 --learning_rate 0.0001 --sampler_disable_classifier true --sampler_learning_regressor_model diffusion --use_torch_gpu true --disable_harmlessness_check true --feasibility_l1_penalty 0 --feasibility_l2_penalty 0 --feasibility_max_itr 6000 --feasibility_num_datapoints_per_iter 3000 --shelves2d_test_num_boxes 5 --feasibility_learning_strategy backtracking --feasibility_load_path "" --feasibility_debug_directory "jigsaw-2/" --feasibility_num_layers 2 --feasibility_token_size 256 --feasibility_threshold_recalibration_percentile 0.0 --diffusion_regressor_timesteps 100 --diffusion_regressor_hid_sizes "[512, 512]" --diffusion_regressor_max_itr 15000 --load_data