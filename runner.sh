#!/bin/bash

#SBATCH --exclusive
#SBATCH --partition=xeon-g6-volta
#SBATCH --output=runner7.out

# Setting up the environment
source /etc/profile
module load anaconda/2023a-pytorch

# Running the code
nvidia-smi
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0,1 python3 predicators/main.py --env donuts --approach search_pruning --seed 7 --sesame_task_planner "astar" --horizon 1000 --option_model_terminate_on_repeat false --make_failure_videos --make_test_videos --video_fps 4 --strips_learner oracle --option_learner no_learning --sesame_max_samples_per_step 40 --timeout 30 --num_test_tasks 50 --num_train_tasks 8000 --sesame_max_skeletons_optimized 1 --learning_rate 0.0005 --sampler_disable_classifier true --sampler_learning_regressor_model diffusion --diffusion_regressor_hid_sizes "[64, 64]" --diffusion_regressor_max_itr 10000 --diffusion_regressor_timesteps 50 --feasibility_learning_strategy backtracking --feasibility_max_itr 4000 --use_torch_gpu true