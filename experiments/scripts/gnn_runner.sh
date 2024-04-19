#!/bin/bash

#SBATCH --exclusive
#SBATCH --partition=xeon-g6-volta
#SBATCH --output=shelves2d-gnn.out

# Setting up the environment
source /etc/profile
module load anaconda/2023a-pytorch

# Running the code
nvidia-smi
FD_EXEC_PATH="downward" PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 predicators/main.py --env shelves2d --approach gnn_action_policy --seed 1 --sesame_task_planner "fdsat" --horizon 10000 --option_model_terminate_on_repeat false --make_failure_videos --make_test_videos --video_fps 4 --strips_learner oracle --option_learner no_learning --diffusion_regressor_timesteps 50 --sesame_max_samples_per_step 15 --timeout 90 --num_test_tasks 0 --num_train_tasks 1000 --sesame_max_skeletons_optimized 1 --learning_rate 0.0001 --use_torch_gpu true --load_data