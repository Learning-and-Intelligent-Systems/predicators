#!/bin/bash

# Setting up the environment
source /etc/profile
module load anaconda/2023a-pytorch

# Running the code
nvidia-smi
FD_EXEC_PATH="downward" PYTHONHASHSEED=0 CUDA_VISIBLE_DEVICES=0,1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_LAUNCH_BLOCKING=1 python3 predicators/main.py $@