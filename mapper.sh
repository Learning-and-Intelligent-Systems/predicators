#!/bin/bash

# Setting up the environment
source /etc/profile
module load anaconda/2023a-pytorch

# Running the code
nvidia-smi
eval "PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0,1 python3 predicators/main.py $(cat $1)"
