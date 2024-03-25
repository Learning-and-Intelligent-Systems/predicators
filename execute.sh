#!/bin/bash

#SBATCH --job-name=exp
# SBATCH --partition=tenenbaum
# SBATCH --qos=tenenbaum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
# SBATCH --gres=gpu:a100:1
#SBATCH --gres=gpu:1
# SBATCH --constraint=high-capacity
#SBATCH --constraint=12GB
# SBATCH --constraint=3GB
#SBATCH --time=2-00:00
# SBATCH --time=1-00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=ycliang6@gmail.edu
#SBATCH --output=slurm_outputs/%x.%j.out
#SBATCH --error=slurm_outputs/%x.%j.err

source activate predl
cd /om2/user/ycliang/predicators

$@