#!/bin/bash
# Run with: sbatch -p normal --time=99:00:00 --partition=xeon-p8 --nodes=1 --exclusive --job-name=debug --array=1-50 -o logs/debug_%j.log scripts/supercloud/supercloud_debug.sh
python src/main.py --env cover --approach oracle --seed $SLURM_ARRAY_TASK_ID
