#!/bin/bash

for SEED in {456..465}
do
    python src/main.py --experiment_id cover_nsrt_baseline --approach nsrt_learning --env singleton_cover --seed $SEED --load_data  --num_train_tasks 1000
done