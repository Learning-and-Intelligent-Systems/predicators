#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="src/main.py"
#FILE="scripts/supercloud/submit_supercloud_job.py"
ALL_NUM_DEMOS=(
    "1"
    "5"
    "10"
    "20"
    "50"
)
ALL_ENVS=(
    "play_blocks"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for ENV in ${ALL_ENVS[@]}; do
        for NUM_DEMOS in ${ALL_NUM_DEMOS[@]}; do
            python $FILE --experiment_id ${ENV}_nsrt_learning_${NUM_DEMOS}demo --env $ENV --approach nsrt_learning --seed $SEED --max_initial_demos $NUM_DEMOS
        done
    done
done
