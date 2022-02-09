#!/bin/bash

START_SEED=456
NUM_SEEDS=50
FILE="analysis/submit.py"
ALL_NUM_TRAIN_TASKS=(
    "50"
    # "100"
    # "150"
    # "200"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    python $FILE --experiment_id oracle --env repeated_nextto --approach oracle --seed $SEED

    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do
        COMMON_ARGS="--seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"

        # repeated_nextto
        python $FILE --experiment_id noside --env repeated_nextto --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id withside --env repeated_nextto --approach nsrt_learning --learn_side_predicates True $COMMON_ARGS
    done
done
