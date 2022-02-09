#!/bin/bash

START_SEED=456
NUM_SEEDS=20
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
        python $FILE --experiment_id withside15.0 --env repeated_nextto --approach nsrt_learning --learn_side_predicates True $COMMON_ARGS --side_predicates_numsidepreds_weight 15.0
        python $FILE --experiment_id withside1.0 --env repeated_nextto --approach nsrt_learning --learn_side_predicates True $COMMON_ARGS --side_predicates_numsidepreds_weight 1.0
        python $FILE --experiment_id withside0.1 --env repeated_nextto --approach nsrt_learning --learn_side_predicates True $COMMON_ARGS --side_predicates_numsidepreds_weight 0.1
        python $FILE --experiment_id withside0.01 --env repeated_nextto --approach nsrt_learning --learn_side_predicates True $COMMON_ARGS --side_predicates_numsidepreds_weight 0.01
        python $FILE --experiment_id withside0.001 --env repeated_nextto --approach nsrt_learning --learn_side_predicates True $COMMON_ARGS --side_predicates_numsidepreds_weight 0.001
        python $FILE --experiment_id withside0.0 --env repeated_nextto --approach nsrt_learning --learn_side_predicates True $COMMON_ARGS --side_predicates_numsidepreds_weight 0.0
    done
done
