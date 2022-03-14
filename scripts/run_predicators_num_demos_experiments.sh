#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/submit.py"
ALL_NUM_TRAIN_TASKS=(
    "25"
    "50"
    "75"
    "100"
    "125"
    "150"
    "175"
)
ALL_ENVS=(
    "cover"
    "pybullet_blocks"
    "painting"
    "tools"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for ENV in ${ALL_ENVS[@]}; do
        for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do
            # Main approach.
            python $FILE --experiment_id ${ENV}_main_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS
            # GNN approach.
            python $FILE --experiment_id ${ENV}_gnn_shooting_${NUM_TRAIN_TASKS}demo --env $ENV --approach gnn_policy --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS
        done
    done
done
