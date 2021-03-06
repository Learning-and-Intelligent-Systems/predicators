#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"
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

for ENV in ${ALL_ENVS[@]}; do
    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do
        # Main approach.
        python $FILE --experiment_id ${ENV}_main_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS
        # GNN option policy approach.
        python $FILE --experiment_id ${ENV}_gnn_shooting_${NUM_TRAIN_TASKS}demo --env $ENV --approach gnn_option_policy --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS
    done
done
