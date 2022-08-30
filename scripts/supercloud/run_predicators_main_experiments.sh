#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"
NUM_TRAIN_TASKS="200"
ALL_ENVS=(
    "cover"
    "pybullet_blocks"
)

for ENV in ${ALL_ENVS[@]}; do
    # python $FILE --experiment_id ${ENV}_oracle --env $ENV --approach oracle --num_train_tasks $NUM_TRAIN_TASKS
    # python $FILE --experiment_id ${ENV}_nsrt_learning --env $ENV --approach nsrt_learning --num_train_tasks $NUM_TRAIN_TASKS
    python $FILE --experiment_id ${ENV}_gnn_yesgpu --env $ENV --approach gnn_option_policy --excluded_predicates all --num_train_tasks $NUM_TRAIN_TASKS
done
