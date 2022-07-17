#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"
ENVS=(
    "cover"
    "blocks"
    "painting"
    "tools"
)
EXPLORERS=(
    "random_options"
    "no_explore"
    "exploit_planning"
    "glib"
)

COMMON_ARGS=" --approach online_nsrt_learning \
    --max_initial_demos 1 \
    --num_train_tasks 1000 \
    --num_test_tasks 10 \
    --min_data_for_nsrt 10"

for ENV in ${ENVS[@]}; do
    for EXPLORER in ${EXPLORERS[@]}; do
         python $FILE $COMMON_ARGS --env $ENV --explorer $EXPLORER --experiment_id ${ENV}_nsrt_learning_${EXPLORER}
    done
done
