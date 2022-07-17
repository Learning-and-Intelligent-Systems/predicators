#!/bin/bash

MAX_TRANSITIONS=10000  # want this to be stop signal
CYCLES=1000000  # way too many cycles

FILE="scripts/supercloud/submit_supercloud_job.py"
ENVS=(
    "cover"
    "blocks"
    "painting"
    "tools"
)
EXPLORERS=(
    "random_options"
    "exploit_planning"
    "glib"
)

COMMON_ARGS=" --approach online_nsrt_learning \
    --max_initial_demos 1 \
    --num_train_tasks 1000 \
    --num_online_learning_cycles $CYCLES \
    --online_learning_max_transitions $MAX_TRANSITIONS \
    --num_test_tasks 10 \
    --min_data_for_nsrt 10 \
    --sesame_allow_noops False"

for ENV in ${ENVS[@]}; do
    for EXPLORER in ${EXPLORERS[@]}; do
         python $FILE $COMMON_ARGS --env $ENV --explorer $EXPLORER --experiment_id ${ENV}_nsrt_learning_${EXPLORER}
    done
done
