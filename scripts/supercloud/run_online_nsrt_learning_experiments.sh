#!/bin/bash

MAX_TRANSITIONS=2500  # want this to be stop signal
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
    --max_num_steps_interaction_request 25 \
    --num_test_tasks 10 \
    --min_data_for_nsrt 10 \
    --sesame_allow_noops False \
    --glib_max_goal_size 2"

for ENV in ${ENVS[@]}; do
    for EXPLORER in ${EXPLORERS[@]}; do
         python $FILE $COMMON_ARGS --env $ENV --explorer $EXPLORER --experiment_id ${ENV}_nsrt_learning_${EXPLORER}
    done
done
