#!/bin/bash

# Note: this script is too large to be run all at once. Comment things out
# and run in multiple passes. For example, start with just 1000 train tasks.
# If that looks good, launch the environments in separate runs.

FILE="scripts/supercloud/submit_supercloud_job.py"
# Note: this script is meant to be run first, to completion, with
# RUN_LOAD_EXPERIMENTS=false, then rerun with RUN_LOAD_EXPERIMENTS=true.
RUN_LOAD_EXPERIMENTS=false
ALL_NUM_TRAIN_TASKS=(
     "50"
    # "100"
    # "250"
    # "500"
    #"1000"
)
ENVS=(
    "cover"
    "painting"
    "screws"
    "repeated_nextto"
    "cluttered_table"
    "coffee"
)

for ENV in ${ENVS[@]}; do
    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do

        COMMON_ARGS="--env $ENV  --sampler_learner oracle \
                 --num_train_tasks $NUM_TRAIN_TASKS \"

        # nsrt learning (oracle operators and options)
        # note: $INCLUDED_OPTIONS excluded because all options are
        # included for this oracle approach.
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_oracle_options_${NUM_TRAIN_TASKS} --approach pg3 --strips_learner oracle

    done
done
