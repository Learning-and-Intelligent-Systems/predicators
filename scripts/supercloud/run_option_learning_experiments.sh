#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    ## all in the stick point environment
    ## all with oracle operators and segmentation for now!
    COMMON_ARGS="--env stick_point --approach nsrt_learning --strips_learner oracle --segmenter oracle --seed $SEED --load_a --load_d"

    # nsrt learning (oracle options) 500, default test time
    python $FILE $COMMON_ARGS --experiment_id given_500 --num_train_tasks 500

    # nsrt learning (oracle options) 5000, default test time
    python $FILE $COMMON_ARGS --experiment_id given_5000 --num_train_tasks 5000

    # nsrt learning (oracle options) 500, long test time
    python $FILE $COMMON_ARGS --experiment_id given_500_long --num_train_tasks 500 --timeout 300

    # nsrt learning (oracle options) 5000, long test time
    python $FILE $COMMON_ARGS --experiment_id given_5000_long --num_train_tasks 5000 --timeout 300

    # direct BC 500, default test time
    python $FILE $COMMON_ARGS --experiment_id direct_bc_500 --option_learner direct_bc --num_train_tasks 500

    # direct BC 5000, default test time
    python $FILE $COMMON_ARGS --experiment_id direct_bc_5000 --option_learner direct_bc --num_train_tasks 5000

    # direct BC 500, long test time
    python $FILE $COMMON_ARGS --experiment_id direct_bc_500_long --option_learner direct_bc --num_train_tasks 500 --timeout 300

    # direct BC 5000, long test time
    python $FILE $COMMON_ARGS --experiment_id direct_bc_5000_long --option_learner direct_bc --num_train_tasks 5000 --timeout 300

done
