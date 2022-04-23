#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    COMMON_ARGS="--env stick_point --min_data_for_nsrt 10 --segmenter oracle \
        --num_train_tasks 500 --timeout 300 --seed $SEED"

    # nsrt learning (oracle operators and options)
    python $FILE $COMMON_ARGS --experiment_id oracle_options --approach nsrt_learning --strips_learner oracle

    # direct BC (main approach)
    python $FILE $COMMON_ARGS --experiment_id direct_bc --approach nsrt_learning --option_learner direct_bc

    # GNN BC with shooting baseline
    python $FILE $COMMON_ARGS --experiment_id gnn_shooting --approach gnn_policy

    # GNN BC model-free
    python $FILE $COMMON_ARGS --experiment_id gnn_modelfree --approach gnn_policy --gnn_policy_solve_with_shooting False

done
