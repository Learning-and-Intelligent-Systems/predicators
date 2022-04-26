#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"
# Note: this script is meant to be run first, to completion, with
# RUN_LOAD_EXPERIMENTS=false, then rerun with RUN_LOAD_EXPERIMENTS=true.
RUN_LOAD_EXPERIMENTS=false
ALL_NUM_TRAIN_TASKS=(
    "50"
    "100"
    "250"
    "500"
    "1000"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do

        # NOTE: segmenter is oracle
        COMMON_ARGS="--env stick_point --min_perc_data_for_nsrt 1 \
            --segmenter oracle --num_train_tasks $NUM_TRAIN_TASKS --timeout 300 \
            --seed $SEED"

        if [ "$RUN_LOAD_EXPERIMENTS" = true ]; then
            # direct BC max skeletons 1
            python $FILE $COMMON_ARGS --load_experiment_id direct_bc_${NUM_TRAIN_TASKS} --experiment_id direct_bc_max_skel1_${NUM_TRAIN_TASKS} --approach nsrt_learning --option_learner direct_bc --sesame_max_skeletons_optimized 1 --load_a --load_d

            # direct BC max samples 1
            python $FILE $COMMON_ARGS --load_experiment_id direct_bc_${NUM_TRAIN_TASKS} --experiment_id direct_bc_max_samp1_${NUM_TRAIN_TASKS} --approach nsrt_learning --option_learner direct_bc --sesame_max_samples_per_step 1 --load_a --load_d

            # GNN BC model-free
            python $FILE $COMMON_ARGS --load_experiment_id gnn_shooting_${NUM_TRAIN_TASKS} --experiment_id gnn_modelfree_${NUM_TRAIN_TASKS} --approach gnn_policy --gnn_policy_solve_with_shooting False --load_a --load_d
        else
            # nsrt learning (oracle operators and options)
            python $FILE $COMMON_ARGS --experiment_id oracle_options_${NUM_TRAIN_TASKS} --approach nsrt_learning --strips_learner oracle

            # direct BC (main approach)
            python $FILE $COMMON_ARGS --experiment_id direct_bc_${NUM_TRAIN_TASKS} --approach nsrt_learning --option_learner direct_bc

            # GNN BC with shooting baseline
            python $FILE $COMMON_ARGS --experiment_id gnn_shooting_${NUM_TRAIN_TASKS} --approach gnn_policy

            # direct BC with nonparameterized options
            python $FILE $COMMON_ARGS --experiment_id direct_bc_nonparam_${NUM_TRAIN_TASKS} --approach nsrt_learning --option_learner direct_bc_nonparameterized

            # GNN metacontroller with nonparameterized options
            python $FILE $COMMON_ARGS --experiment_id gnn_metacontroller_${NUM_TRAIN_TASKS} --approach gnn_metacontroller --option_learner direct_bc_nonparameterized
        fi

    done
done
