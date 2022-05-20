#!/bin/bash

# Note: this script is too large to be run all at once. Comment things out
# and run in multiple passes. For example, start with just 1000 train tasks.
# If that looks good, launch the environments in separate runs.

FILE="scripts/supercloud/submit_supercloud_job.py"
# Note: this script is meant to be run first, to completion, with
# RUN_LOAD_EXPERIMENTS=false, then rerun with RUN_LOAD_EXPERIMENTS=true.
RUN_LOAD_EXPERIMENTS=false
ALL_NUM_TRAIN_TASKS=(
    # "50"
    # "100"
    # "250"
    # "500"
    "1000"
)
ENVS=(
    "cover_multistep_options"
    "stick_button"
    "doors"
    "coffee"
)

for ENV in ${ENVS[@]}; do
    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do

        COMMON_ARGS="--env $ENV --min_perc_data_for_nsrt 1 \
                --segmenter contacts --num_train_tasks $NUM_TRAIN_TASKS --timeout 300 \
                --gnn_num_epochs 1000 --disable_harmlessness_check True \
                --neural_gaus_regressor_max_itr 50000"

        # Include the motion planning options for the doors environment.
        if [ $ENV = "doors" ]; then
            INCLUDED_OPTIONS="--included_options MoveToDoor,MoveThroughDoor"
        else
            INCLUDED_OPTIONS=""
        fi

        if [ "$RUN_LOAD_EXPERIMENTS" = true ]; then
            # direct BC max skeletons 1
            python $FILE $COMMON_ARGS $INCLUDED_OPTIONS --load_experiment_id ${ENV}_main_${NUM_TRAIN_TASKS} --experiment_id ${ENV}_direct_bc_max_skel1_${NUM_TRAIN_TASKS} --approach nsrt_learning --option_learner direct_bc --sesame_max_skeletons_optimized 1 --load_a --load_d

            # direct BC max samples 1
            python $FILE $COMMON_ARGS $INCLUDED_OPTIONS --load_experiment_id ${ENV}_main_${NUM_TRAIN_TASKS} --experiment_id ${ENV}_direct_bc_max_samp1_${NUM_TRAIN_TASKS} --approach nsrt_learning --option_learner direct_bc --sesame_max_samples_per_step 1 --load_a --load_d
        else
            # nsrt learning (oracle operators and options)
            # note: $INCLUDED_OPTIONS excluded because all options are
            # included for this oracle approach.
            python $FILE $COMMON_ARGS --experiment_id ${ENV}_oracle_options_${NUM_TRAIN_TASKS} --approach nsrt_learning --strips_learner oracle

            # direct BC (main approach)
            python $FILE $COMMON_ARGS $INCLUDED_OPTIONS --experiment_id ${ENV}_main_${NUM_TRAIN_TASKS} --approach nsrt_learning --option_learner direct_bc

            # GNN metacontroller with direct BC options
            python $FILE $COMMON_ARGS $INCLUDED_OPTIONS --experiment_id ${ENV}_gnn_metacontroller_param_${NUM_TRAIN_TASKS} --approach gnn_metacontroller --option_learner direct_bc

            # GNN action policy BC
            # python $FILE $COMMON_ARGS $INCLUDED_OPTIONS --experiment_id ${ENV}_gnn_action_policy_${NUM_TRAIN_TASKS} --approach gnn_action_policy

            # direct BC with nonparameterized options
            python $FILE $COMMON_ARGS $INCLUDED_OPTIONS --experiment_id ${ENV}_direct_bc_nonparam_${NUM_TRAIN_TASKS} --approach nsrt_learning --option_learner direct_bc_nonparameterized --mlp_regressor_max_itr 60000

            # GNN metacontroller with nonparameterized options
            python $FILE $COMMON_ARGS $INCLUDED_OPTIONS --experiment_id ${ENV}_gnn_metacontroller_nonparam_${NUM_TRAIN_TASKS} --approach gnn_metacontroller --option_learner direct_bc_nonparameterized --mlp_regressor_max_itr 60000
        fi
    done
done
