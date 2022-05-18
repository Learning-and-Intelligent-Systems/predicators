#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"
# Note: this script is meant to be run first, to completion, with
# RUN_LOAD_EXPERIMENTS=false, then rerun with RUN_LOAD_EXPERIMENTS=true.
RUN_LOAD_EXPERIMENTS=false
NUM_TRAIN_TASKS=50
ALL_ENVS=(
    "repeated_nextto_single_option"
    "repeated_nextto_painting"
    "screws"
    "painting"
)

for ENV in ${ALL_ENVS[@]}; do
    COMMON_ARGS = "--env $ENV --num_train_tasks $NUM_TRAIN_TASKS"

    if [ "$RUN_LOAD_EXPERIMENTS" = true ]; then
        # Model-free GNN option policy baseline.
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_gnn_modelfree_${NUM_TRAIN_TASKS}demo --load_experiment_id ${ENV}_gnn_shooting_${NUM_TRAIN_TASKS}demo --approach gnn_option_policy --gnn_option_policy_solve_with_shooting False --load_approach --load_data
    else
        # Main backchaining approach.
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_backchaining_${NUM_TRAIN_TASKS}demo --approach nsrt_learning --strips_learner backchaining
        # Cluster-and-intersect (RLDM) baseline. Although it is guaranteed to
        # preserve harmlessness, we disable the check because it takes a long
        # time (since the operators have high arity).
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_cluster_and_intersect_${NUM_TRAIN_TASKS}demo --approach nsrt_learning --strips_learner cluster_and_intersect --disable_harmlessness_check True
        # LOFT baseline. Same note on harmlessness as for cluster-and-intersect.
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_cluster_and_search_${NUM_TRAIN_TASKS}demo --approach nsrt_learning --strips_learner cluster_and_search --disable_harmlessness_check True
        # Prediction error baseline that optimizes via hill climbing. Not
        # guaranteed to preserve harmlessness.
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_pred_error_${NUM_TRAIN_TASKS}demo --approach nsrt_learning --strips_learner cluster_and_intersect_sideline_prederror --disable_harmlessness_check True
        # Model-based GNN option policy baseline.
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_gnn_shooting_${NUM_TRAIN_TASKS}demo --approach gnn_option_policy
    fi

done
