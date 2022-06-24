#!/bin/bash

# Note: this script is too large to be run all at once. Comment things out
# and run in multiple passes.

FILE="scripts/supercloud/submit_supercloud_job.py"
# Note: this script is meant to be run first, to completion, with
# RUN_LOAD_EXPERIMENTS=false, then rerun with RUN_LOAD_EXPERIMENTS=true.
RUN_LOAD_EXPERIMENTS=false
ALL_ENVS=(
    "repeated_nextto_single_option"
    "repeated_nextto_painting"
    "screws"
    "painting"
    "satellites"
    "satellites_simple"
)

for ENV in ${ALL_ENVS[@]}; do
    COMMON_ARGS="--env $ENV"

    if [ "$RUN_LOAD_EXPERIMENTS" = true ]; then
        # Model-free GNN option policy baseline.
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_gnn_modelfree_50demo --load_experiment_id ${ENV}_gnn_shooting_50demo --approach gnn_option_policy --num_train_tasks 50 --gnn_option_policy_solve_with_shooting False --load_approach --load_data
    else
        # # Oracle.
        # python $FILE $COMMON_ARGS --experiment_id ${ENV}_oracle --approach oracle --num_train_tasks 0
        # Main backchaining approach with various numbers of demonstrations.
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_backchaining_5demo --approach nsrt_learning --strips_learner backchaining --num_train_tasks 5
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_backchaining_10demo --approach nsrt_learning --strips_learner backchaining --num_train_tasks 10
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_backchaining_25demo --approach nsrt_learning --strips_learner backchaining --num_train_tasks 25
        python $FILE $COMMON_ARGS --experiment_id ${ENV}_backchaining_50demo --approach nsrt_learning --strips_learner backchaining --num_train_tasks 50
        # # Cluster-and-intersect (RLDM) baseline. Although it is guaranteed to
        # # preserve harmlessness, we disable the check because it takes a long
        # # time (since the operators have high arity).
        # python $FILE $COMMON_ARGS --experiment_id ${ENV}_cluster_and_intersect_50demo --approach nsrt_learning --strips_learner cluster_and_intersect --num_train_tasks 50 --disable_harmlessness_check True
        # # LOFT baseline. Same note on harmlessness as for cluster-and-intersect.
        # python $FILE $COMMON_ARGS --experiment_id ${ENV}_cluster_and_search_50demo --approach nsrt_learning --strips_learner cluster_and_search --num_train_tasks 50 --disable_harmlessness_check True
        # # Prediction error baseline that optimizes via hill climbing. Not
        # # guaranteed to preserve harmlessness.
        # python $FILE $COMMON_ARGS --experiment_id ${ENV}_pred_error_50demo --approach nsrt_learning --strips_learner cluster_and_intersect_sideline_prederror --num_train_tasks 50 --disable_harmlessness_check True
        # # Model-based GNN option policy baseline.
        # python $FILE $COMMON_ARGS --experiment_id ${ENV}_gnn_shooting_50demo --approach gnn_option_policy --num_train_tasks 50
    fi

done
