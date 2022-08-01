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
NUM_DEMOS=(
    5
    10
    25
    50
)

for ENV in ${ALL_ENVS[@]}; do
    COMMON_ARGS="--env $ENV"
    # Oracle.
    python $FILE $COMMON_ARGS --experiment_id ${ENV}_oracle --approach oracle --num_train_tasks 0

    for DEMOS in ${NUM_DEMOS[@]}; do
        if [ "$RUN_LOAD_EXPERIMENTS" = true ]; then
            # Model-free GNN option policy baseline.
            python $FILE $COMMON_ARGS --experiment_id ${ENV}_gnn_modelfree_${DEMOS}demo --load_experiment_id ${ENV}_gnn_shooting_${DEMOS}demo --approach gnn_option_policy --num_train_tasks ${DEMOS} --gnn_option_policy_solve_with_shooting False --load_approach --load_data
        else
            # Main backchaining approach with various numbers of demonstrations.
            python $FILE $COMMON_ARGS --experiment_id ${ENV}_backchaining_${DEMOS}demo --approach nsrt_learning --strips_learner backchaining --num_train_tasks ${DEMOS}
            # Cluster-and-intersect (RLDM) baseline. Although it is guaranteed to
            # preserve harmlessness, we disable the check because it takes a long
            # time (since the operators have high arity).
            python $FILE $COMMON_ARGS --experiment_id ${ENV}_cluster_and_intersect_${DEMOS}demo --approach nsrt_learning --strips_learner cluster_and_intersect --num_train_tasks ${DEMOS} --disable_harmlessness_check True
            # LOFT baseline. Same note on harmlessness as for cluster-and-intersect.
            python $FILE $COMMON_ARGS --experiment_id ${ENV}_cluster_and_search_${DEMOS}demo --approach nsrt_learning --strips_learner cluster_and_search --num_train_tasks ${DEMOS} --disable_harmlessness_check True
            # Prediction error baseline that optimizes via hill climbing. Not
            # guaranteed to preserve harmlessness.
            python $FILE $COMMON_ARGS --experiment_id ${ENV}_pred_error_${DEMOS}demo --approach nsrt_learning --strips_learner cluster_and_intersect_sideline_prederror --num_train_tasks ${DEMOS} --disable_harmlessness_check True
            # Model-based GNN option policy baseline.
            python $FILE $COMMON_ARGS --experiment_id ${ENV}_gnn_shooting_${DEMOS}demo --approach gnn_option_policy --num_train_tasks ${DEMOS}
        fi
    done
done
