#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"
ALL_ENVS=(
    "cover"
    "blocks"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for ENV in ${ALL_ENVS[@]}; do

        # GNN option policy (model-free).
        python $FILE --experiment_id ${ENV}_options_mf --env $ENV --approach gnn_option_policy --seed $SEED --gnn_option_policy_solve_with_shooting False

        # GNN action policy (model-free).
        python $FILE --experiment_id ${ENV}_actions_mf --env $ENV --approach gnn_action_policy --seed $SEED --gnn_option_policy_solve_with_shooting False

    done
done