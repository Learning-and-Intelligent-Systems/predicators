#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"
NUM_TRAIN_TASKS="200"
ALL_ENVS=(
    "cover"
    "pybullet_blocks"
    "painting"
    "tools"
)

# We want this to crash if backup_results already exists, because overwriting it would be bad.
mkdir backup_results && mv results/* backup_results

if [ $? -ne 0 ]; then
    echo "backup_results/ already exists, exiting"
    exit 1
fi

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for ENV in ${ALL_ENVS[@]}; do
        # Downrefeval ablation.
        echo python $FILE --experiment_id ${ENV}_main_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS --sesame_max_skeletons_optimized 1 --load_approach --load_data

        # GNN model-free baseline.
        echo python $FILE --experiment_id ${ENV}_gnn_shooting_${NUM_TRAIN_TASKS}demo --env $ENV --approach gnn_option_policy --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS --gnn_option_policy_solve_with_shooting False --load_approach --load_data
    done
done

# Commands to run after all jobs are finished:
# for i in results/*main* ; do mv "$i" "${i/main/downrefeval}" ; done
# for i in results/*gnn* ; do mv "$i" "${i/gnn_shooting/gnn_modelfree}" ; done
# mv backup_results/* results/
# rm -r backup_results
