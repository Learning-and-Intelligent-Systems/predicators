#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"
ALL_NUM_TRAIN_TASKS=(
    # "25"
    # "50"
    # "75"
    # "100"
    # "125"
    # "150"
    # "175"
    "200"
)
ALL_ENVS=(
    "cover_regrasp"
    "blocks"
    "painting"
    "tools"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for ENV in ${ALL_ENVS[@]}; do
        echo "$FILE --experiment_id ${ENV}_random --env $ENV --approach random_options --seed $SEED"

        for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do
            echo "$FILE --experiment_id ${ENV}_main_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"
            echo "$FILE --experiment_id ${ENV}_downref_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --sesame_max_skeletons_optimized 1 --offline_data_max_skeletons_optimized 8 --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"
            echo "$FILE --experiment_id ${ENV}_prederror_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --grammar_search_score_function prediction_error --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"
            echo "$FILE --experiment_id ${ENV}_branchfac_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --grammar_search_score_function branching_factor --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"
            echo "$FILE --experiment_id ${ENV}_energy_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --grammar_search_score_function lmcut_energy_lookaheaddepth0 --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"
            echo "$FILE --experiment_id ${ENV}_noinventallexclude_${NUM_TRAIN_TASKS}demo --env $ENV --approach nsrt_learning --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"
            echo "$FILE --experiment_id ${ENV}_noinventnoexclude_${NUM_TRAIN_TASKS}demo --env $ENV --approach nsrt_learning --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"

            # Blocks-specific experiments.
            if [ $ENV = "blocks" ]; then
                echo "$FILE --experiment_id ${ENV}_mainhadd_${NUM_TRAIN_TASKS}demo --env $ENV --approach grammar_search_invention --sesame_task_planning_heuristic hadd --offline_data_task_planning_heuristic lmcut --excluded_predicates all --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"
                echo "$FILE --experiment_id ${ENV}_noinventnoexcludehadd_${NUM_TRAIN_TASKS}demo --env $ENV --approach nsrt_learning --sesame_task_planning_heuristic hadd --offline_data_task_planning_heuristic lmcut --seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"
            fi
        done
    done
done
