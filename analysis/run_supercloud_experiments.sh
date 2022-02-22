#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"
ALL_NUM_TRAIN_TASKS=(
    "200"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do
        COMMON_ARGS="--seed $SEED --num_train_tasks $NUM_TRAIN_TASKS --nsrt_learning_learn_goal_predicates True --offline_data_method demo+goal_atoms"

        # NSRT learning
        python $FILE --experiment_id cover_nsrt_learning_$NUM_TRAIN_TASKS --approach nsrt_learning --excluded_predicates goals --env cover $COMMON_ARGS
        python $FILE --experiment_id blocks_nsrt_learning_$NUM_TRAIN_TASKS --approach nsrt_learning --excluded_predicates goals --env blocks $COMMON_ARGS
        python $FILE --experiment_id painting_nsrt_learning_$NUM_TRAIN_TASKS --approach nsrt_learning --excluded_predicates goals --env painting $COMMON_ARGS
        python $FILE --experiment_id tools_nsrt_learning_$NUM_TRAIN_TASKS  --approach nsrt_learning --excluded_predicates goals --env tools $COMMON_ARGS

        # Grammar search
        python $FILE --experiment_id cover_invent_allexclude_$NUM_TRAIN_TASKS --approach grammar_search_invention --excluded_predicates all --env cover $COMMON_ARGS
        python $FILE --experiment_id blocks_invent_allexclude_$NUM_TRAIN_TASKS --approach grammar_search_invention --excluded_predicates all --env blocks $COMMON_ARGS
        python $FILE --experiment_id painting_invent_allexclude_$NUM_TRAIN_TASKS --approach grammar_search_invention --excluded_predicates all --env painting $COMMON_ARGS
        python $FILE --experiment_id tools_invent_allexclude_$NUM_TRAIN_TASKS  --approach grammar_search_invention --excluded_predicates all --env tools $COMMON_ARGS

    done
done
