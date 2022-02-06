#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"
ALL_NUM_TRAIN_TASKS=(
    "50"
    # "100"
    # "150"
    # "200"
)

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do
        COMMON_ARGS="--seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"

        # cover
        python $FILE --experiment_id cover_oracle_$NUM_TRAIN_TASKS --env cover --approach oracle $COMMON_ARGS
        python $FILE --experiment_id cover_noinvent_noexclude_$NUM_TRAIN_TASKS --env cover --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id cover_noinvent_allexclude_$NUM_TRAIN_TASKS --env cover --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id cover_invent_noexclude_$NUM_TRAIN_TASKS --env cover --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id cover_invent_allexclude_$NUM_TRAIN_TASKS --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

        # blocks
        python $FILE --experiment_id blocks_oracle_$NUM_TRAIN_TASKS --env blocks --approach oracle $COMMON_ARGS
        python $FILE --experiment_id blocks_noinvent_noexclude_$NUM_TRAIN_TASKS --env blocks --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id blocks_noinvent_allexclude_$NUM_TRAIN_TASKS --env blocks --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id blocks_invent_noexclude_$NUM_TRAIN_TASKS --env blocks --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id blocks_invent_allexclude_$NUM_TRAIN_TASKS --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

        # painting
        python $FILE --experiment_id painting_oracle_$NUM_TRAIN_TASKS --env painting --approach oracle $COMMON_ARGS
        python $FILE --experiment_id painting_noinvent_noexclude_$NUM_TRAIN_TASKS --env painting --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id painting_noinvent_allexclude_$NUM_TRAIN_TASKS --env painting --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id painting_invent_noexclude_$NUM_TRAIN_TASKS --env painting --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id painting_invent_allexclude_$NUM_TRAIN_TASKS --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

        # cluttered_table
        python $FILE --experiment_id ctable_oracle_$NUM_TRAIN_TASKS --env cluttered_table --approach oracle $COMMON_ARGS
        python $FILE --experiment_id ctable_noinvent_noexclude_$NUM_TRAIN_TASKS --env cluttered_table --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id ctable_noinvent_allexclude_$NUM_TRAIN_TASKS --env cluttered_table --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id ctable_invent_noexclude_$NUM_TRAIN_TASKS --env cluttered_table --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id ctable_invent_allexclude_$NUM_TRAIN_TASKS --env cluttered_table --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

        # tools
        python $FILE --experiment_id tools_oracle_$NUM_TRAIN_TASKS --env tools --approach oracle $COMMON_ARGS
        python $FILE --experiment_id tools_noinvent_noexclude_$NUM_TRAIN_TASKS --env tools --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id tools_noinvent_allexclude_$NUM_TRAIN_TASKS --env tools --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id tools_invent_noexclude_$NUM_TRAIN_TASKS --env tools --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id tools_invent_allexclude_$NUM_TRAIN_TASKS --env tools --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS
    done
done
