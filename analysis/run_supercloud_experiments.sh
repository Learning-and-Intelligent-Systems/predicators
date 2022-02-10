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
    python $FILE --experiment_id cover_regrasp_oracle --env cover_regrasp --approach oracle --seed $SEED
    python $FILE --experiment_id blocks_oracle --env blocks --approach oracle --seed $SEED
    python $FILE --experiment_id painting_oracle --env painting --approach oracle --seed $SEED
    python $FILE --experiment_id tools_oracle --env tools --approach oracle --seed $SEED
    python $FILE --experiment_id repeated_nextto_oracle --env repeated_nextto --approach oracle --seed $SEED

    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do
        COMMON_ARGS="--seed $SEED --num_train_tasks $NUM_TRAIN_TASKS"

        # cover_regrasp
        python $FILE --experiment_id cover_regrasp_noinvent_noexclude_$NUM_TRAIN_TASKS --env cover_regrasp --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id cover_regrasp_noinvent_allexclude_$NUM_TRAIN_TASKS --env cover_regrasp --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id cover_regrasp_invent_noexclude_$NUM_TRAIN_TASKS --env cover_regrasp --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id cover_regrasp_invent_allexclude_$NUM_TRAIN_TASKS --env cover_regrasp --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

        # blocks
        python $FILE --experiment_id blocks_noinvent_noexclude_$NUM_TRAIN_TASKS --env blocks --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id blocks_noinvent_allexclude_$NUM_TRAIN_TASKS --env blocks --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id blocks_invent_noexclude_$NUM_TRAIN_TASKS --env blocks --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id blocks_invent_allexclude_$NUM_TRAIN_TASKS --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

        # painting
        python $FILE --experiment_id painting_noinvent_noexclude_$NUM_TRAIN_TASKS --env painting --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id painting_noinvent_allexclude_$NUM_TRAIN_TASKS --env painting --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id painting_invent_noexclude_$NUM_TRAIN_TASKS --env painting --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id painting_invent_allexclude_$NUM_TRAIN_TASKS --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

        # tools
        python $FILE --experiment_id tools_noinvent_noexclude_$NUM_TRAIN_TASKS --env tools --approach nsrt_learning $COMMON_ARGS
        python $FILE --experiment_id tools_noinvent_allexclude_$NUM_TRAIN_TASKS --env tools --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
        python $FILE --experiment_id tools_invent_noexclude_$NUM_TRAIN_TASKS --env tools --approach grammar_search_invention $COMMON_ARGS
        python $FILE --experiment_id tools_invent_allexclude_$NUM_TRAIN_TASKS --env tools --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

        # repeated_nextto
        python $FILE --experiment_id repeated_nextto_noinvent_noexclude_$NUM_TRAIN_TASKS --env repeated_nextto --approach nsrt_learning --learn_side_predicates True $COMMON_ARGS
        python $FILE --experiment_id repeated_nextto_noinvent_allexclude_$NUM_TRAIN_TASKS --env repeated_nextto --approach nsrt_learning --learn_side_predicates True --excluded_predicates all $COMMON_ARGS
    done
done
