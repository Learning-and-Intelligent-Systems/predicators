#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    COMMON_ARGS="--seed $SEED"

    # cover
    python $FILE --experiment_id cover_oracle --env cover --approach oracle $COMMON_ARGS
    python $FILE --experiment_id cover_nsrt_learning --env cover --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id cover_none_excluded --env cover --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id cover_all_excluded --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # blocks
    python $FILE --experiment_id blocks_oracle --env blocks --approach oracle $COMMON_ARGS
    python $FILE --experiment_id blocks_nsrt_learning --env blocks --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id blocks_none_excluded --env blocks --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id blocks_all_excluded --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting
    python $FILE --experiment_id painting_oracle --env painting --approach oracle $COMMON_ARGS
    python $FILE --experiment_id painting_nsrt_learning --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id painting_none_excluded --env painting --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id painting_all_excluded --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting with lid always open
    python $FILE --experiment_id painting_always_open_oracle --painting_lid_open_prob 1.0 --env painting --approach oracle $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_nsrt_learning --painting_lid_open_prob 1.0 --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_none_excluded --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_all_excluded --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # repeated_nextto
    python $FILE --experiment_id repeated_nextto_oracle --env repeated_nextto --approach oracle $COMMON_ARGS

    # cluttered_table
    python $FILE --experiment_id cluttered_table_oracle --env cluttered_table --approach oracle $COMMON_ARGS
    python $FILE --experiment_id cluttered_table_nsrt_learning --env cluttered_table --approach nsrt_learning $COMMON_ARGS
done
