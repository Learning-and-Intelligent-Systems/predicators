#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    COMMON_ARGS="--seed $SEED"

    # cover
    python $FILE --experiment_id cover_oracle --env cover --approach oracle $COMMON_ARGS
    python $FILE --experiment_id cover_noinvent_noexclude --env cover --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id cover_noinvent_allexclude --env cover --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
    python $FILE --experiment_id cover_invent_noexclude --env cover --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id cover_invent_allexclude --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # blocks
    python $FILE --experiment_id blocks_oracle --env blocks --approach oracle $COMMON_ARGS
    python $FILE --experiment_id blocks_noinvent_noexclude --env blocks --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id blocks_noinvent_allexclude --env blocks --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
    python $FILE --experiment_id blocks_invent_noexclude --env blocks --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id blocks_invent_allexclude --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting
    python $FILE --experiment_id painting_oracle --env painting --approach oracle $COMMON_ARGS
    python $FILE --experiment_id painting_noinvent_noexclude --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id painting_noinvent_allexclude --env painting --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
    python $FILE --experiment_id painting_invent_noexclude --env painting --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id painting_invent_allexclude --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting with lid always open
    python $FILE --experiment_id painting_always_open_oracle --painting_lid_open_prob 1.0 --env painting --approach oracle $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_noinvent_noexclude --painting_lid_open_prob 1.0 --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_noinvent_allexclude --painting_lid_open_prob 1.0 --env painting --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_invent_noexclude --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_invent_allexclude --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # cluttered_table
    python $FILE --experiment_id ctable_oracle --env cluttered_table --approach oracle $COMMON_ARGS
    python $FILE --experiment_id ctable_noinvent_noexclude --env cluttered_table --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id ctable_noinvent_allexclude --env cluttered_table --approach nsrt_learning --excluded_predicates all $COMMON_ARGS
    python $FILE --experiment_id ctable_invent_noexclude --env cluttered_table --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id ctable_invent_allexclude --env cluttered_table --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS
done
