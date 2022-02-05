#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    COMMON_ARGS="--seed $SEED  --cover_initial_holding_prob 0.0 --painting_initial_holding_prob 0.0"

    # cover regrasp
    python $FILE --experiment_id cover_regrasp_targeted_noexclude --env cover_regrasp --approach grammar_search_invention --grammar_search_interactive_strategy targeted $COMMON_ARGS
    python $FILE --experiment_id cover_regrasp_targeted_allexclude --env cover_regrasp --approach grammar_search_invention --grammar_search_interactive_strategy targeted --excluded_predicates all $COMMON_ARGS
    python $FILE --experiment_id cover_regrasp_naive_noexclude --env cover_regrasp --approach grammar_search_invention --grammar_search_interactive_strategy naive $COMMON_ARGS
    python $FILE --experiment_id cover_regrasp_naive_allexclude --env cover_regrasp --approach grammar_search_invention --grammar_search_interactive_strategy naive --excluded_predicates all $COMMON_ARGS

    # blocks
    python $FILE --experiment_id blocks_targeted_noexclude --env blocks --approach grammar_search_invention --grammar_search_interactive_strategy targeted $COMMON_ARGS
    python $FILE --experiment_id blocks_targeted_allexclude --env blocks --approach grammar_search_invention --grammar_search_interactive_strategy targeted --excluded_predicates all $COMMON_ARGS
    python $FILE --experiment_id blocks_naive_noexclude --env blocks --approach grammar_search_invention --grammar_search_interactive_strategy naive $COMMON_ARGS
    python $FILE --experiment_id blocks_naive_allexclude --env blocks --approach grammar_search_invention --grammar_search_interactive_strategy naive --excluded_predicates all $COMMON_ARGS

    # painting
    python $FILE --experiment_id painting_targeted_noexclude --env painting --approach grammar_search_invention --grammar_search_interactive_strategy targeted $COMMON_ARGS
    python $FILE --experiment_id painting_targeted_allexclude --env painting --approach grammar_search_invention --grammar_search_interactive_strategy targeted --excluded_predicates all $COMMON_ARGS
    python $FILE --experiment_id painting_naive_noexclude --env painting --approach grammar_search_invention --grammar_search_interactive_strategy naive $COMMON_ARGS
    python $FILE --experiment_id painting_naive_allexclude --env painting --approach grammar_search_invention --grammar_search_interactive_strategy naive --excluded_predicates all $COMMON_ARGS

done
