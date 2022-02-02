#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    COMMON_ARGS="--seed $SEED --grammar_search_expected_nodes_max_skeletons 1"

    # cover_regrasp
    python $FILE --experiment_id cover_regrasp_invent_noexclude --env cover_regrasp --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id cover_regrasp_invent_allexclude --env cover_regrasp --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # cover_regrasp with original grammar_search_expected_nodes_max_skeletons (COMMON_ARGS excluded!)
    python $FILE --experiment_id cover_regrasp_invent_noexclude_multiskeleton --env cover_regrasp --approach grammar_search_invention --seed $SEED
    python $FILE --experiment_id cover_regrasp_invent_allexclude_multiskeleton --env cover_regrasp --approach grammar_search_invention --excluded_predicates all --seed $SEED

    # cover
    python $FILE --experiment_id cover_invent_noexclude --env cover --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id cover_invent_allexclude --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # blocks
    python $FILE --experiment_id blocks_invent_noexclude --env blocks --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id blocks_invent_allexclude --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting
    python $FILE --experiment_id painting_invent_noexclude --env painting --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id painting_invent_allexclude --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting with lid always open
    python $FILE --experiment_id painting_always_open_invent_noexclude --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_invent_allexclude --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # cluttered_table
    python $FILE --experiment_id ctable_invent_noexclude --env cluttered_table --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id ctable_invent_allexclude --env cluttered_table --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS
done
