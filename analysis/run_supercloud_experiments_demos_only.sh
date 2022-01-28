#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    COMMON_ARGS="--seed $SEED --offline_data_method demo --grammar_search_expected_nodes_allow_noops True"

    ## 50 max predicates and training wheel painting grammar

    # cover
    python $FILE --experiment_id cover_nsrt_learning_demo --env cover --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id cover_none_excluded_demo --env cover --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id cover_all_excluded_demo --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # blocks
    python $FILE --experiment_id blocks_nsrt_learning_demo --env blocks --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id blocks_none_excluded_demo --env blocks --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id blocks_all_excluded_demo --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting
    python $FILE --experiment_id painting_nsrt_learning_demo --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id painting_none_excluded_demo --env painting --approach grammar_search_invention --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS
    python $FILE --experiment_id painting_all_excluded_demo --env painting --approach grammar_search_invention --excluded_predicates all --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS

    # painting with box lid always open
    python $FILE --experiment_id painting_always_open_nsrt_learning_demo --painting_lid_open_prob 1.0 --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_none_excluded_demo --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_all_excluded_demo --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS

    ## 200 max predicates and full painting grammar

    # cover
    python $FILE --experiment_id cover_nsrt_learning_demo_large --env cover --approach nsrt_learning $COMMON_ARGS --grammar_search_max_predicates 200
    python $FILE --experiment_id cover_none_excluded_demo_large --env cover --approach grammar_search_invention $COMMON_ARGS --grammar_search_max_predicates 200
    python $FILE --experiment_id cover_all_excluded_demo_large --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_max_predicates 200

    # blocks
    python $FILE --experiment_id blocks_nsrt_learning_demo_large --env blocks --approach nsrt_learning $COMMON_ARGS --grammar_search_max_predicates 200
    python $FILE --experiment_id blocks_none_excluded_demo_large --env blocks --approach grammar_search_invention $COMMON_ARGS --grammar_search_max_predicates 200
    python $FILE --experiment_id blocks_all_excluded_demo_large --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS --grammar_search_max_predicates 200

    # painting
    python $FILE --experiment_id painting_nsrt_learning_demo_large --env painting --approach nsrt_learning $COMMON_ARGS --grammar_search_max_predicates 200
    python $FILE --experiment_id painting_none_excluded_demo_large --env painting --approach grammar_search_invention  $COMMON_ARGS --grammar_search_max_predicates 200
    python $FILE --experiment_id painting_all_excluded_demo_large --env painting --approach grammar_search_invention --excluded_predicates all  $COMMON_ARGS --grammar_search_max_predicates 200

    # painting with box lid always open
    python $FILE --experiment_id painting_always_open_nsrt_learning_demo_large --painting_lid_open_prob 1.0 --env painting --approach nsrt_learning $COMMON_ARGS --grammar_search_max_predicates 200
    python $FILE --experiment_id painting_always_open_none_excluded_demo_large --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention  $COMMON_ARGS --grammar_search_max_predicates 200
    python $FILE --experiment_id painting_always_open_all_excluded_demo_large --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all  $COMMON_ARGS --grammar_search_max_predicates 200

done
