#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    COMMON_ARGS="--remake_data --num_train_tasks 50 --grammar_search_score_function expected_nodes --seed $SEED"

    ## demo + replay

    # cover
    python $FILE --experiment_id cover_nsrt_learning --env cover --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id cover_none_excluded --env cover --approach grammar_search_invention --cover_initial_holding_prob 0.75 $COMMON_ARGS
    python $FILE --experiment_id cover_all_excluded --env cover --approach grammar_search_invention --excluded_predicates all --cover_initial_holding_prob 0.75 $COMMON_ARGS

    # blocks
    python $FILE --experiment_id blocks_nsrt_learning --env blocks --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id blocks_none_excluded --env blocks --approach grammar_search_invention $COMMON_ARGS
    python $FILE --experiment_id blocks_all_excluded --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting
    python $FILE --experiment_id painting_nsrt_learning --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id painting_none_excluded --env painting --approach grammar_search_invention --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS
    python $FILE --experiment_id painting_all_excluded --env painting --approach grammar_search_invention --excluded_predicates all --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS

    # painting with box lid always open
    python $FILE --experiment_id painting_always_open_nsrt_learning --painting_lid_open_prob 1.0 --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_none_excluded --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_all_excluded --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --excluded_predicates all --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS

    ## demo only

    # cover
    python $FILE --experiment_id cover_nsrt_learning_demo --env cover --approach nsrt_learning --offline_data_method demo $COMMON_ARGS
    python $FILE --experiment_id cover_none_excluded_demo --env cover --approach grammar_search_invention --offline_data_method demo --cover_initial_holding_prob 0.75 $COMMON_ARGS
    python $FILE --experiment_id cover_all_excluded_demo --env cover --approach grammar_search_invention --offline_data_method demo --excluded_predicates all --cover_initial_holding_prob 0.75 $COMMON_ARGS

    # blocks
    python $FILE --experiment_id blocks_nsrt_learning_demo --env blocks --approach nsrt_learning --offline_data_method demo $COMMON_ARGS
    python $FILE --experiment_id blocks_none_excluded_demo --env blocks --approach grammar_search_invention --offline_data_method demo $COMMON_ARGS
    python $FILE --experiment_id blocks_all_excluded_demo --env blocks --approach grammar_search_invention --offline_data_method demo --excluded_predicates all $COMMON_ARGS

    # painting
    python $FILE --experiment_id painting_nsrt_learning_demo --env painting --approach nsrt_learning --offline_data_method demo $COMMON_ARGS
    python $FILE --experiment_id painting_none_excluded_demo --env painting --approach grammar_search_invention --offline_data_method demo --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS
    python $FILE --experiment_id painting_all_excluded_demo --env painting --approach grammar_search_invention --offline_data_method demo --excluded_predicates all --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS

    # painting with box lid always open
    python $FILE --experiment_id painting_always_open_nsrt_learning_demo --painting_lid_open_prob 1.0 --env painting --approach nsrt_learning --offline_data_method demo $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_none_excluded_demo --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --offline_data_method demo --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS
    python $FILE --experiment_id painting_always_open_all_excluded_demo --painting_lid_open_prob 1.0 --env painting --approach grammar_search_invention --offline_data_method demo --excluded_predicates all --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS

done
