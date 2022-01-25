#!/bin/bash

START_SEED=456
NUM_SEEDS=5
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    COMMON_ARGS="--remake_data --seed $SEED"

    # cover
    python $FILE --env cover --approach oracle $COMMON_ARGS
    python $FILE --env cover --approach nsrt_learning $COMMON_ARGS
    python $FILE --env cover --approach grammar_search_invention $COMMON_ARGS
    python $FILE --env cover --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # blocks
    python $FILE --env blocks --approach oracle $COMMON_ARGS
    python $FILE --env blocks --approach nsrt_learning $COMMON_ARGS
    python $FILE --env blocks --approach grammar_search_invention $COMMON_ARGS
    python $FILE --env blocks --approach grammar_search_invention --excluded_predicates all $COMMON_ARGS

    # painting
    python $FILE --env painting --approach oracle $COMMON_ARGS
    python $FILE --env painting --approach nsrt_learning $COMMON_ARGS
    python $FILE --env painting --approach grammar_search_invention --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS
    python $FILE --env painting --approach grammar_search_invention --excluded_predicates all --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0 $COMMON_ARGS

    # repeated_nextto
    python $FILE --env repeated_nextto --approach oracle $COMMON_ARGS

    # cluttered_table
    python $FILE --env cluttered_table --approach oracle $COMMON_ARGS
    python $FILE --env cluttered_table --approach nsrt_learning $COMMON_ARGS
done
