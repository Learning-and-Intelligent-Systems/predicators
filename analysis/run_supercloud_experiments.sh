#!/bin/bash

START_SEED=456
NUM_SEEDS=5
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # cover
    python $FILE --env cover --approach oracle --seed $SEED
    python $FILE --env cover --approach nsrt_learning --seed $SEED
    python $FILE --env cover --approach grammar_search_invention --seed $SEED
    python $FILE --env cover --approach grammar_search_invention --seed $SEED --excluded_predicates all

    # blocks
    python $FILE --env blocks --approach oracle --seed $SEED
    python $FILE --env blocks --approach nsrt_learning --seed $SEED
    python $FILE --env blocks --approach grammar_search_invention --seed $SEED
    python $FILE --env blocks --approach grammar_search_invention --seed $SEED --excluded_predicates all

    # painting
    python $FILE --env painting --approach oracle --seed $SEED
    python $FILE --env painting --approach nsrt_learning --seed $SEED
    python $FILE --env painting --approach grammar_search_invention --seed $SEED --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0
    python $FILE --env painting --approach grammar_search_invention --seed $SEED --excluded_predicates all --grammar_search_grammar_includes_givens 0 --grammar_search_grammar_includes_foralls 0

    # repeated_nextto
    python $FILE --env repeated_nextto --approach oracle --seed $SEED
    python $FILE --env repeated_nextto --approach nsrt_learning --seed $SEED

    # cluttered_table
    python $FILE --env cluttered_table --approach oracle --seed $SEED
    python $FILE --env cluttered_table --approach nsrt_learning --seed $SEED
done
