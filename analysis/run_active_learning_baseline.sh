#!/bin/bash

START_SEED=456
NUM_SEEDS=10
CYCLES=100
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial --num_online_learning_cycles $CYCLES --max_initial_demos 1
done
