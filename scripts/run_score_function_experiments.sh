#!/bin/bash

START_SEED=456
NUM_SEEDS=10
CYCLES=100
FILE="scripts/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # section kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial --num_online_learning_cycles $CYCLES
    # ours
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --interactive_query_policy threshold --interactive_score_threshold 0.5 --interactive_score_function entropy --num_online_learning_cycles $CYCLES
    # silent kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --interactive_query_policy threshold --interactive_score_threshold 1.0 --interactive_score_function entropy --num_online_learning_cycles $CYCLES
done
