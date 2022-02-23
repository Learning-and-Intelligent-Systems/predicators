#!/bin/bash

START_SEED=456
NUM_SEEDS=5
START_THRESH=0.1
INCREMENT=0.1
END_THRESH=0.9
CYCLES=100
FILE="analysis/submit.py"

# to find an appropriate threshold
for SEED in $(seq $START_THRESH $((NUM_SEEDS+START_THRESH-1))); do
    for THRESH in $(seq $START_THRESH $INCREMENT $END_THRESH); do
        python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function entropy --num_online_learning_cycles $CYCLES
        python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function BALD --num_online_learning_cycles $CYCLES
    done
done
