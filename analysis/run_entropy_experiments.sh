#!/bin/bash

START_SEED=456
NUM_SEEDS=5
START_THRESH=0
INCREMENT=0.1
END_THRESH=0.6
CYCLES=100
FILE="analysis/submit.py"

# to find an appropriate threshold
for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    for THRESH in $(seq $START_THRESH $INCREMENT $END_THRESH); do
        python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function entropy --experiment_id entropy_$THRESH --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request 3
    done
    for THRESH in $(seq 0.01 0.01 0.1); do
        python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function BALD --experiment_id BALD_$THRESH --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request 3
    done
done
