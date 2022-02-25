#!/bin/bash

START_SEED=456
NUM_SEEDS=10
CYCLES=100
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # section kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --experiment_id section_kid --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request 3
    # ours
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --experiment_id entropy_0.3 --interactive_query_policy threshold --interactive_score_threshold 0.3 --interactive_score_function entropy --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request 3
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --experiment_id BALD_0.05 --interactive_query_policy threshold --interactive_score_threshold 0.05 --interactive_score_function BALD --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request 3
    # silent kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers --experiment_id silent_kid --interactive_query_policy threshold --interactive_score_threshold 1.0 --interactive_score_function entropy --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request 3
done
