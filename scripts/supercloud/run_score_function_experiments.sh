#!/bin/bash

START_SEED=456
NUM_SEEDS=5
CYCLES=20
REQUESTS=10
MAX_STEPS=3
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # section kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_section_kid --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS
    # ours, entropy
    for THRESH in $(seq 0.1 0.1 0.3); do
        python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_entropy_$THRESH --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function entropy --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS
    done
    # ours, BALD
    for THRESH in $(seq 0.01 0.01 0.05); do
        python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_BALD_$THRESH --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function BALD --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS
    done
    # silent kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_silent_kid --interactive_query_policy threshold --interactive_score_threshold 1.0 --interactive_score_function entropy --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS
done
