#!/bin/bash

START_SEED=456
NUM_SEEDS=5
MAX_TRANSITIONS=1000  # want this to be stop signal
CYCLES=100  # way too many cycles
REQUESTS=10
MAX_STEPS=3
MIN_DATA=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # section kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_mindata_section_kid --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial --online_learning_max_transitions $MAX_TRANSITIONS --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA
    # ours, entropy
    for THRESH in $(seq 0.1 0.1 0.2); do
        python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_mindata_entropy_$THRESH --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function entropy --online_learning_max_transitions $MAX_TRANSITIONS --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA
    done
    # ours, BALD
    for THRESH in $(seq 0.01 0.01 0.02); do
        python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_mindata_BALD_$THRESH --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function BALD --online_learning_max_transitions $MAX_TRANSITIONS --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA
    done
    # silent kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_mindata_silent_kid --interactive_query_policy threshold --interactive_score_threshold 1.0 --interactive_score_function entropy --online_learning_max_transitions $MAX_TRANSITIONS --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA
done
