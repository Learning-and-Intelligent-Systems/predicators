#!/bin/bash

START_SEED=456
NUM_SEEDS=10
MAX_TRANSITIONS=1000  # want this to be stop signal
CYCLES=100  # way too many cycles
REQUESTS=10
MAX_STEPS=3
MIN_DATA=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    COMMON_ARGS="--env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --online_learning_max_transitions $MAX_TRANSITIONS --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA --sampler_disable_classifier True --mlp_classifier_balance_data False"

    # section kid
    python $FILE $COMMON_ARGS --experiment_id excludeall_section_kid --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial
    # ours, entropy
    python $FILE $COMMON_ARGS --experiment_id excludeall_entropy_0.1 --interactive_query_policy threshold --interactive_score_threshold 0.1 --interactive_score_function entropy
    # ours, BALD
    python $FILE $COMMON_ARGS --experiment_id excludeall_BALD_0.01 --interactive_query_policy threshold --interactive_score_threshold 0.01 --interactive_score_function BALD
    # silent kid
    python $FILE $COMMON_ARGS --experiment_id excludeall_silent_kid --interactive_query_policy threshold --interactive_score_threshold 1.0 --interactive_score_function trivial

done
