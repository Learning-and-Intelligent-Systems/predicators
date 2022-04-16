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

    COMMON_ARGS="--env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding \
    --interactive_query_policy threshold --interactive_score_function entropy --interactive_score_threshold 0.05 \
    --num_online_learning_cycles $CYCLES --online_learning_max_transitions $MAX_TRANSITIONS \
    --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS \
    --min_data_for_nsrt $MIN_DATA --sampler_disable_classifier True --mlp_classifier_balance_data False"

    # glib
    # python $FILE $COMMON_ARGS --experiment_id glib --interactive_action_strategy glib
    # greedy lookahead
    # python $FILE $COMMON_ARGS --experiment_id greedy_lookahead --interactive_action_strategy greedy_lookahead
    # glib + 10k max iters
    python $FILE $COMMON_ARGS --experiment_id glib_10k --interactive_action_strategy glib --predicate_mlp_classifier_max_itr 10000

done
