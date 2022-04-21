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
    --online_learning_max_transitions $MAX_TRANSITIONS --num_online_learning_cycles $CYCLES \
    --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS \
    --min_data_for_nsrt $MIN_DATA --sampler_disable_classifier True --mlp_classifier_balance_data False"

    # glib
    python $FILE $COMMON_ARGS --experiment_id glib_default --interactive_action_strategy glib --interactive_query_policy threshold --interactive_score_function entropy --interactive_score_threshold 0.05 --predicate_mlp_classifier_init default --predicate_mlp_classifier_init_param None
    python $FILE $COMMON_ARGS --experiment_id glib_normal_1 --interactive_action_strategy glib --interactive_query_policy threshold --interactive_score_function entropy --interactive_score_threshold 0.05 --predicate_mlp_classifier_init normal --predicate_mlp_classifier_init_param 1.0
    python $FILE $COMMON_ARGS --experiment_id glib_uniform_3 --interactive_action_strategy glib --interactive_query_policy threshold --interactive_score_function entropy --interactive_score_threshold 0.05 --predicate_mlp_classifier_init uniform --predicate_mlp_classifier_init_param 3.0

    # greedy
    python $FILE $COMMON_ARGS --experiment_id greedy_normal_1 --interactive_action_strategy greedy_lookahead --interactive_query_policy threshold --interactive_score_function entropy --interactive_score_threshold 0.05 --predicate_mlp_classifier_init normal --predicate_mlp_classifier_init_param 1.0
    python $FILE $COMMON_ARGS --experiment_id greedy_uniform_3 --interactive_action_strategy greedy_lookahead --interactive_query_policy threshold --interactive_score_function entropy --interactive_score_threshold 0.05 --predicate_mlp_classifier_init uniform --predicate_mlp_classifier_init_param 3.0
done
