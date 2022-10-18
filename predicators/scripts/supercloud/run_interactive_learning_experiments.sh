#!/bin/bash

ENV="cover"
MAX_TRANSITIONS=1000  # want this to be stop signal
CYCLES=100  # way too many cycles
REQUESTS=10
MAX_STEPS=3
MIN_DATA=10
THRESH="0.05"
MAX_ITR=100000
QUERY_PROB="0.03"  # so random kid queries about the same amount as main
FILE="scripts/supercloud/submit_supercloud_job.py"

COMMON_ARGS="--env $ENV --approach interactive_learning \
    --excluded_predicates Covers,Holding --interactive_score_function entropy \
    --interactive_score_threshold $THRESH --num_online_learning_cycles $CYCLES \
    --online_learning_max_transitions $MAX_TRANSITIONS \
    --interactive_num_requests_per_cycle $REQUESTS \
    --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA \
    --sampler_disable_classifier True --mlp_classifier_balance_data False \
    --predicate_mlp_classifier_max_itr $MAX_ITR"

## Main approach
python $FILE $COMMON_ARGS --experiment_id main --explorer greedy_lookahead

## MLP baseline
python $FILE $COMMON_ARGS --experiment_id main_mlp --explorer greedy_lookahead --interactive_num_ensemble_members 1

## Query baselines
# Section kid
python $FILE $COMMON_ARGS --experiment_id section_kid --explorer greedy_lookahead --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial
# Silent kid
python $FILE $COMMON_ARGS --experiment_id silent_kid --explorer greedy_lookahead  --interactive_query_policy threshold --interactive_score_threshold 1.0 --interactive_score_function trivial
# Random kid
python $FILE $COMMON_ARGS --experiment_id random_kid --explorer greedy_lookahead --interactive_query_policy random --interactive_random_query_prob $QUERY_PROB --interactive_score_function trivial

## Action baselines
# GLIB
python $FILE $COMMON_ARGS --experiment_id glib --explorer glib
# Random actions
python $FILE $COMMON_ARGS --experiment_id random_actions --explorer random_options
# No actions
python $FILE $COMMON_ARGS --experiment_id no_actions --explorer no_explore
