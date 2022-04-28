#!/bin/bash

ENV="cover"
START_SEED=456
NUM_SEEDS=5
MAX_TRANSITIONS=1000  # want this to be stop signal
CYCLES=100  # way too many cycles
REQUESTS=10
MAX_STEPS=3
MIN_DATA=10
THRESH="0.05"
MAX_ITR=100000
START_PROB="0.1"
INCREMENT="0.1"
END_PROB="0.9"
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    COMMON_ARGS="--env $ENV --approach interactive_learning --seed $SEED \
    --excluded_predicates Covers,Holding --interactive_score_function entropy \
    --interactive_score_threshold $THRESH --num_online_learning_cycles $CYCLES \
    --online_learning_max_transitions $MAX_TRANSITIONS \
    --interactive_num_requests_per_cycle $REQUESTS \
    --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA \
    --sampler_disable_classifier True --mlp_classifier_balance_data False \
    --predicate_mlp_classifier_max_itr $MAX_ITR"

    for PROB in $(seq $START_PROB $INCREMENT $END_PROB); do

        # Random kid with different query probabilities
        python $FILE $COMMON_ARGS --experiment_id random_kid_$PROB --interactive_query_policy random --interactive_random_query_prob $PROB --interactive_score_function trivial

    done

done
