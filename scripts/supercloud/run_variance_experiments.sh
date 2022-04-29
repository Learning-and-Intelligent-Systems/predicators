#!/bin/bash

ENV="cover"
START_SEED=456
NUM_SEEDS=5
MAX_TRANSITIONS=1000  # want this to be stop signal
CYCLES=100  # way too many cycles
REQUESTS=10
MAX_STEPS=3
MIN_DATA=10
MAX_ITR=100000
START_THRESH="0.01"
INCREMENT="0.01"
END_THRESH="0.1"
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    COMMON_ARGS="--env $ENV --approach interactive_learning --seed $SEED \
    --excluded_predicates Covers,Holding --num_online_learning_cycles $CYCLES \
    --online_learning_max_transitions $MAX_TRANSITIONS \
    --interactive_num_requests_per_cycle $REQUESTS \
    --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA \
    --sampler_disable_classifier True --mlp_classifier_balance_data False \
    --predicate_mlp_classifier_max_itr $MAX_ITR"

    # Main approach
    python $FILE $COMMON_ARGS --experiment_id main --interactive_score_function entropy --interactive_score_threshold 0.01

    for THRESH in $(seq $START_THRESH $INCREMENT $END_THRESH); do
        python $FILE $COMMON_ARGS --experiment_id variance_$THRESH --interactive_score_function variance --interactive_score_threshold $THRESH
    done

done
