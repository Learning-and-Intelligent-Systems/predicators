#!/bin/bash

START_SEED=456
NUM_SEEDS=1
CYCLES=100
MAX_STEPS=3
FILE="analysis/submit.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # section kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_section_kid --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request $MAX_STEPS --load_approach --make_videos --make_failure_videos
    # ours, entropy
    # for THRESH in $(seq 0.1 0.1 0.3); do
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_entropy_0.1 --interactive_query_policy threshold --interactive_score_threshold 0.1 --interactive_score_function entropy --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request $MAX_STEPS --load_approach --make_videos --make_failure_videos
    # done
    # ours, BALD
    # for THRESH in $(seq 0.01 0.01 0.05); do
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_BALD_0.01 --interactive_query_policy threshold --interactive_score_threshold 0.01 --interactive_score_function BALD --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request $MAX_STEPS --load_approach --make_videos --make_failure_videos
    # done
    # silent kid
    python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_silent_kid --interactive_query_policy threshold --interactive_score_threshold 1.0 --interactive_score_function entropy --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request $MAX_STEPS --load_approach --make_videos --make_failure_videos
done
