#!/bin/bash

SEED=456
MAX_TRANSITIONS=1000  # want this to be stop signal
CYCLES=100  # way too many cycles
REQUESTS=10
MAX_STEPS=3
MIN_DATA=10
FILE="scripts/supercloud/submit_supercloud_job.py"

# section kid
# python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_section_kid --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request $MAX_STEPS --load_approach --make_interaction_videos

# ours, entropy
# python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_mindata_entropy_0.1 --interactive_query_policy threshold --interactive_score_threshold 0.1 --interactive_score_function entropy --online_learning_max_transitions $MAX_TRANSITIONS --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA --load_approach --make_interaction_videos
python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_entropy_0.1 --interactive_query_policy threshold --interactive_score_threshold 0.1 --interactive_score_function entropy --num_online_learning_cycles 20 --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --load_approach --make_interaction_videos

# ours, BALD
# python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_mindata_BALD_0.01 --interactive_query_policy threshold --interactive_score_threshold 0.01 --interactive_score_function BALD --online_learning_max_transitions $MAX_TRANSITIONS --num_online_learning_cycles $CYCLES --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --min_data_for_nsrt $MIN_DATA --load_approach --make_interaction_videos
python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_BALD_0.01 --interactive_query_policy threshold --interactive_score_threshold 0.01 --interactive_score_function BALD --num_online_learning_cycles 20 --interactive_num_requests_per_cycle $REQUESTS --max_num_steps_interaction_request $MAX_STEPS --load_approach --make_interaction_videos

# silent kid
# python $FILE --env cover --approach interactive_learning --seed $SEED --excluded_predicates Covers,Holding --experiment_id excludeall_silent_kid --interactive_query_policy threshold --interactive_score_threshold 1.0 --interactive_score_function entropy --num_online_learning_cycles $CYCLES --max_num_steps_interaction_request $MAX_STEPS --load_approach --make_interaction_videos
