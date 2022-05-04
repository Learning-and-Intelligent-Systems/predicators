#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    python $FILE --experiment_id pretend_play_blocks --env play_blocks \
    --approach pretend_play --seed $SEED --max_initial_demos 1 \
    --num_online_learning_cycles 10000 --online_learning_max_transitions 1000 \
    --allow_interaction_in_demo_tasks False --max_num_steps_interaction_request 10 \
    --interactive_action_strategy random
done
