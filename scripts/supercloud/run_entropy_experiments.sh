#!/bin/bash

START_THRESH=0.1
INCREMENT=0.1
END_THRESH=0.9
CYCLES=100
FILE="scripts/supercloud/submit_supercloud_job.py"

# to find an appropriate threshold
for THRESH in $(seq $START_THRESH $INCREMENT $END_THRESH); do
    python $FILE --env cover --approach interactive_learning --excluded_predicates Covers --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function entropy --num_online_learning_cycles $CYCLES
    python $FILE --env cover --approach interactive_learning --excluded_predicates Covers --interactive_query_policy threshold --interactive_score_threshold $THRESH --interactive_score_function BALD --num_online_learning_cycles $CYCLES
done
