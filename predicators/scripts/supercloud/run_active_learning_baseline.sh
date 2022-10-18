#!/bin/bash

CYCLES=100
FILE="scripts/supercloud/submit_supercloud_job.py"

python $FILE --env cover --approach interactive_learning --excluded_predicates Covers --interactive_query_policy nonstrict_best_seen --interactive_score_function trivial --num_online_learning_cycles $CYCLES --max_initial_demos 1 --explorer greedy_lookahead
