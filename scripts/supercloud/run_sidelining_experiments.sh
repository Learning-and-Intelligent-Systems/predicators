#!/bin/bash

START_SEED=456
NUM_SEEDS=100
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # repeated_nextto
    # python $FILE --experiment_id rnt_oracle --env repeated_nextto --approach oracle --seed $SEED --num_train_tasks 0
    # python $FILE --experiment_id rnt_no_sidepreds --env repeated_nextto --approach nsrt_learning --side_predicate_learner no_learning --seed $SEED --num_train_tasks 50
    # python $FILE --experiment_id rnt_harmlessness --env repeated_nextto --approach nsrt_learning --side_predicate_learner preserve_skeletons_hill_climbing --seed $SEED --num_train_tasks 50
    # python $FILE --experiment_id rnt_pred_error --env repeated_nextto --approach nsrt_learning --side_predicate_learner prediction_error_hill_climbing --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id rnt_backchain --env repeated_nextto --approach nsrt_learning --side_predicate_learner backchaining --seed $SEED --num_train_tasks 50

    # repeated_nextto_single_option
    # python $FILE --experiment_id rnt_single_option_oracle --env repeated_nextto_single_option --approach oracle --seed $SEED --num_train_tasks 0
    # python $FILE --experiment_id rnt_single_option_no_sidepreds --env repeated_nextto_single_option --approach nsrt_learning --side_predicate_learner no_learning --seed $SEED --num_train_tasks 50
    # python $FILE --experiment_id rnt_single_option_harmlessness --env repeated_nextto_single_option --approach nsrt_learning --side_predicate_learner preserve_skeletons_hill_climbing --seed $SEED --num_train_tasks 50
    # python $FILE --experiment_id rnt_single_option_pred_error --env repeated_nextto_single_option --approach nsrt_learning --side_predicate_learner prediction_error_hill_climbing --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id rnt_single_option_backchain --env repeated_nextto_single_option --approach nsrt_learning --side_predicate_learner backchaining --seed $SEED --num_train_tasks 50

    # repeated_nextto_painting
    # python $FILE --experiment_id rnt_painting_oracle --env repeated_nextto_painting --approach oracle --seed $SEED --num_train_tasks 0
done
