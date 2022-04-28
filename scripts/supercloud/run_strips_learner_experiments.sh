#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # repeated_nextto
    python $FILE --experiment_id rnt_oracle --env repeated_nextto --approach oracle --seed $SEED --num_train_tasks 0
    ## Although cluster_and_intersect is guaranteed to preserve harmlessness, we disable the check because it takes a long time (since the operators have high arity).
    python $FILE --experiment_id rnt_no_sidepreds --env repeated_nextto --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED --num_train_tasks 50 --disable_harmlessness_check True
    ## Hill climbing to optimize prediction error is not guaranteed to preserve harmlessness.
    python $FILE --experiment_id rnt_pred_error --env repeated_nextto --approach nsrt_learning --strips_learner cluster_and_intersect_sideline_prederror --seed $SEED --num_train_tasks 50 --disable_harmlessness_check True
    python $FILE --experiment_id rnt_backchain --env repeated_nextto --approach nsrt_learning --strips_learner backchaining --seed $SEED --num_train_tasks 50

    # repeated_nextto_single_option
    python $FILE --experiment_id rnt_single_option_oracle --env repeated_nextto_single_option --approach oracle --seed $SEED --num_train_tasks 0
    ## Although cluster_and_intersect is guaranteed to preserve harmlessness, we disable the check because it takes a long time (since the operators have high arity).
    python $FILE --experiment_id rnt_single_option_no_sidepreds --env repeated_nextto_single_option --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED --num_train_tasks 50 --disable_harmlessness_check True
    ## Hill climbing to optimize prediction error is not guaranteed to preserve harmlessness.
    python $FILE --experiment_id rnt_single_option_pred_error --env repeated_nextto_single_option --approach nsrt_learning --strips_learner cluster_and_intersect_sideline_prederror --seed $SEED --num_train_tasks 50 --disable_harmlessness_check True
    python $FILE --experiment_id rnt_single_option_backchain --env repeated_nextto_single_option --approach nsrt_learning --strips_learner backchaining --seed $SEED --num_train_tasks 50

    # repeated_nextto_painting
    python $FILE --experiment_id rnt_painting_oracle --env repeated_nextto_painting --approach oracle --seed $SEED --num_train_tasks 0
    python $FILE --experiment_id rnt_painting_backchain --env repeated_nextto_painting --approach nsrt_learning --strips_learner backchaining --seed $SEED --num_train_tasks 50
done
