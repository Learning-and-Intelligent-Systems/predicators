#!/bin/bash
FILE="scripts/supercloud/submit_supercloud_job.py"
ALL_NUM_TRAIN_TASKS=(
     "10"
     "50"
     "100"
)
ENVS=(
    "cover"
    "painting"
    "screws"
    "repeated_nextto"
    "cluttered_table"
    "coffee"
)

for ENV in ${ENVS[@]}; do
  for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do

      COMMON_ARGS="--env $ENV  --strips_learner oracle \
          --sampler_learner oracle --num_train_tasks $NUM_TRAIN_TASKS \
          --pg3_hc_enforced_depth 1 --painting_lid_open_prob 1.0 \
          --coffee_jug_init_rot_amt 0 --cover_initial_holding_prob 0.0"

      python $FILE $COMMON_ARGS \
          --experiment_id ${ENV}_pg3_policy_with_options_${NUM_TRAIN_TASKS} \
          --approach pg3

      COMMON_ARGS="--env $ENV  --strips_learner oracle \
          --sampler_learner oracle --num_train_tasks $NUM_TRAIN_TASKS \
          --pg3_hc_enforced_depth 1"

      python $FILE $COMMON_ARGS \
          --experiment_id ${ENV}_pg3_policy_without_options_${NUM_TRAIN_TASKS}\
          --approach pg3

      if [ $ENV = "painting" ]; then
        #Painting environment test for when lid is open and there isn't initial
        #holding
        python $FILE $COMMON_ARGS \
          --experiment_id ${ENV}_pg3_policy_lid_noinithold_${NUM_TRAIN_TASKS} \
          --approach pg3 --painting_lid_open_prob 1.0 \
          --painting_initial_holding_prob 0.0

        #Painting environment test for when lid is sometimes closed and there
        #isn't initial holding
        python $FILE $COMMON_ARGS \
          --experiment_id ${ENV}_pg3_policy_lid_noinithold_${NUM_TRAIN_TASKS} \
          --approach pg3 --painting_initial_holding_prob 0.0
  done
done
