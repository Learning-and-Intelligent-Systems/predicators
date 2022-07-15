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

      if [ $ENV = "painting" ] || [ $ENV = "cluttered_table" ] || \
          [ $ENV = "cluttered_table" ] then
        COMMON_ARGS="--env $ENV  --strips_learner oracle \
              --sampler_learner oracle --num_train_tasks $NUM_TRAIN_TASKS \
              --pg3_hc_enforced_depth 1"
      else
        COMMON_ARGS="--env $ENV  --strips_learner oracle \
              --sampler_learner oracle --num_train_tasks $NUM_TRAIN_TASKS \"
      fi

      # If in painting enviroment override common arguments and included options
      if [ $ENV = "painting" ]; then
        INCLUDED_OPTIONS="--painting_lid_open_prob 1.0"
      else
        INCLUDED_OPTIONS="""
      fi

      # If in coffee enviroment override included options
      if [ $ENV = "coffee" ]; then
        INCLUDED_OPTIONS="--coffee_jug_init_rot_amt 0"
      else
        INCLUDED_OPTIONS="""
      fi

      if [ $ENV = "cover" ]; then
        INCLUDED_OPTIONS="--cover_initial_holding_prob 0.0"
      else
        INCLUDED_OPTIONS="""
      fi

      if [ $INCLUDED_OPTIONS = ""]; then
        # PG3 policy generation on TAMP problems using default options
        python $FILE $COMMON_ARGS \
              --experiment_id ${ENV}_pg3_policy_${NUM_TRAIN_TASKS} \
              --approach pg3
      else
        # PG3 policy generation on TAMP problems using default options
        # and included options
        python $FILE $COMMON_ARGS \
              --experiment_id ${ENV}_pg3_policy_${NUM_TRAIN_TASKS} \
              --approach pg3
        python $FILE $COMMON_ARGS $INCLUDED_OPTIONS \
              --experiment_id ${ENV}_main_${NUM_TRAIN_TASKS} --approach pg3
      fi
  done
done
