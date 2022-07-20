#!/bin/bash
FILE="scripts/supercloud/submit_supercloud_job.py"
ALL_NUM_TRAIN_TASKS=(
     "10"
     "50"
     "100"
)
ENVS=(
    "cover"
    "screws"
    "repeated_nextto"
    "cluttered_table"
    "coffee"
    "painting"
)
APPROACHES=(
        "pg3"
        "pg4"
)
for APPROACH in ${APPROACHES[@]}; do
  for ENV in ${ENVS[@]}; do
    for NUM_TRAIN_TASKS in ${ALL_NUM_TRAIN_TASKS[@]}; do
        # The common arguments will run PG3 and PG4 in each enviroment with their
        # default options
        # PG3 should fail here. PG4 should succeed
        COMMON_ARGS="--env $ENV  --strips_learner oracle \
            --sampler_learner oracle --num_train_tasks $NUM_TRAIN_TASKS \
            --pg3_hc_enforced_depth 1"

        # The additional options allow the following tests:
          # Cover: The agent will never be initially holding a block
          # Coffee: No tasks will require the agent to rotate the mug
          # Painting: All tasks will have the box lid open.
        # PG3 and PG4 should both succeed here. PG4 will simply run the
        # PG3 policy without any planning in these cases
        ADDITIONAL_OPTIONS="--painting_lid_open_prob 1.0 \
            --coffee_jug_init_rot_amt 0 --cover_initial_holding_prob 0.0"

        python $FILE $COMMON_ARGS \
            --experiment_id ${ENV}_${APPROACH}_without_options_${NUM_TRAIN_TASKS} \
            --approach ${APPROACH}

        python $FILE $COMMON_ARGS $ADDITIONAL_OPTIONS \
            --experiment_id ${ENV}_${APPROACH}_with_options_${NUM_TRAIN_TASKS} \
            --approach ${APPROACH}

        # Painting enviromemnt tests for when there isn't initial holding prob
        # In these tests the following should be expected:
        #   1) PG3 should fail when box lid is sometimes closed
        #   2) PG4 should succeed in both cases
        if [ $ENV = "painting" ]; then
          # Here lid will always be open
          python $FILE $COMMON_ARGS \
            --experiment_id ${ENV}_pg3_policy_lid_noinithold_${NUM_TRAIN_TASKS} \
            --approach ${APPROACH} --painting_lid_open_prob 1.0 \
            --painting_initial_holding_prob 0.0

          # Here lid can sometimes be closed
          python $FILE $COMMON_ARGS \
            --experiment_id ${ENV}_pg3_policy_lid_noinithold_${NUM_TRAIN_TASKS} \
            --approach ${APPROACH} --painting_initial_holding_prob 0.0
    done
  done
done
