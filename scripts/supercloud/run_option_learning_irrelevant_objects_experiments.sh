#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"
NUM_TRAIN_TASKS="1000"
ALL_NUM_IRRELEVANT_OBJECTS=(
    "0"
    "25"
    "50"
    "75"
    "100"
)
ENV="cover_multistep_options"


for NUM_IRRELEVANT_OBJECTS in ${ALL_NUM_IRRELEVANT_OBJECTS[@]}; do

    COMMON_ARGS="--env $ENV --min_perc_data_for_nsrt 1 \
            --segmenter contacts --num_train_tasks $NUM_TRAIN_TASKS --timeout 300 \
            --gnn_num_epochs 1000 --disable_harmlessness_check True \
            --neural_gaus_regressor_max_itr 50000"

    # train has extra objs
    python $FILE $COMMON_ARGS --cover_train_num_irrelevant_blocks $NUM_IRRELEVANT_OBJECTS --experiment_id ${ENV}_main_train_irrelevant_${NUM_IRRELEVANT_OBJECTS} --approach nsrt_learning --option_learner direct_bc

    # test has extra objs
    python $FILE $COMMON_ARGS --cover_test_num_irrelevant_blocks $NUM_IRRELEVANT_OBJECTS --experiment_id ${ENV}_main_test_irrelevant_${NUM_IRRELEVANT_OBJECTS} --approach nsrt_learning --option_learner direct_bc

done
