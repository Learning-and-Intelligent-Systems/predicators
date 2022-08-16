#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"
NUM_TRAIN_TASKS="1000"
ALL_NUM_IRRELEVANT_PREDICATES=(
    "0"
    # "25"
    # "50"
    # "75"
    # "100"
)
ENV="cover_multistep_options"

# NOTE: first run with 0 only, then add --load_data

for NUM_IRRELEVANT in ${ALL_NUM_IRRELEVANT_PREDICATES[@]}; do

    COMMON_ARGS="--env $ENV --min_perc_data_for_nsrt 1 \
            --segmenter contacts --num_train_tasks $NUM_TRAIN_TASKS --timeout 300 \
            --gnn_num_epochs 1000 --disable_harmlessness_check True \
            --neural_gaus_regressor_max_itr 50000"

    # irrelevant static predicates
    python $FILE $COMMON_ARGS --cover_num_irrelevant_static_preds $NUM_IRRELEVANT --experiment_id ${ENV}_main_irrelevant_static_${NUM_IRRELEVANT} --approach nsrt_learning --option_learner direct_bc

    # irrelevant dynamic predicates
    python $FILE $COMMON_ARGS --cover_num_irrelevant_dynamic_preds $NUM_IRRELEVANT --experiment_id ${ENV}_main_irrelevant_dynamic_${NUM_IRRELEVANT} --approach nsrt_learning --option_learner direct_bc

    # irrelevant random predicates
    python $FILE $COMMON_ARGS --cover_num_irrelevant_random_preds $NUM_IRRELEVANT --experiment_id ${ENV}_main_irrelevant_random_${NUM_IRRELEVANT} --approach nsrt_learning --option_learner direct_bc

done
