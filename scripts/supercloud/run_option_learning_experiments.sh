#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    COMMON_ARGS="--approach nsrt_learning --implicit_mlp_regressor_num_samples_per_inference 16384 --implicit_mlp_regressor_grid_num_ticks_per_dim 50 --num_train_tasks 1000 --num_test_tasks 100 --seed $SEED"

    ## touch point
    # oracle
    python $FILE $COMMON_ARGS --experiment_id touch_point_oracle --env touch_point

    # direct BC
    python $FILE $COMMON_ARGS --experiment_id touch_point_direct --env touch_point --option_learner direct_bc

    # implicit BC: derivative_free
    python $FILE $COMMON_ARGS --experiment_id touch_point_implicit_df --env touch_point --option_learner implicit_bc --implicit_mlp_regressor_inference_method derivative_free

    # implicit BC: grid
    python $FILE $COMMON_ARGS --experiment_id touch_point_implicit_grid --env touch_point --option_learner implicit_bc --implicit_mlp_regressor_inference_method grid

    ## cover_multistep_options
    # oracle
    python $FILE $COMMON_ARGS --experiment_id cover_multi_oracle --env cover_multistep_options

    # direct BC
    python $FILE $COMMON_ARGS --experiment_id cover_multi_direct --env cover_multistep_options --option_learner direct_bc

    # implicit BC: derivative_free
    python $FILE $COMMON_ARGS --experiment_id cover_multi_implicit_df --env cover_multistep_options --option_learner implicit_bc --implicit_mlp_regressor_inference_method derivative_free

    # implicit BC: grid
    python $FILE $COMMON_ARGS --experiment_id cover_multi_implicit_grid --env cover_multistep_options --option_learner implicit_bc --implicit_mlp_regressor_inference_method grid

    ## cover_multistep_options_holding
    # oracle
    python $FILE $COMMON_ARGS --experiment_id cover_holding_multi_oracle --env cover_multistep_options_holding

    # direct BC
    python $FILE $COMMON_ARGS --experiment_id cover_holding_multi_direct --env cover_multistep_options_holding --option_learner direct_bc

    # implicit BC: derivative_free
    python $FILE $COMMON_ARGS --experiment_id cover_holding_multi_implicit_df --env cover_multistep_options_holding --option_learner implicit_bc --implicit_mlp_regressor_inference_method derivative_free

    # implicit BC: grid
    python $FILE $COMMON_ARGS --experiment_id cover_holding_multi_implicit_grid --env cover_multistep_options_holding --option_learner implicit_bc --implicit_mlp_regressor_inference_method grid

    ## cover_multistep_options single block
    # oracle
    python $FILE $COMMON_ARGS --experiment_id cover_multi_single_oracle --env cover_multistep_options --cover_num_targets 1 --cover_num_blocks 1

    # direct BC
    python $FILE $COMMON_ARGS --experiment_id cover_multi_single_direct --env cover_multistep_options --cover_num_targets 1 --cover_num_blocks 1 --option_learner direct_bc

    # implicit BC: derivative_free
    python $FILE $COMMON_ARGS --experiment_id cover_multi_single_implicit_df --env cover_multistep_options --cover_num_targets 1 --cover_num_blocks 1 --option_learner implicit_bc --implicit_mlp_regressor_inference_method derivative_free

    # implicit BC: grid
    python $FILE $COMMON_ARGS --experiment_id cover_multi_single_implicit_grid --env cover_multistep_options --cover_num_targets 1 --cover_num_blocks 1 --option_learner implicit_bc --implicit_mlp_regressor_inference_method grid

done
