#!/bin/bash

START_SEED=0
NUM_SEEDS=10

START_SAMPLES_PER_STEP=1
TOTAL_NUM_SAMPLES_PER_STEP=10

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do

    # Run experiments varying sesame_max_samples_per_step.
    for SAMPLES in $(seq $START_SAMPLES_PER_STEP $((TOTAL_NUM_SAMPLES_PER_STEP+START_SAMPLES_PER_STEP-1))); do

        # Goal-conditioned cover
        python src/main.py --experiment_id "cover_goal_conditioned_sampling_${SAMPLES}" --env cover_multistep_options --approach nsrt_learning --seed $SEED --num_train_tasks 50 --num_test_tasks 50 --cover_num_blocks 1 --cover_num_targets 1 --cover_multistep_bimodal_goal True --cover_multistep_thr_percent 0.3 --cover_multistep_bhr_percent 0.99 --cover_block_widths "[0.12]" --cover_target_widths "[0.1]" --sesame_max_skeletons_optimized 1 --sesame_max_samples_per_step $SAMPLES --cover_multistep_goal_conditioned_sampling True --sampler_learning_use_goals_cover_version True --sampler_learning_use_goals True

        # Non-goal-conditioned cover
        python src/main.py --experiment_id "cover_nongoal_conditioned_sampling_${SAMPLES}" --env cover_multistep_options --approach nsrt_learning --seed $SEED --num_train_tasks 50 --num_test_tasks 50 --cover_num_blocks 1 --cover_num_targets 1 --cover_multistep_bimodal_goal True --cover_multistep_thr_percent 0.3 --cover_multistep_bhr_percent 0.99 --cover_block_widths "[0.12]" --cover_target_widths "[0.1]" --sesame_max_skeletons_optimized 1 --sesame_max_samples_per_step $SAMPLES --cover_multistep_goal_conditioned_sampling True --sampler_learning_use_goals_cover_version True --sampler_learning_use_goals False

        # Goal-conditioned cluttered table
        python src/main.py --experiment_id "cluttered_table_goal_conditioned_sampling_${SAMPLES}" --env cluttered_table_place --approach nsrt_learning --sesame_max_skeletons_optimized 2 --sesame_max_samples_per_step $SAMPLES --num_test_tasks 50 --cluttered_table_num_cans_test 2 --cluttered_table_place_goal_conditioned_sampling True --seed 0 --sampler_learning_use_goals True --sampler_disable_classifier True

        # Non-goal-conditioned cluttered table
        python src/main.py --experiment_id "cluttered_table_nongoal_conditioned_sampling_${SAMPLES}" --env cluttered_table_place --approach nsrt_learning --sesame_max_skeletons_optimized 2 --sesame_max_samples_per_step $SAMPLES --num_test_tasks 50 --cluttered_table_num_cans_test 2 --cluttered_table_place_goal_conditioned_sampling True --seed 0 --sampler_learning_use_goals False --sampler_disable_classifier True

    done

done

