#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # cover
    python $FILE --experiment_id cover_oracle --env cover --approach oracle --seed $SEED --num_train_tasks 0
    python $FILE --experiment_id cover_nsrt_learning --env cover --approach nsrt_learning --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id cover_invent_noexclude --env cover --approach grammar_search_invention --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id cover_invent_allexclude --env cover --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 50

    # blocks
    python $FILE --experiment_id blocks_oracle --env blocks --approach oracle --seed $SEED --num_train_tasks 0
    python $FILE --experiment_id blocks_nsrt_learning --env blocks --approach nsrt_learning --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id blocks_invent_noexclude --env blocks --approach grammar_search_invention --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id blocks_invent_allexclude --env blocks --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 50

    # painting
    python $FILE --experiment_id painting_oracle --env painting --approach oracle --seed $SEED --num_train_tasks 0
    python $FILE --experiment_id painting_nsrt_learning --env painting --approach nsrt_learning --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id painting_invent_noexclude --env painting --approach grammar_search_invention --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id painting_invent_allexclude --env painting --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 50

    # tools
    # requires more data: "--num_train_tasks 200"
    python $FILE --experiment_id tools_oracle --env tools --approach oracle --seed $SEED --num_train_tasks 0
    python $FILE --experiment_id tools_nsrt_learning --env tools --approach nsrt_learning --seed $SEED --num_train_tasks 200
    python $FILE --experiment_id tools_invent_noexclude --env tools --approach grammar_search_invention --seed $SEED --num_train_tasks 200
    python $FILE --experiment_id tools_invent_allexclude --env tools --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 200

    # pybullet_cover
    python $FILE --experiment_id pybullet_cover_oracle --env pybullet_cover --approach oracle --seed $SEED --num_train_tasks 0
    python $FILE --experiment_id pybullet_cover_nsrt_learning --env pybullet_cover --approach nsrt_learning --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id pybullet_cover_invent_noexclude --env pybullet_cover --approach grammar_search_invention --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id pybullet_cover_invent_allexclude --env pybullet_cover --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 50

    # pybullet_blocks
    python $FILE --experiment_id pybullet_blocks_oracle --env pybullet_blocks --approach oracle --seed $SEED --num_train_tasks 0
    python $FILE --experiment_id pybullet_blocks_nsrt_learning --env pybullet_blocks --approach nsrt_learning --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id pybullet_blocks_invent_noexclude --env pybullet_blocks --approach grammar_search_invention --seed $SEED --num_train_tasks 50
    python $FILE --experiment_id pybullet_blocks_invent_allexclude --env pybullet_blocks --approach grammar_search_invention --excluded_predicates all --seed $SEED --num_train_tasks 50
done
