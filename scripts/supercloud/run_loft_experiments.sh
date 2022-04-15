#!/bin/bash

START_SEED=456
NUM_SEEDS=10
FILE="scripts/supercloud/submit_supercloud_job.py"

for SEED in $(seq $START_SEED $((NUM_SEEDS+START_SEED-1))); do
    # cover
    python $FILE --experiment_id cover_intersect_demoonly --env cover --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED
    python $FILE --experiment_id cover_intersect_demoreplay --env cover --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED --offline_data_method demo+replay --sesame_allow_noops False
    python $FILE --experiment_id cover_search_demoonly --env cover --approach nsrt_learning --strips_learner cluster_and_search --seed $SEED
    python $FILE --experiment_id cover_search_demoreplay --env cover --approach nsrt_learning --strips_learner cluster_and_search --seed $SEED --offline_data_method demo+replay --sesame_allow_noops False

    # blocks
    python $FILE --experiment_id blocks_intersect_demoonly --env blocks --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED
    python $FILE --experiment_id blocks_intersect_demoreplay --env blocks --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED --offline_data_method demo+replay --sesame_allow_noops False
    python $FILE --experiment_id blocks_search_demoonly --env blocks --approach nsrt_learning --strips_learner cluster_and_search --seed $SEED
    python $FILE --experiment_id blocks_search_demoreplay --env blocks --approach nsrt_learning --strips_learner cluster_and_search --seed $SEED --offline_data_method demo+replay --sesame_allow_noops False

    # painting
    python $FILE --experiment_id painting_intersect_demoonly --env painting --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED
    python $FILE --experiment_id painting_intersect_demoreplay --env painting --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED --offline_data_method demo+replay --sesame_allow_noops False
    python $FILE --experiment_id painting_search_demoonly --env painting --approach nsrt_learning --strips_learner cluster_and_search --seed $SEED
    python $FILE --experiment_id painting_search_demoreplay --env painting --approach nsrt_learning --strips_learner cluster_and_search --seed $SEED --offline_data_method demo+replay --sesame_allow_noops False

    # tools
    # requires more data: "--num_train_tasks 200" and "--offline_data_num_replays 2500"
    python $FILE --experiment_id tools_intersect_demoonly --env tools --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED --num_train_tasks 200
    python $FILE --experiment_id tools_intersect_demoreplay --env tools --approach nsrt_learning --strips_learner cluster_and_intersect --seed $SEED --num_train_tasks 200 --offline_data_method demo+replay --sesame_allow_noops False --offline_data_num_replays 2500
    python $FILE --experiment_id tools_search_demoonly --env tools --approach nsrt_learning --strips_learner cluster_and_search --seed $SEED --num_train_tasks 200
    python $FILE --experiment_id tools_search_demoreplay --env tools --approach nsrt_learning --strips_learner cluster_and_search --seed $SEED --num_train_tasks 200 --offline_data_method demo+replay --sesame_allow_noops False --offline_data_num_replays 2500
done
