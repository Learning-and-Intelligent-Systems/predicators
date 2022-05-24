#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"

# cover
python $FILE --experiment_id cover_intersect_demoonly --env cover --approach nsrt_learning --strips_learner cluster_and_intersect
python $FILE --experiment_id cover_intersect_demoreplay --env cover --approach nsrt_learning --strips_learner cluster_and_intersect --offline_data_method demo+replay --sesame_allow_noops False
python $FILE --experiment_id cover_search_demoonly --env cover --approach nsrt_learning --strips_learner cluster_and_search
python $FILE --experiment_id cover_search_demoreplay --env cover --approach nsrt_learning --strips_learner cluster_and_search --offline_data_method demo+replay --sesame_allow_noops False

# blocks
python $FILE --experiment_id blocks_intersect_demoonly --env blocks --approach nsrt_learning --strips_learner cluster_and_intersect
python $FILE --experiment_id blocks_intersect_demoreplay --env blocks --approach nsrt_learning --strips_learner cluster_and_intersect --offline_data_method demo+replay --sesame_allow_noops False
python $FILE --experiment_id blocks_search_demoonly --env blocks --approach nsrt_learning --strips_learner cluster_and_search
python $FILE --experiment_id blocks_search_demoreplay --env blocks --approach nsrt_learning --strips_learner cluster_and_search --offline_data_method demo+replay --sesame_allow_noops False

# painting
python $FILE --experiment_id painting_intersect_demoonly --env painting --approach nsrt_learning --strips_learner cluster_and_intersect
python $FILE --experiment_id painting_intersect_demoreplay --env painting --approach nsrt_learning --strips_learner cluster_and_intersect --offline_data_method demo+replay --sesame_allow_noops False
python $FILE --experiment_id painting_search_demoonly --env painting --approach nsrt_learning --strips_learner cluster_and_search
python $FILE --experiment_id painting_search_demoreplay --env painting --approach nsrt_learning --strips_learner cluster_and_search --offline_data_method demo+replay --sesame_allow_noops False

# tools
# requires more data: "--num_train_tasks 200" and "--offline_data_num_replays 2500"
python $FILE --experiment_id tools_intersect_demoonly --env tools --approach nsrt_learning --strips_learner cluster_and_intersect --num_train_tasks 200
python $FILE --experiment_id tools_intersect_demoreplay --env tools --approach nsrt_learning --strips_learner cluster_and_intersect --num_train_tasks 200 --offline_data_method demo+replay --sesame_allow_noops False --offline_data_num_replays 2500
python $FILE --experiment_id tools_search_demoonly --env tools --approach nsrt_learning --strips_learner cluster_and_search --num_train_tasks 200
python $FILE --experiment_id tools_search_demoreplay --env tools --approach nsrt_learning --strips_learner cluster_and_search --num_train_tasks 200 --offline_data_method demo+replay --sesame_allow_noops False --offline_data_num_replays 2500
