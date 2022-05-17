#!/bin/bash

FILE="scripts/supercloud/submit_supercloud_job.py"

# cover
python $FILE --experiment_id cover_oracle --env cover --approach oracle --num_train_tasks 0
python $FILE --experiment_id cover_nsrt_learning --env cover --approach nsrt_learning --num_train_tasks 50
# python $FILE --experiment_id cover_invent_noexclude --env cover --approach grammar_search_invention --num_train_tasks 50
# python $FILE --experiment_id cover_invent_allexclude --env cover --approach grammar_search_invention --excluded_predicates all --num_train_tasks 50

# # pybullet_blocks
# python $FILE --experiment_id pybullet_blocks_oracle --env pybullet_blocks --approach oracle --num_train_tasks 0
# python $FILE --experiment_id pybullet_blocks_nsrt_learning --env pybullet_blocks --approach nsrt_learning --num_train_tasks 50
# python $FILE --experiment_id pybullet_blocks_invent_noexclude --env pybullet_blocks --approach grammar_search_invention --num_train_tasks 50
# python $FILE --experiment_id pybullet_blocks_invent_allexclude --env pybullet_blocks --approach grammar_search_invention --excluded_predicates all --num_train_tasks 50

# # painting
# python $FILE --experiment_id painting_oracle --env painting --approach oracle --num_train_tasks 0
# python $FILE --experiment_id painting_nsrt_learning --env painting --approach nsrt_learning --num_train_tasks 50
# python $FILE --experiment_id painting_invent_noexclude --env painting --approach grammar_search_invention --num_train_tasks 50
# python $FILE --experiment_id painting_invent_allexclude --env painting --approach grammar_search_invention --excluded_predicates all --num_train_tasks 50

# # tools
# # requires more data: "--num_train_tasks 200"
# python $FILE --experiment_id tools_oracle --env tools --approach oracle --num_train_tasks 0
# python $FILE --experiment_id tools_nsrt_learning --env tools --approach nsrt_learning --num_train_tasks 200
# python $FILE --experiment_id tools_invent_noexclude --env tools --approach grammar_search_invention --num_train_tasks 200
# python $FILE --experiment_id tools_invent_allexclude --env tools --approach grammar_search_invention --excluded_predicates all --num_train_tasks 200

# # playroom
# python $FILE --experiment_id playroom_oracle --env playroom --approach oracle --num_train_tasks 0
# python $FILE --experiment_id playroom_nsrt_learning --env playroom --approach nsrt_learning --num_train_tasks 50

# # pybullet_cover
# python $FILE --experiment_id pybullet_cover_oracle --env pybullet_cover --approach oracle --num_train_tasks 0
# python $FILE --experiment_id pybullet_cover_nsrt_learning --env pybullet_cover --approach nsrt_learning --num_train_tasks 50
# python $FILE --experiment_id pybullet_cover_invent_noexclude --env pybullet_cover --approach grammar_search_invention --num_train_tasks 50
# python $FILE --experiment_id pybullet_cover_invent_allexclude --env pybullet_cover --approach grammar_search_invention --excluded_predicates all --num_train_tasks 50

# # stick button
# # requires longer timeout: "--timeout 300"
# python $FILE --experiment_id stick_button_oracle --env stick_button --approach oracle --num_train_tasks 0 --timeout 300
# # requires more data: "--num_train_tasks 500"
# # requires filtering: "--min_perc_data_for_nsrt 1"
# python $FILE --experiment_id stick_button_nsrt_learning --env stick_button --approach nsrt_learning --timeout 300 --num_train_tasks 500 --min_perc_data_for_nsrt 1
