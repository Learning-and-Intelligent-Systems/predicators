#!/bin/bash

echo "Running sorting_books on Pomaria_1_int"
python src/main.py --env behavior --approach oracle --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 1 --num_test_tasks 1 --behavior_scene_name Pomaria_1_int --behavior_task_name sorting_books --approach nsrt_learning --strips_learner backchaining --seed 0 --offline_data_planning_timeout 10000.0 &> Pomaria_1_int_sorting_books.txt

echo "Running re-shelving_library_books on Pomaria_1_int"
python src/main.py --env behavior --approach oracle --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 1 --num_test_tasks 1 --behavior_scene_name Pomaria_1_int --behavior_task_name re-shelving_library_books --approach nsrt_learning --strips_learner backchaining --seed 0 --offline_data_planning_timeout 10000.0 &> Pomaria_1_int_re-shelving_library_books.txt

echo "Running unpacking_suitcase on Ihlen_1_int"
python src/main.py --env behavior --approach oracle --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 1 --num_test_tasks 1 --behavior_scene_name Ihlen_1_int --behavior_task_name unpacking_suitcase --approach nsrt_learning --strips_learner backchaining --seed 0 --offline_data_planning_timeout 10000.0 &> Ihlen_1_int_unpacking_suitcase.txt

echo "Running unpacking_suitcase on Benevolence_1_int"
python src/main.py --env behavior --approach oracle --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 1 --num_test_tasks 1 --behavior_scene_name Benevolence_1_int --behavior_task_name unpacking_suitcase --approach nsrt_learning --strips_learner backchaining --seed 0 --offline_data_planning_timeout 10000.0 &> Benevolence_1_int_unpacking_suitcase.txt

echo "Running unpacking_suitcase on Beechwood_0_int"
python src/main.py --env behavior --approach oracle --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 1 --num_test_tasks 1 --behavior_scene_name Beechwood_0_int --behavior_task_name unpacking_suitcase --approach nsrt_learning --strips_learner backchaining --seed 0 --offline_data_planning_timeout 10000.0 &> Beechwood_0_int_unpacking_suitcase.txt
