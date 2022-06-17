#!/bin/bash

echo "Running sorting_books on Rs_int"
python src/main.py --env behavior --approach oracle --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 1 --num_test_tasks 1 --behavior_scene_name Rs_int --behavior_task_name sorting_books --approach nsrt_learning --strips_learner backchaining --seed 0 --offline_data_planning_timeout 10000.0 &> Rs_int_sorting_books.txt


echo "Running sorting_books on Ihlen_0_int"
python src/main.py --env behavior --approach oracle --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 1 --num_test_tasks 1 --behavior_scene_name Ihlen_0_int --behavior_task_name sorting_books --approach nsrt_learning --strips_learner backchaining --seed 0 --offline_data_planning_timeout 10000.0 &> Ihlen_0_int_1_int_sorting_books.txt


echo "Running sorting_books on Pomaria_1_int"
python src/main.py --env behavior --approach oracle --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 1 --num_test_tasks 1 --behavior_scene_name Pomaria_1_int --behavior_task_name sorting_books --approach nsrt_learning --strips_learner backchaining --seed 0 --offline_data_planning_timeout 10000.0 &> Pomaria_1_int_sorting_books.txt