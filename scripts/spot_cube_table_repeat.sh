#!/bin/bash
for i in {1..100}
do
   # Cube starts on original table
   python predicators/main.py --env spot_bike_env --approach spot_wrapper[oracle] --seed 0 --num_train_tasks 0 --num_test_tasks 1 --spot_robot_ip 10.17.30.30 --bilevel_plan_without_sim True --spot_grasp_use_apriltag True --perceiver spot_bike_env --test_task_json_dir predicators/envs/assets/task_jsons/spot_bike_env/cube_table/ --execution_monitor expected_atoms --spot_cube_only True
   # Cube on extra table
   python predicators/main.py --env spot_bike_env --approach spot_wrapper[oracle] --seed 0 --num_train_tasks 0 --num_test_tasks 1 --spot_robot_ip 10.17.30.30 --bilevel_plan_without_sim True --spot_grasp_use_apriltag True --perceiver spot_bike_env --test_task_json_dir predicators/envs/assets/task_jsons/spot_bike_env/cube_table_reverse/ --execution_monitor expected_atoms --spot_cube_only True
done
