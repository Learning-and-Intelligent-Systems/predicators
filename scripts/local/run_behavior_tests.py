"""Easily launch experiments on a variety of BEHAVIOR environments."""

import json
import os
import shutil

NUM_TEST = 1
SEED = 0
TIMEOUT = 1000
OPEN_PICK_PLACE_TASKS = [
    'collecting_aluminum_cans', 'throwing_away_leftovers',
    'packing_bags_or_suitcase', 'packing_boxes_for_household_move_or_trip',
    'opening_presents', 'organizing_file_cabinet', 'locking_every_window',
    'packing_car_for_trip', 're-shelving_library_books', 'storing_food',
    'organizing_boxes_in_garage', 'putting_leftovers_away',
    'unpacking_suitcase', 'putting_away_toys', 'boxing_books_up_for_storage',
    'sorting_books', 'clearing_the_table_after_dinner', 'opening_packages',
    'picking_up_take-out_food', 'collect_misplaced_items',
    'locking_every_door', 'putting_dishes_away_after_cleaning',
    'picking_up_trash', 'cleaning_a_car', 'packing_food_for_work'
]


def _run_behavior_pickplaceopen_tests() -> None:
    path_to_file = "predicators/behavior_utils/task_to_preselected_scenes.json"
    f = open(path_to_file, 'rb')
    data = json.load(f)
    f.close()

    tasks_to_test = OPEN_PICK_PLACE_TASKS

    # Create commands to run
    cmds = []
    for task, scenes in data.items():
        if task in tasks_to_test:
            for scene in scenes:
                logfolder = os.path.join(
                    "logs", f"{task}_{scene}_{SEED}"
                    f"_{NUM_TEST}_{TIMEOUT}/")
                if os.path.exists(logfolder):
                    shutil.rmtree(logfolder)
                os.makedirs(logfolder)

                cmds.append("python predicators/main.py "
                            "--env behavior "
                            "--approach oracle "
                            "--behavior_mode headless "
                            "--option_model_name oracle_behavior "
                            "--num_train_tasks 1 "
                            f"--num_test_tasks {NUM_TEST} "
                            f"--behavior_scene_name {scene} "
                            f"--behavior_task_name {task} "
                            f"--seed {SEED} "
                            f"--offline_data_planning_timeout {TIMEOUT} "
                            f"--timeout {TIMEOUT} "
                            "--behavior_option_model_eval True "
                            "--plan_only_eval True "
                            f"--results_dir {logfolder} "
                            "--sesame_task_planner fdopt")

    # Run the commands in order.
    num_cmds = len(cmds)
    for i, cmd in enumerate(cmds):
        print(f"********* RUNNING COMMAND {i+1} of {num_cmds} *********")
        _ = os.popen(cmd).read()


if __name__ == "__main__":
    _run_behavior_pickplaceopen_tests()
