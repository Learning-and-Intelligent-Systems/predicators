"""Easily launch experiments on a variety of BEHAVIOR environments."""

import json
import os
import shutil

NUM_TEST = 1
SEED = 0
TIMEOUT = 300
TASKS_AND_SCENE = [["opening_presents", "Benevolence_2_int"],
["opening_presents", "Pomaria_2_int"],
["opening_packages", "Pomaria_2_int"],
["opening_packages", "Wainscott_1_int"],
["opening_presents", "Wainscott_1_int"],
["re-shelving_library_books", "Ihlen_0_int"],
["opening_packages", "Benevolence_2_int"],
["locking_every_door", "Merom_1_int"],
["re-shelving_library_books", "Pomaria_1_int"],
["locking_every_window", "Wainscott_0_int"],
["sorting_books", "Pomaria_1_int"],
["locking_every_door", "Pomaria_0_int"],
["locking_every_window", "Merom_1_int"]]


def _run_behavior_pickplaceopen_tests() -> None:

    tasks_to_test = TASKS_AND_SCENE

    # Create commands to run
    cmds = []
    for task, scene in tasks_to_test:
        logfolder = os.path.join(
            "visuals", f"{task}_{scene}_{SEED}"
            f"_{NUM_TEST}_{TIMEOUT}/")
        if os.path.exists(logfolder):
            shutil.rmtree(logfolder)
        os.makedirs(logfolder)

        cmds.append("python predicators/main.py "
                    "--env behavior "
                    "--approach oracle "
                    "--behavior_mode simple "
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
