import subprocess
import time

all_seeds = {"0", "1", "2", "3", "4"}

for i in range(0, 5):
    try:
        theproc2 = subprocess.run([
            "python", "src/main.py", "--seed",
            str(i), "--approach", "gnn_option_policy", "--strips_learner",
            "oracle", "--env", "pddl_delivery_procedural_tasks",
            "--num_train_tasks", "5", "--num_test_tasks", "10",
            "--pddl_delivery_procedural_train_min_num_locs", "4",
            "--pddl_delivery_procedural_train_max_num_locs", "7",
            "--pddl_delivery_procedural_train_min_want_locs", "1",
            "--pddl_delivery_procedural_train_max_want_locs", "3",
            "--pddl_delivery_procedural_test_min_num_locs", "17",
            "--pddl_delivery_procedural_test_max_num_locs", "23",
            "--pddl_delivery_procedural_test_min_want_locs", "11",
            "--pddl_delivery_procedural_test_max_want_locs", "16",
            "--pddl_delivery_procedural_test_max_extra_newspapers", "5",
            "--gnn_option_policy_solve_with_shooting", "False",
            "--offline_data_planning_timeout", "100.0", "--timeout", "100.0",
            "--debug"
        ],
                                  shell=True)
        time.sleep(1)
    except:
        continue

print("medium ----- delivery ---- finished")

for i in range(0, 5):
    try:
        theproc = subprocess.run([
            "python", "src/main.py", "--seed",
            str(i), "--approach", "gnn_option_policy", "--strips_learner",
            "oracle", "--env", "pddl_blocks_procedural_tasks",
            "--num_train_tasks", "5", "--num_test_tasks", "10",
            "--pddl_blocks_procedural_train_min_num_blocks", "7",
            "--pddl_blocks_procedural_train_max_num_blocks", "8",
            "--pddl_blocks_procedural_train_min_num_blocks_goal", "6",
            "--pddl_blocks_procedural_train_max_num_blocks_goal", "7",
            "--pddl_blocks_procedural_test_min_num_blocks", "9",
            "--pddl_blocks_procedural_test_max_num_blocks", "10",
            "--pddl_blocks_procedural_test_min_num_blocks_goal", "6",
            "--pddl_blocks_procedural_test_max_num_blocks_goal", "9",
            "--gnn_option_policy_solve_with_shooting", "False",
            "--offline_data_planning_timeout", "100.0", "--timeout", "100.0",
            "--debug"
        ],
                                 shell=True)
        time.sleep(1)
    except:
        continue

print("medium ----- blocks ---- finished")
