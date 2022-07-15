import sys
import subprocess
import time
all_seeds = {"0","1","2","3","4"}
for i in range(0,5):
    try:
        theproc = subprocess.run(["python","src/main.py", "--seed", str(i), 
        "--approach", "gnn_option_policy", "--strips_learner", "oracle", "--env", "pddl_blocks_procedural_tasks", 
        "--num_train_tasks", "5", "--num_test_tasks", "10", "--pddl_blocks_procedural_train_min_num_blocks", 
        "11", "--pddl_blocks_procedural_train_max_num_blocks","12", "--pddl_blocks_procedural_train_min_num_blocks_goal", 
        "10", "--pddl_blocks_procedural_train_max_num_blocks_goal", "11", "--pddl_blocks_procedural_test_min_num_blocks", 
        "13","--pddl_blocks_procedural_test_max_num_blocks", "14","--pddl_blocks_procedural_test_min_num_blocks_goal",
        "10","--pddl_blocks_procedural_test_max_num_blocks_goal","13","--gnn_option_policy_solve_with_shooting","false",
        "--offline_data_planning_timeout", "100.0", "--timeout", "100.0","--debug"], shell = True)

        time.sleep(1)
    except:
        continue
print("hard ----- blocks ---- finished")
for i in range(0,5):
    try:
        theproc2 = subprocess.run(["python","src/main.py", "--seed", str(i), 
        "--approach", "gnn_option_policy", "--strips_learner", "oracle", "--env", "pddl_delivery_procedural_tasks", 
        "--num_train_tasks", "5", "--num_test_tasks", "10", "--pddl_delivery_procedural_train_min_num_locs", 
        "5", "--pddl_delivery_procedural_train_max_num_locs","10", "--pddl_delivery_procedural_train_min_want_locs", 
        "2", "--pddl_delivery_procedural_train_max_want_locs", "4", "--pddl_delivery_procedural_test_min_num_locs", 
        "31","--pddl_delivery_procedural_test_max_num_locs", "40","--pddl_delivery_procedural_test_min_want_locs",
        "20","--pddl_delivery_procedural_test_max_want_locs","30","--pddl_delivery_procedural_test_max_extra_newspapers",
        "10","--gnn_option_policy_solve_with_shooting","false","--offline_data_planning_timeout", "100.0", "--timeout", 
        "100.0", "--debug"], shell = True)
        time.sleep(1)
    except:
        continue
print("hard ----- delivery ---- finished")

