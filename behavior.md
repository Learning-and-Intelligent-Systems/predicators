# Running BEHAVIOR Experiments

## Installation
This repository and some of the functionality it enables are integrated with the [BEHAVIOR benchmark of tasks](https://behavior.stanford.edu/benchmark-guide) simulated with the [iGibson simulator](https://github.com/StanfordVL/iGibson). To install iGibson with BEHAVIOR, follow [**Option 1** from these instructions](https://stanfordvl.github.io/behavior/installation.html). However, note that you need to use our forks of the [iGibson](https://github.com/Learning-and-Intelligent-Systems/iGibson) and [bddl](https://github.com/Learning-and-Intelligent-Systems/bddl) repositories. Thus, instead of steps 2a. and 2b. of Option 1 in the linked instructions, run:
```
git clone https://github.com/Learning-and-Intelligent-Systems/iGibson.git --recursive
git clone https://github.com/Learning-and-Intelligent-Systems/bddl.git
```

## Running Experiments
* Currently, only the `oracle` approach is implemented to integrate with BEHAVIOR.
* Note that you'll probably want to provide the command line argument `--timeout 1000` to prevent early stopping.
* Set `--option_model_name oracle_behavior` to use the behavior option model and speed up planning by a significant factor.
* Set `--behavior_task_list` to the list of the particular bddl tasks you'd like to run (e.g. `"[re-shelving_library_books]"`).
* Set `--behavior_scene_name` to the name of the house setting (e.g. `Pomaria_1_int`) you want to try running the particular task in. Note that not all tasks are available in all houses (e.g. `re-shelving_library_books` might only be available with `Pomaria_1_int`).
* `--behavior_randomize_init_state` can be set to True if you want to generate multiple different initial states that correspond to the BDDL init conditions of a particular task.
* If you'd like to see a visual of the agent planning in iGibson, set the command line argument `--behavior_mode simple`. If you want to run in headless mode without any visuals, leave the default (i.e `--behavior_mode headless`).
* Example command: `python predicators/main.py --env behavior --approach oracle --seed 0 --timeout 1000 --sesame_max_samples_per_step 20 --behavior_mode simple --option_model_name oracle_behavior --num_train_tasks 2 --num_test_tasks 2 --behavior_randomize_init_state True --behavior_scene_name Pomaria_1_int --behavior_task_list "[re-shelving_library_books]"`.