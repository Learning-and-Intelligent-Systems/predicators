# Running BEHAVIOR Experiments

## Installation
This repository is integrated with the [BEHAVIOR benchmark of tasks](https://behavior.stanford.edu/benchmark-guide) simulated with the [iGibson simulator](https://github.com/StanfordVL/iGibson). To install iGibson with BEHAVIOR, follow the below steps (NOTE: if you're on a Mac, [these instructions here](https://github.com/Learning-and-Intelligent-Systems/iGibson/blob/master/mac_behavior_installation.md) might be helpful to reference as well):

1. Make sure you have all the prerequisites for iGibson installation. These are listed [here](https://stanfordvl.github.io/iGibson/installation.html#installing-dependencies).
1. Install this (`predicators_behavior`) repository as described in [the README](https://github.com/Learning-and-Intelligent-Systems/predicators_behavior#installation).
    1. Preferably, do this in a new virtual or conda environment!
1. Clone the necessary repositories to run BEHAVIOR:
    1. Our fork of the iGibson simulation environment:
        ```
        git clone https://github.com/Learning-and-Intelligent-Systems/iGibson.git --recursive
        ```
    1. Our fork of the BDDL repository, which contains all the task definitions:
        ```
        git clone https://github.com/Learning-and-Intelligent-Systems/bddl.git
        ```
1. Download and obtain access to the BEHAVIOR Dataset of Objects (3D assets with physical and semantic annotations) 
    1. Accept the license agreement filling the [form](https://forms.gle/GXAacjpnotKkM2An7). This allows you to use the assets within iGibson for free for your research.    
    1. You will receive a encryption key (`igibson.key`). Move the key into the data folder of the iGibson repository, `iGibson/igibson/data`.    
    1. Download the BEHAVIOR data bundle including the BEHAVIOR Dataset of Objects and the iGibson2 Dataset of scenes, and particular robot assets. There will be two sub-folders: `ig_dataset` and `assets` that need to be extracted into the `iGibson/igibson/data` folder.
    ```
    wget https://www.dropbox.com/s/p8ljo8yeanrjfgc/assets_ig_dataset.zip?dl=0
    mv assets_ig_dataset.zip\?dl\=0 ./assets_ig_dataset.zip
    unzip assets_ig_dataset.zip -d .
    mv assets_ig_dataset/* ./iGibson/igibson/data/
    rm -rf assets_ig_dataset.zip
    rm -rf assets_ig_dataset
    ```
1. Make sure there is no version of `pybullet` currently installed in your virtual environment (if there is, it will create problems for the next step). You can do this with `pip uninstall pybullet`.
1. Within a virtual environment (preferably, the one you created to install this overall repository), install the downloaded repositories:
    ```
    pip install -e ./iGibson
    pip install -e ./bddl
    ```
    1. Note that if you're on MIT Supercloud, you'll need to create a separate temp directory for pip to cache files to avoid an out-of-memory issue:
    ```
    mkdir /state/partition1/user/$USER
    export TMPDIR=/state/partition1/user/$USER
    pip install --user --no-cache-dir -e ./iGibson
    pip install --user --no-cache-dir -e ./bddl
    ```

That's it! You can verify installation by running a simple command such as:
```
python predicators/main.py --env behavior --approach oracle --option_model_name oracle_behavior --num_train_tasks 0 --num_test_tasks 1 --behavior_scene_name Pomaria_2_int --behavior_task_list "[opening_packages]" --seed 1000 --offline_data_planning_timeout 500.0 --timeout 500.0 --behavior_option_model_eval True --plan_only_eval True
```

## Installing on MIT Supercloud
First, follow steps in our [Supercloud guide](supercloud.md) to get an account and setup this repository on Supercloud.

Next, simply follow the steps linked in the [above section](#installation) (though ignore the first step; supercloud already has the iGibson pre-reqs installed)! Note that for various driver-related reasons, this code only works on GPU-machines with supercloud (so always remember to request a GPU when submitting jobs involving this codebase).

To test installation, do:
```
LLsub -i -g volta:1  # request a compute node with GPU in interactive mode.
predicate_behavior  # cd to the predicators_behavior repo and activate the relevant conda environment.
# Run a sample command. You should see the agent output 1/1 tasks solved!
python predicators/main.py --env behavior --approach oracle --option_model_name oracle_behavior --num_train_tasks 0 --num_test_tasks 1 --behavior_scene_name Pomaria_2_int --behavior_task_list "[opening_packages]" --seed 1000 --offline_data_planning_timeout 500.0 --timeout 500.0 --behavior_option_model_eval True --plan_only_eval True
```

### (Optional) Installing Fast-Downward
A number of more complex and long-horizon BEHAVIOR tasks require use of the Fast-Downward planning system to solve. To install this on Supercloud:
```
cd ~ # NOTE: Can alternatively cd anywhere else to place the downward repo.
git clone https://github.com/ronuchit/downward.git
cd downward && ./build.py
export FD_EXEC_PATH="~/downward" # NOTE: If you installed downward elsewhere, provide this path.
```
It is a good idea to add this last command (exporting of the FD_EXEC_PATH variable) to your `~/.bashrc` (ideally as part of the `predicate_behavior` command if you followed our Supercloud installation instructions). This ensures that this command is always executed prior to running planning.

To test installtion, do:
```
LLsub -i -g volta:1  # request a compute node with GPU in interactive mode.
predicate_behavior  # cd to the predicators_behavior repo and activate the relevant conda environment.
# Run a sample command. You should see the agent output 1/1 tasks solved!
python predicators/main.py --env behavior --approach oracle --option_model_name oracle_behavior --num_train_tasks 0 --num_test_tasks 1 --behavior_scene_name Pomaria_2_int --behavior_task_list "[opening_packages]" --seed 1000 --offline_data_planning_timeout 500.0 --timeout 500.0 --behavior_option_model_eval True --plan_only_eval True --sesame_task_planner fdopt
```


## Running Experiments
* Currently, only the `oracle` approach is implemented to integrate with BEHAVIOR.
* Note that you'll probably want to provide the command line argument `--timeout 1000` to prevent early stopping.
* Set `--option_model_name oracle_behavior` to use the behavior option model and speed up planning by a significant factor.
* Set `--behavior_task_list` to the list of the particular bddl tasks you'd like to run (e.g. `"[re-shelving_library_books]"`).
* Set `--behavior_scene_name` to the name of the house setting (e.g. `Pomaria_1_int`) you want to try running the particular task in. Note that not all tasks are available in all houses (e.g. `re-shelving_library_books` might only be available with `Pomaria_1_int`).
* If you'd like to see a visual of the agent planning in iGibson, set the command line argument `--behavior_mode simple`. If you want to run in headless mode without any visuals, leave the default (i.e `--behavior_mode headless`).
* Be sure to set `--plan_only_eval True`: this is necessary to account for the fact that the iGibson simulator is non-deterministic when saving and loading states (which is currently an unresolved bug).
* Example command: `python predicators/main.py --env behavior --approach oracle --option_model_name oracle_behavior --num_train_tasks 0 --num_test_tasks 1 --behavior_scene_name Pomaria_2_int --behavior_task_list "[opening_packages]" --seed 1000 --offline_data_planning_timeout 500.0 --timeout 500.0 --behavior_option_model_eval True --plan_only_eval True`.
