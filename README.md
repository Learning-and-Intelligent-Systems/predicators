# predicators

## Repository Description

This codebase provides a concrete implementation of [Neuro-Symbolic Relational Transition Models](https://arxiv.org/abs/2105.14074) for task and motion planning. **Several features are concurrently under active development -- please contact <tslvr@mit.edu> and <ronuchit@mit.edu> before attempting to use it for your own research.**

The scope of this codebase extends far beyond the scope of the paper linked above. In particular, this codebase aims to ultimately provide an integrated system for learning the ingredients of search-then-sample task and motion planning. That includes: options, predicates, operators, and samplers.

### Code structure

In `src/`, the environments are defined in the `envs/` directory, and the approaches (both learning-based and not) are defined in the `approaches/` directory. The core NSRT learning algorithm happens in `src/nsrt_learning/nsrt_learning_main.py`, which has the following steps:
* Segment data based on changes in predicates.
* Learn how many NSRTs we need, along with the symbolic operator components of each (parameters, preconditions, and effects).
* Learn options and annotate data with them.
* Learn samplers.
* Finalize the NSRTs.

Methods for predicate learning are implemented as Approaches (e.g., `src/approaches/grammar_search_invention_approach.py`), and may interface with the core structure of `src/nsrt_learning/nsrt_learning_main.py` in various ways.

A simple implementation of search-then-sample task and motion planning is provided in `src/planning.py`. This implementation uses the "SeSamE" strategy: SEarch-and-SAMple planning, then Execution.

## Installation
### Pip
* This repository uses Python versions 3.8+.
* Run `pip install -r requirements.txt` to install dependencies.

### (Optional) BEHAVIOR Installation
This repository and some of the functionality it enables are integrated with the [BEHAVIOR benchmark of tasks](https://behavior.stanford.edu/benchmark-guide) simulated with the [iGibson simulator](https://github.com/StanfordVL/iGibson). To install iGibson with BEHAVIOR, follow [**Option 1** from these instructions](https://stanfordvl.github.io/behavior/installation.html). Importantly, note that you need to use our forks of the [iGibson](https://github.com/Learning-and-Intelligent-Systems/iGibson) and [bddl](https://github.com/Learning-and-Intelligent-Systems/bddl) repositories. Thus, instead of steps 2a. and 2b. of Option 1 in the linked instructions, run:
```
git clone https://github.com/Learning-and-Intelligent-Systems/iGibson.git --recursive
git clone https://github.com/Learning-and-Intelligent-Systems/bddl.git
```

## Instructions For Running Code

### `PYTHONHASHSEED`
Our code assumes that python hashing is deterministic between processes, which is [not true by default](https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program).
Please make sure to `export PYTHONHASHSEED=0` when running the code. You can add this line to your bash profile, or prepend `export PYTHONHASHSEED=0` to any command line call, e.g., `export PYTHONHASHSEED=0 python src/main.py --env ...`.

### Locally
* (recommended) Make a new virtual env or conda env.
* Make sure the parent of the repository is on your PYTHONPATH.
* Run, e.g., `python src/main.py --env cover --approach oracle --seed 0` to run the system.

### Running Experiments on Supercloud
* Log into supercloud (ask Rohan if you don't know how to do this).
* Go into the `predicators` folder and `git pull` if necessary.
* Edit `./analysis/run_supercloud_experiments.sh` as desired, and run that script to launch parallelized jobs.
* Monitor with `squeue`, or cancel jobs with `scancel` (standard Slurm commands).
* When all jobs are done, run `python analysis/analyze_supercloud_experiments.py` (still on supercloud) to print out the results table.

### Running Experiments on BEHAVIOR
* Currently, only the `oracle` approach is implemented to integrate with BEHAVIOR.
* Note that you'll probably want to provide the command line arguments `--timeout 1000`, `--max_num_steps_check_policy 1000` to prevent early stopping.
* Set `--option_model_name behavior_oracle` to use the behavior option model and speed up planning by a significant factor.
* Set `--behavior_task_name` to the name of the particular bddl task you'd like to run (e.g. `re-shelving_library_books`).
* Set `--behavior_scene_name` to the name of the house setting (e.g. `Pomaria_1_int`) you want to try running the particular task in. Note that not all tasks are available in all houses (e.g. `re-shelving_library_books` might only be available with `Pomaria_1_int`).
* `--behavior_randomize_init_state` can be set to True if you want to generate multiple different initial states that correspond to the BDDL init conditions of a particular task.
* If you'd like to see a visual of the agent planning in iGibson, set the command line argument `--behavior_mode simple`. If you want to run in headless mode without any visuals, leave the default (i.e `--behavior_mode headless`).
* Example command: `python src/main.py --env behavior --approach oracle --seed 0 --timeout 1000 --sesame_max_samples_per_step 20 --behavior_mode simple --max_num_steps_check_policy 1000 --option_model_name behavior_oracle --num_train_tasks 2 --num_test_tasks 2 --behavior_randomize_init_state True --behavior_scene_name Pomaria_1_int --behavior_task_name re-shelving_library_books`.

## Instructions For Contributing
* You can't push directly to master. Make a PR and merge that in.
* To merge a PR, you have to pass 4 checks, all defined in `.github/workflows/predicators.yml`.
* The unit testing check verifies that tests pass and that code is adequately covered. To run locally: `pytest -s tests/ --cov-config=.coveragerc --cov=src/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered`, which will print out the lines that are uncovered in every file. The "100" here means that all lines in every file must be covered.
* The static typing check uses Mypy to verify type annotations. To run locally: `mypy . --config-file mypy.ini`. If this doesn't work due to import errors, try `mypy -p predicators --config-file predicators/mypy.ini` from one directory up.
* The linter check runs pylint with the custom config file `.predicators_pylintrc` in the root of this repository. Feel free to edit this file as necessary. To run locally: `pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc`.
* The autoformatting check uses the custom `.style.yapf` in this repo. You can run the autoformatter locally with `yapf -i -r --style .style.yapf . && docformatter -i -r .`.
* In addition to the packages in `requirements.txt`, please `pip install` the following packages if you want to contribute to the repository: `pytest-cov>=2.12.1`, `pytest-pylint>=0.18.0`, `yapf` and `docformatter`. Also, install `mypy` from source: `pip install -U git+git://github.com/python/mypy.git@9a10967fdaa2ac077383b9eccded42829479ef31`. (Note: if [this mypy issue](https://github.com/python/mypy/issues/5485) gets resolved, we can install from head again.)
