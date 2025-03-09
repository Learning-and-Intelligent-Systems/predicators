# predicators_robocasa

## Predicators_Robocasa Intsallation Instructions
1. Follow instruction for robocasa from it's online documentation (In step 4, use fork [here](https://github.com/shaoyifei96/robocasa.git)). Don't forget step 5 to download dataset.[here](https://robocasa.ai/docs/introduction/installation.html)
1. Install this repo using the instructions below, also in the conda env (There should be nothing red appearing)
1. all three repos should be in a directory, in that directory, create .vscode folder and copy the launch.json file in this repo to that folder, making edits to the file, to enable debugging.
1. put ```export PYTHONHASHSEED=0``` in .bashrc
1. In VSCode, you should see a few debug profiles, the teleop profile should work as is.(Try complete the task, it is not easy). The predicator_robocasa main file should solve the task, and output a NSRT plan. It will then start executing the plan in a GUI. The door should be opened.

## Running inside docker
When running inside docker, this seems to work: ```export MUJOCO_GL=glfw && export PYOPENGL_PLATFORM=egl && cd /workspace/predicators_robocasa && python3 predicators/main.py --env robo_kitchen --approach grammar_search_invention --seed 0 --bilevel_plan_without_sim True --debug --excluded_predicates all```

and this [link](https://github.com/Andrew-Luo1/Mujoco-Headless-Tutorial/tree/main?tab=readme-ov-file)

[link](https://github.com/google-deepmind/mujoco/issues/572)

## Learning Predicates for RoboCasa (Currently buggy)
Robocasa is a mujoco based simulator. For each task, it provides 50 demonstrations, which can be used to learn to plan. This is the same as as in this paper [Learning Neuro-Symbolic Skills for Bilevel Planning](http://arxiv.org/abs/2206.10680), where the appendix explored the possibility of using demonstrations to a set of predicates first, by starting with the goal predicates. We are going to approach this problem in a similar way.
1. Defined the goal predicates for each task (starting with OpenSingleDoor). We can get these from the definition of tasks in each single stage ones. They need to be evaluated, with one of the following approaches: 1) evaluate the predicates during demo loading, since we have the simulator reset to the state, evaluating them is easy. 2) save the relevant states in the state variable, and use the ```abstract``` function as intended. The advantage is during predicate invention, we can use the state to evaluate new predicates.
1. Currently, the InContact is done with approach 1) since it is difficult to evaluate them online given each object has different mesh. Other predicates need to be evaluated online during planning and sampling, going to implement with apporoach 2). The values that need to go from mujoco state to predicator state are: 1) button states (coffee, microwave) 2) knob angle 3) faucet state (can be continuous or already in predicate form on/off, left/right) 4) door and drawer state 5) spatial relationship (in contact, not in contact) 6) position and orientation of the robot 
1. (Optionally) When inventing new predicates, keep ```InContact``` predicate since we are using that for trajectory segmentation anyway (TODO: Run Abalation including InContact and excluding InContact)
1. Run Grammar Invention




# Predicators

## Repository Description

This codebase implements a framework for *bilevel planning with learned neuro-symbolic relational abstractions*, as described in the following papers:

1. [Learning Symbolic Operators for Task and Motion Planning](https://arxiv.org/abs/2103.00589). Silver*, Chitnis*, Tenenbaum, Kaelbling, Lozano-Perez. IROS 2021.
2. [Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning](https://arxiv.org/abs/2105.14074). Chitnis*, Silver*, Tenenbaum, Lozano-Perez, Kaelbling. IROS 2022.
3. [Learning Neuro-Symbolic Skills for Bilevel Planning](http://arxiv.org/abs/2206.10680). Silver, Athalye, Tenenbaum, Lozano-Perez, Kaelbling. CoRL 2022.
4. [Predicate Invention for Bilevel Planning](https://arxiv.org/abs/2203.09634). Silver*, Chitnis*, Kumar, McClinton, Lozano-Perez, Kaelbling, Tenenbaum. AAAI 2023.
5. [Embodied Active Learning of Relational State Abstractions for Bilevel Planning](https://arxiv.org/abs/2303.04912). Li, Silver. CoLLAs 2023.
6. [Learning Efficient Abstract Planning Models that Choose What to Predict](https://arxiv.org/abs/2208.07737). Kumar*, McClinton*, Chitnis, Silver, Lozano-Perez, Kaelbling. CoRL 2023.

The codebase is still under active development. **Please contact <tslvr@mit.edu> or <njk@mit.edu> before attempting to use it for your own research.**

### Code Structure

In `predicators/`, the environments are defined in the `envs/` directory, and the approaches (both learning-based and not) are defined in the `approaches/` directory. The core [NSRT learning algorithm](https://arxiv.org/abs/2105.14074) happens in `predicators/nsrt_learning/nsrt_learning_main.py`, which has the following steps:
* Segment data.
* Learn how many NSRTs we need, along with the symbolic operator components of each (parameters, preconditions, and effects).
* Learn options and annotate data with them.
* Learn samplers.
* Finalize the NSRTs.

Methods for predicate learning are implemented as Approaches (e.g., `predicators/approaches/grammar_search_invention_approach.py`), and may interface with the core structure of `predicators/nsrt_learning/nsrt_learning_main.py` in various ways.

A simple implementation of search-then-sample bilevel planning is provided in `predicators/planning.py`. This implementation uses the "SeSamE" strategy: SEarch-and-SAMple planning, then Execution.

## Installation
* This repository uses Python versions 3.10-3.11. We recommend 3.10.14.
* Run `pip install -e .` to install dependencies.

## Instructions For Running Code

### `PYTHONHASHSEED`
Our code assumes that python hashing is deterministic between processes, which is [not true by default](https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program).
Please make sure to `export PYTHONHASHSEED=0` when running the code. You can add this line to your bash profile, or prepend `export PYTHONHASHSEED=0` to any command line call, e.g., `export PYTHONHASHSEED=0 python predicators/main.py --env ...`.

### Locally
* (recommended) Make a new virtual env or conda env.
* Run, e.g., `python predicators/main.py --env cover --approach oracle --seed 0` to run the system.

### Running Experiments on Supercloud
See [these instructions](supercloud.md).

## Instructions For Contributing
* Run `pip install -e .[develop]` to install all dependencies for development.
* You can't push directly to master. Make a new branch in this repository (don't use a fork, since that will not properly trigger the checks when you make a PR). When your code is ready for review, make a PR and request reviews from the appropriate people.
* To merge a PR, you need at least one approval, and you have to pass the 4 checks defined in `.github/workflows/predicators.yml`, which you can run locally in one line via `./scripts/run_checks.sh`, or individually as follows:
    * `pytest -s tests/ --cov-config=.coveragerc --cov=predicators/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered --durations=0`
    * `mypy . --config-file mypy.ini`
    * `pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc`
    * `./run_autoformat.sh`
* The first one is the unit testing check, which verifies that unit tests pass and that code is adequately covered. The "100" means that all lines in every file must be covered.
* The second one is the static typing check, which uses Mypy to verify type annotations. If it doesn't work due to import errors, try `mypy -p predicators --config-file predicators/mypy.ini` from one directory up.
* The third one is the linter check, which runs Pylint with the custom config file `.predicators_pylintrc` in the root of this repository. Feel free to edit this file as necessary.
* The fourth one is the autoformatting check, which uses the custom config files `.style.yapf` and `.isort.cfg` in the root of this repository.
