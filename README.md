# predicators

## Repository Description

This codebase implements a framework for *bilevel planning with learned neuro-symbolic relational abstractions*, as described in the following papers:

1. [Learning Symbolic Operators for Task and Motion Planning](https://arxiv.org/abs/2103.00589). Silver*, Chitnis*, Tenenbaum, Kaelbling, Lozano-Perez. IROS 2021.
2. [Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning](https://arxiv.org/abs/2105.14074). Chitnis*, Silver*, Tenenbaum, Lozano-Perez, Kaelbling. IROS 2022.
3. [Learning Neuro-Symbolic Skills for Bilevel Planning](http://arxiv.org/abs/2206.10680). Silver, Athalye, Tenenbaum, Lozano-Perez, Kaelbling. CoRL 2022.
4. [Predicate Invention for Bilevel Planning](https://arxiv.org/abs/2203.09634). Silver*, Chitnis*, Kumar, McClinton, Lozano-Perez, Kaelbling, Tenenbaum. AAAI 2023.
5. [Embodied Active Learning of Relational State Abstractions for Bilevel Planning](https://arxiv.org/abs/2303.04912). Li, Silver. CoLLAs 2023.
6. [Learning Efficient Abstract Planning Models that Choose What to Predict](https://arxiv.org/abs/2208.07737). Kumar*, McClinton*, Chitnis, Silver, Lozano-Perez, Kaelbling. CoRL 2023.
7. [VisualPredicator: Learning Abstract World Models with Neuro-Symbolic Predicates for Robot Planning](https://arxiv.org/abs/2410.23156). Liang, Kumar, Tang, Weller, Tenenbaum, Silver, Henriques, Ellis. ICLR 2025.
8. [ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning](https://arxiv.org/abs/2509.26255). Liang, Nguyen, Yang, Li, Tenenbaum, Rasmussen, Weller, Tavares, Silver*, Ellis*. ICLR 2026.

### Code Structure

In `predicators/`, the environments are defined in the `envs/` directory, and the approaches (both learning-based and not) are defined in the `approaches/` directory. The core [NSRT learning algorithm](https://arxiv.org/abs/2105.14074) happens in `predicators/nsrt_learning/nsrt_learning_main.py`, which has the following steps:
* Segment data.
* Learn how many NSRTs we need, along with the symbolic operator components of each (parameters, preconditions, and effects).
* Learn options and annotate data with them.
* Learn samplers.
* Finalize the NSRTs.

Methods for predicate learning are implemented as Approaches (e.g., `predicators/approaches/grammar_search_invention_approach.py`), and may interface with the core structure of `predicators/nsrt_learning/nsrt_learning_main.py` in various ways.

A simple implementation of search-then-sample bilevel planning is provided in `predicators/planning.py`. This implementation uses the "SeSamE" strategy: SEarch-and-SAMple planning, then Execution.

## RoboDisco
Predicators ships **RoboDisco** (Robot Model Discovery Benchmark), a collection of PyBullet manipulation environments exposed through a standard [Gymnasium](https://gymnasium.farama.org/) API and suitable for world-model learning, causal discovery, and RL research independent of the planning framework. See [`predicators/envs/README.md`](predicators/envs/README.md) for the env list, install instructions, quick-start code, standalone API, and getting-started notebook.

## Installation
* This repository uses Python versions 3.10-3.11. We recommend 3.10.14.
* Run `pip install -e .` to install dependencies.

### Optional: real-robot execution (BabyRobotPredicator)

Driving a real Franka arm additionally needs
[BabyRobotPredicator](https://github.com/BasisResearch/BabyRobotPredicator), which
predicators carries as the git submodule `submodules/BabyRobotPredicator`. It is a
**private** repository and is deliberately **not** in `install_requires`, so:

* Everything except real-robot execution works without it. The submodule simply does
  not clone if you lack access.
* With access, check it out and install it from the submodule path:

  ```bash
  git submodule update --init submodules/BabyRobotPredicator
  pip install -e submodules/BabyRobotPredicator
  ```

Predicators then drives the arm in-process: it constructs babyrobot's `RealRobot`
and calls it like any Python object, in the same interpreter. Set `--real_robot_execute True`
to execute (and `--real_robot_dry True` to run the whole path with no arm).

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
    * `pytest -s tests/ --cov-config=.coveragerc --cov=predicators/ --cov=tests/ --cov-report=term-missing:skip-covered --durations=0` (in CI, this is split across 8 parallel shards using `pytest-split`)
    * `mypy . --config-file mypy.ini`
    * `pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc`
    * `./run_autoformat.sh`
* The first one is the unit testing check, which verifies that unit tests pass and that code is adequately covered. In CI, tests are split into 8 parallel groups via `pytest-split` and coverage is merged afterward.
* The second one is the static typing check, which uses Mypy to verify type annotations. If it doesn't work due to import errors, try `mypy -p predicators --config-file predicators/mypy.ini` from one directory up.
* The third one is the linter check, which runs Pylint with the custom config file `.predicators_pylintrc` in the root of this repository. Feel free to edit this file as necessary.
* The fourth one is the autoformatting check, which uses the custom config files `.style.yapf` and `.isort.cfg` in the root of this repository.
