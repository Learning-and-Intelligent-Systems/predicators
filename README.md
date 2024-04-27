# predicators

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
* This repository uses Python versions 3.10+. We recommend 3.10.14.
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
