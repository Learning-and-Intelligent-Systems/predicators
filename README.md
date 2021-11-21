# predicators

## Repository Description

This codebase provides a concrete implementation of [Neuro-Symbolic Relational Transition Models](https://arxiv.org/abs/2105.14074) for task and motion planning. **Several features are concurrently under active development -- please contact <tslvr@mit.edu> and <ronuchit@mit.edu> before attempting to use it for your own research.**

The scope of this codebase extends far beyond the scope of the paper linked above. In particular, this codebase aims to ultimately provide an integrated system for learning the ingredients of search-then-sample task and motion planning. That includes: options, predicates, operators, and samplers.

### Code structure

In `src/`, the environments are defined in the `envs/` directory, and the approaches (both learning-based and not) are defined in the `approaches/` directory. The core NSRT learning algorithm happens in `src/nsrt_learning.py`, which has the following steps:
* Segment data based on changes in predicates.
* Learn how many NSRTs we need, along with the symbolic operator components of each (parameters, preconditions, and effects).
* Learn options and annotate data with them.
* Learn samplers.
* Finalize the NSRTs.

Methods for predicate learning are implemented as Approaches (e.g., `src/approaches/iterative_invention_approach.py`), and may interface with the core structure of `src/nsrt_learning.py` in various ways. Meanwhile, sampler learning and option learning are implemented functionally, as part of the algorithm above.

A simple implementation of search-then-sample task and motion planning is provided in `src/planning.py`. This implementation uses the "SeSamE" strategy: SEarch-and-SAMple planning, then Execution.

## Installation
### Pip
* This repository uses Python versions 3.8+.
* Run `pip install -r requirements.txt` to install dependencies.

### Poetry
Run the below commands from a folder that will contain the `predicators` folder after install.

```
git clone https://github.com/Learning-and-Intelligent-Systems/predicators.git
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/install-poetry.py | python -
cd PGMax
poetry shell
poetry install
```

## Instructions For Running Code
### Pip
* (recommended) Make a new virtual env or conda env
* Make sure the parent of the repository is on your PYTHONPATH.
* Run, e.g., `python src/main.py --env cover --approach oracle --seed 0` to run the system.

### Poetry
* Activate your poetry shell from within the repository with `poetry shell`
* Run, e.g., `python src/main.py --env cover --approach oracle --seed 0` to run the system.

## Instructions For Contributing
* You can't push directly to master. Make a PR and merge that in.
* To merge a PR, you have to pass 3 checks, all defined in `.github/workflows/predicators.yml`.
* The unit testing check verifies that tests pass and that code is adequately covered. To run locally: `pytest -s tests/ --cov-config=.coveragerc --cov=src/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered`, which will print out the lines that are uncovered in every file. The "100" here means that all lines in every file must be covered. If that turns out to be too stringent, we can decrease it later.
* The static typing check uses Mypy to verify type annotations. To run locally: `mypy .`. If this doesn't work due to import errors, try `mypy -p predicators` from one directory up.
* The linter check runs pylint with the custom config file `.predicators_pylintrc` in the root of this repository. Feel free to edit this file as necessary. To run locally: `pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc`.

### Using pip
* In addition to the packages in `requirements.txt`, please `pip install` the following packages if you want to contribute to the repository: `pytest-cov>=2.12.1` and `pytest-pylint>=0.18.0`. Also, install `mypy` from source: `pip install -U git+git://github.com/python/mypy.git@9a10967fdaa2ac077383b9eccded42829479ef31`. (Note: if [this mypy issue](https://github.com/python/mypy/issues/5485) gets resolved, we can install from head again.)

### Using poetry
* If your contribution introduces a new dependency, be sure to add it to Poetry ([instructions here](https://python-poetry.org/docs/cli/#add)). In brief, if you add a new dependency that users will require (e.g. a new deep learning package, or something that affects some functionality of the codebaase), run: `poetry add <package-name>`. If you add a new dependency that's really only for developers (e.g. new linter or formatter), run: `poetry add -D <package-name>`.