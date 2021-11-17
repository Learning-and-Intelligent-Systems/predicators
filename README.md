# predicators

Instructions for running code:
* This repository uses Python versions 3.8+.
* Run `pip install -r requirements.txt` to install dependencies.
* Make sure the parent of the repository is on your PYTHONPATH.
* Run, e.g., `python src/main.py --env cover --approach oracle --seed 0` to run the system.

Instructions for contributing:
* In addition to the packages in `requirements.txt`, please `pip install` the following packages if you want to contribute to the repository: `pytest-cov>=2.12.1` and `pytest-pylint>=0.18.0`. Also, install `mypy` from source: `pip install -U git+git://github.com/python/mypy.git@@9a10967fdaa2ac077383b9eccded42829479ef31`. (Note: if [this mypy issue](https://github.com/python/mypy/issues/5485) gets resolved, we can install from head again.)
* You can't push directly to master. Make a PR and merge that in.
* To merge a PR, you have to pass 3 checks, all defined in `.github/workflows/predicators.yml`.
* The unit testing check verifies that tests pass and that code is adequately covered. To run locally: `pytest -s tests/ --cov-config=.coveragerc --cov=src/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered`, which will print out the lines that are uncovered in every file. The "100" here means that all lines in every file must be covered. If that turns out to be too stringent, we can decrease it later.
* The static typing check uses Mypy to verify type annotations. To run locally: `mypy .`. If this doesn't work due to import errors, try `mypy -p predicators` from one directory up.
* The linter check runs pylint with the custom config file `.predicators_pylintrc` in the root of this repository. Feel free to edit this file as necessary. To run locally: `pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc`.
