# predicators

Requirements for pushing code:
* You can't push directly to master. Make a PR and merge that in.
* To merge a PR, you have to pass 3 checks, all defined in `.github/workflows/predicators.yml`.
* The first check is unit testing, which verifies that tests pass and that code is adequately covered. To run locally: `pytest tests/ --cov=src/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered`, which will print out the lines that are uncovered in every file. The "100" here means that all lines in every file must be covered. If that turns out to be too stringent, we can decrease it later.
* The second check is static type checking, which uses Mypy to verify type annotations. To run locally: `mypy .`.
* The third check is the linter, which runs pylint with the custom config file `.predicators_pylintrc` in the root of this repository. Feel free to edit this file as necessary. To run locally: `pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc`.
