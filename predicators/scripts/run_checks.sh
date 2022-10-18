#!/bin/bash

echo "Running autoformatting."
yapf -i -r --style .style.yapf --exclude '**/third_party' predicators
yapf -i -r --style .style.yapf scripts
yapf -i -r --style .style.yapf tests
docformatter -i -r . --exclude venv predicators/third_party
isort .
echo "Autoformatting complete."

echo "Running type checking."
mypy . --config-file mypy.ini
if [ $? -eq 0 ]; then
    echo "Type checking passed."
else
    echo "Type checking failed! Terminating check script early."
    exit
fi

echo "Running linting."
pytest . --pylint -m pylint --pylint-rcfile=.predicators_pylintrc
if [ $? -eq 0 ]; then
    echo "Linting passed."
else
    echo "Linting failed! Terminating check script early."
    exit
fi

echo "Running unit tests."
pytest -s tests/ --cov-config=.coveragerc --cov=predicators/ --cov=tests/ --cov-fail-under=100 --cov-report=term-missing:skip-covered --durations=10
if [ $? -eq 0 ]; then
    echo "Unit tests passed."
else
    echo "Unit tests failed!"
    exit
fi

echo "All checks passed!"
