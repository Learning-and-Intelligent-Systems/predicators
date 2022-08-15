# Learning Operators with Ignore Effects for Bilevel Planning in Continuous Domains

Anonymized code for AAAI 2023 submission.

### Code Structure
Note that this codebase is part of a larger effort to learn the various different componenets of bilevel planning: our project only uses a small part of it.

In `src/`, the environments are defined in the `envs/` directory, and the approaches (both learning-based and not) are defined in the `approaches/` directory. The core operator and sampler learning algorithm happens in `src/nsrt_learning/nsrt_learning_main.py`, which has the following steps:
* Segment data based on changes in options (i.e controllers) used.
* Learn symbolic operators.
* Learn samplers.
* Finalize the NSRTs.

Our core approach ("Learning Operators by Backchaining") is used to perform the second step listed above. It can be found in `src/nsrt_learning/strips_learning/gen_to_spec_learner.py:BackchainingSTRIPSLearner`. Other baseline approaches that learn symbolic operators can also be found under `src/nsrt_learning/strips_learning/`.

Search-then-sample bilevel planning is in `src/planning.py`.

## Installation
### Pip
* This repository uses Python versions 3.8+.
* Run `pip install -r requirements.txt` to install dependencies.

## Instructions For Running Code

### `PYTHONHASHSEED`
Our code assumes that python hashing is deterministic between processes, which is [not true by default](https://stackoverflow.com/questions/30585108/disable-hash-randomization-from-within-python-program).
Please make sure to `export PYTHONHASHSEED=0` when running the code. You can add this line to your bash profile, or prepend `export PYTHONHASHSEED=0` to any command line call, e.g., `export PYTHONHASHSEED=0 python src/main.py --env ...`.

### Single Runs
* (recommended) Make a new virtual env or conda env.
* Make sure the parent of the repository is on your PYTHONPATH.
* Run, e.g., `python src/main.py --env screws --approach oracle --seed 0` to run the system and verify that your installation is correct.

### Reproducing Results
* See `scripts/run_main_experiments.sh`.
