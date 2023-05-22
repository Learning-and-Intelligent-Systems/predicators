# Spot ROS2 in predicators

**TODO Fix This and make more extensive

## Installation
* (recommended) Make a new virtual env or conda env.
* Install bdai Docker
* This repository uses Python versions 3.10+.

## Instructions For Running Code

### Locally
* Start the docker container `bdai docker start`
* cd into prediators repo in bdai
* Run `pip install -e .` to install dependencies.
* Run, e.g., `python predicators/main.py --env cover --approach oracle --seed 0` to run the system.