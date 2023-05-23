# Spot ROS2 in predicators

## Installation
* (recommended) Make a new virtual env or conda env.
* Install [bdai Docker](https://www.notion.so/theaiinstitute/Docker-Build-321dc44d19424d0a847cc4a3e81e05d4)
* This repository uses Python versions 3.10+.

## Instructions For Running Code

### Locally
* Start the docker container `bdai docker start`
* cd into Spot ROS2 Simple Walk Forward Example repo (potentially located here: /workspaces/bdai/ws/src/external/spot_ros2/examples/simple_walk_forward)
* Run `pip install -e .` to install dependencies.
* cd into prediators repo in bdai (potentially located here: /workspaces/bdai/ws/src/external/predicators)
* Run `pip install -e .` to install dependencies.
* Run, e.g., `python spot_os2_examples/walk_forward.py` to run the system.