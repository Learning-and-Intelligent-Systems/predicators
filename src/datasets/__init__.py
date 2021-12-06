"""Create offline datasets for training, given a set of training tasks
for an environment.
"""

from typing import List
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.settings import CFG


def create_dataset(env: BaseEnv, train_tasks: List[Task]) -> Dataset:
    """Create offline datasets for training, given a set of training tasks
    for an environment.
    """
    if CFG.offline_data_method == "demo":
        return create_demo_data(env, train_tasks)
    if CFG.offline_data_method == "demo+replay":
        return create_demo_replay_data(env, train_tasks)
    raise NotImplementedError("Unrecognized dataset method.")
