"""Create offline datasets by collecting demonstrations and replaying.
"""
from typing import List
from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset, Task
from predicators.src.datasets.demo_only import create_demo_data


def create_demo_replay_data(env: BaseEnv, train_tasks: List[Task],
                            data_config: dict) -> Dataset:
    """Create offline datasets by collecting demos and replaying.
    """
    import ipdb; ipdb.set_trace()
