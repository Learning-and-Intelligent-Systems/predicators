"""Create offline datasets.
"""

from predicators.src.envs import BaseEnv
from predicators.src.structs import Dataset
from predicators.src.datasets.demo_only import create_demo_data
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.settings import CFG


def create_dataset(env: BaseEnv) -> Dataset:
    """Create offline datasets.
    """
    if CFG.offline_data_method == "demo":
        return create_demo_data(env)
    if CFG.offline_data_method == "demo+replay":
        return create_demo_replay_data(env)
    raise NotImplementedError("Unrecognized dataset method.")
