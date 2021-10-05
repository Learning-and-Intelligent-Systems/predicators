"""Test cases for dataset generation.
"""

from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv
from predicators.src import utils


def test_demo_dataset():
    """Test demo-only dataset creation with Covers env.
    """
    utils.update_config({
        "env": "cover",
        "num_train_tasks": 5,
        "max_num_steps_check_policy": 10,
    })
    env = CoverEnv()
    config = {
        "method": "demo",  # demo or demo+replay
        "planning_timeout": 5,
    }
    dataset = create_dataset(env, env.get_train_tasks(), config)
    assert len(dataset) == 5
    assert len(dataset[0]) == 2
    assert len(dataset[0][0]) == 3
    assert len(dataset[0][1]) == 2
