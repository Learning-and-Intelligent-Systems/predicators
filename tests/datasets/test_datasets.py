"""Test cases for dataset generation.
"""

import pytest
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv
from predicators.src import utils


def test_demo_dataset():
    """Test demo-only dataset creation with Covers env.
    """
    # Test that data does not contain options since approach is random
    utils.update_config({
        "env": "cover",
        "approach": "random",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "seed": 0,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    dataset = create_dataset(env)
    assert len(dataset) == 7
    assert len(dataset[0]) == 2
    assert len(dataset[0][0]) == 3
    assert len(dataset[0][1]) == 2
    for _, actions in dataset:
        for action in actions:
            assert not action.has_option()
    # Test that data contains options since approach is trivial_learning
    utils.update_config({
        "env": "cover",
        "approach": "trivial_learning",
        "seed": 0,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    dataset = create_dataset(env)
    assert len(dataset) == 7
    assert len(dataset[0]) == 2
    assert len(dataset[0][0]) == 3
    assert len(dataset[0][1]) == 2
    for _, actions in dataset:
        for action in actions:
            assert action.has_option()
    utils.update_config({
        "offline_data_method": "not a real method",
    })
    with pytest.raises(NotImplementedError):
        create_dataset(env)


def test_demo_replay_dataset():
    """Test demo+replay dataset creation with Covers env.
    """
    utils.update_config({
        "env": "cover",
        "approach": "trivial_learning",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "seed": 0,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    dataset = create_dataset(env)
    assert len(dataset) == 5 + 3
    assert len(dataset[-1]) == 2
    assert len(dataset[-1][0]) == 2
    assert len(dataset[-1][1]) == 1
    for _, actions in dataset:
        for action in actions:
            assert action.has_option()
    # Test that data does not contain options since approach is random
    utils.update_config({
        "env": "cover",
        "approach": "random",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "seed": 0,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    dataset = create_dataset(env)
    assert len(dataset) == 5 + 3
    assert len(dataset[-1]) == 2
    assert len(dataset[-1][0]) == 2
    assert len(dataset[-1][1]) == 1
    for _, actions in dataset:
        for action in actions:
            assert not action.has_option()
