"""Test cases for dataset generation.
"""

import pytest
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv, ClutteredTableEnv
from predicators.src import utils


def test_demo_dataset():
    """Test demo-only dataset creation with Covers env.
    """
    # Test that data does not contain options since do_option_learning is True
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
    })
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "do_option_learning": True,
        "seed": 123,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    assert len(dataset) == 7
    assert len(dataset[0].states) == 3
    assert len(dataset[0].actions) == 2
    for traj in dataset:
        assert traj.is_demo
        for action in traj.actions:
            assert not action.has_option()
    # Test that data contains options since do_option_learning is False
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "do_option_learning": False,
        "seed": 123,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    assert len(dataset) == 7
    assert len(dataset[0].states) == 3
    assert len(dataset[0].actions) == 2
    for traj in dataset:
        assert traj.is_demo
        for action in traj.actions:
            assert action.has_option()
    utils.update_config({
        "offline_data_method": "not a real method",
    })
    with pytest.raises(NotImplementedError):
        create_dataset(env, train_tasks)


def test_demo_replay_dataset():
    """Test demo+replay dataset creation with Covers env.
    """
    # Test that data contains options since do_option_learning is False
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "do_option_learning": False,
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    assert len(dataset) == 5 + 3
    assert len(dataset[-1].states) == 2
    assert len(dataset[-1].actions) == 1
    num_demos = 0
    for traj in dataset:
        num_demos += int(traj.is_demo)
        for action in traj.actions:
            assert action.has_option()
    assert num_demos == 5
    # Test that data does not contain options since do_option_learning is True
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "do_option_learning": True,
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    assert len(dataset) == 5 + 3
    assert len(dataset[-1].states) == 2
    assert len(dataset[-1].actions) == 1
    num_demos = 0
    for traj in dataset:
        num_demos += int(traj.is_demo)
        for action in traj.actions:
            assert not action.has_option()
    assert num_demos == 5
    # Test cluttered table data collection
    utils.update_config({
        "env": "cluttered_table",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 10,
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = ClutteredTableEnv()
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    assert len(dataset[-1].states) == 2
    assert len(dataset[-1].actions) == 1
