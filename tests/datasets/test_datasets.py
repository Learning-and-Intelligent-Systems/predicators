"""Test cases for dataset generation."""

import pytest
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv, ClutteredTableEnv
from predicators.src.structs import Dataset
from predicators.src import utils


def test_demo_dataset():
    """Test demo-only dataset creation with Covers env."""
    # Test that data does not contain options since
    # option_learner is not "no_learning"
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
    })
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "option_learner": "arbitrary_dummy",
        "seed": 123,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 7
    assert len(dataset.trajectories[0].states) == 3
    assert len(dataset.trajectories[0].actions) == 2
    for traj in dataset.trajectories:
        assert traj.is_demo
        for action in traj.actions:
            assert not action.has_option()
    # Test that data contains options since option_learner is "no_learning"
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "option_learner": "no_learning",
        "seed": 123,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 7
    assert len(dataset.trajectories[0].states) == 3
    assert len(dataset.trajectories[0].actions) == 2
    for traj in dataset.trajectories:
        assert traj.is_demo
        for action in traj.actions:
            assert action.has_option()
    with pytest.raises(AssertionError):
        _ = dataset.annotations
    utils.update_config({
        "offline_data_method": "not a real method",
    })
    with pytest.raises(NotImplementedError):
        create_dataset(env, train_tasks)


def test_demo_replay_dataset():
    """Test demo+replay dataset creation with Covers env."""
    # Test that data contains options since
    # option_learner is "no_learning"
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "no_learning",
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 5 + 3
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1
    num_demos = 0
    for traj in dataset.trajectories:
        num_demos += int(traj.is_demo)
        for action in traj.actions:
            assert action.has_option()
    assert num_demos == 5
    # Test that data does not contain options since
    # option_learner is not "no_learning"
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "arbitrary_dummy",
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 5 + 3
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1
    num_demos = 0
    for traj in dataset.trajectories:
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
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1


def test_demo_nonoptimal_replay_dataset():
    """Test demo+nonoptimalreplay dataset creation with Covers env."""
    # Note that the planning timeout is intentionally set low enough that we
    # cover some failures to plan from the replays, but not so low that
    # planning always fails. Also the number of replays is set high enough
    # that we consistently cover the failure case.
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+nonoptimalreplay",
        "offline_data_planning_timeout": 1e-1,
        "offline_data_num_replays": 50,
        "option_learner": "no_learning",
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 5 + 50
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1


def test_dataset_with_annotations():
    """Test the creation of a Dataset with annotations."""
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "no_learning",
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    trajectories = create_dataset(env, train_tasks).trajectories
    # The annotations and trajectories need to be the same length.
    with pytest.raises(AssertionError):
        dataset = Dataset(trajectories, [])
    annotations = ["label" for _ in trajectories]
    dataset = Dataset(trajectories, annotations)
    assert dataset.annotations == annotations
