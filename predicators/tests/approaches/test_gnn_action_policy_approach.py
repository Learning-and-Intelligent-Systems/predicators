"""Test cases for the GNN action policy approach."""

import pytest

from predicators.src import utils
from predicators.src.approaches import create_approach
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.settings import CFG


def test_gnn_action_policy_approach():
    """Tests for GNNActionPolicyApproach class."""
    utils.reset_config({
        "env": "cover",
        # Include replay data for coverage. It will be ignored.
        "offline_data_method": "demo+replay",
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "gnn_num_epochs": 20,
        "gnn_do_normalization": True,
        "horizon": 10
    })
    env = create_new_env("cover")
    train_tasks = env.get_train_tasks()
    approach = create_approach("gnn_action_policy", env.predicates,
                               env.options, env.types, env.action_space,
                               train_tasks)
    dataset = create_dataset(env, train_tasks, env.options)
    assert approach.is_learning_based
    task = env.get_test_tasks()[0]
    with pytest.raises(AssertionError):  # haven't learned yet!
        approach.solve(task, timeout=CFG.timeout)
    approach.learn_from_offline_dataset(dataset)
    policy = approach.solve(task, timeout=CFG.timeout)
    act = policy(task.init)
    assert env.action_space.contains(act.arr)
    # Test predictions by executing policy.
    utils.run_policy_with_simulator(policy,
                                    env.simulate,
                                    task.init,
                                    task.goal_holds,
                                    max_num_steps=CFG.horizon)
    # Test loading.
    approach2 = create_approach("gnn_action_policy", env.predicates,
                                env.options, env.types, env.action_space,
                                train_tasks)
    approach2.load(online_learning_cycle=None)
