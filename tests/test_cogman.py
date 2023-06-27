"""Tests for cogman.py."""

import numpy as np
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan
from predicators.envs import get_or_create_env
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver


@pytest.mark.parametrize("exec_monitor_name", ["trivial", "mpc"])
def test_cogman(exec_monitor_name):
    """Tests for CogMan()."""
    env_name = "cover"
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 0,
        "num_test_tasks": 2,
    })
    env = get_or_create_env(env_name)
    env_train_tasks = env.get_train_tasks()
    env_test_tasks = env.get_test_tasks()
    train_tasks = [t.task for t in env_train_tasks]
    options = get_gt_options(env.get_name())
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor(exec_monitor_name)
    approach = create_approach("random_actions", env.predicates, options,
                               env.types, env.action_space, train_tasks)
    cogman = CogMan(approach, perceiver, exec_monitor)
    env.reset("test", 0)
    env_task = env_test_tasks[0]
    cogman.reset(env_task)
    obs = env_task.init_obs
    act = cogman.step(obs)
    assert env.action_space.contains(act.arr)
    next_obs = env.step(act)
    next_act = cogman.step(next_obs)
    assert not np.allclose(act.arr, next_act.arr)


def test_cogman_with_expected_atoms_monitor():
    """Tests for CogMan() with bilevel planning and the 'expected_atoms'
    execution monitor."""
    env_name = "cover"
    utils.reset_config({
        "env": env_name,
        "num_train_tasks": 0,
        "num_test_tasks": 2,
        "bilevel_plan_without_sim": True,
        "approach": "oracle"
    })
    env = get_or_create_env(env_name)
    env_train_tasks = env.get_train_tasks()
    env_test_tasks = env.get_test_tasks()
    train_tasks = [t.task for t in env_train_tasks]
    options = get_gt_options(env.get_name())
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("expected_atoms")
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space, train_tasks)
    cogman = CogMan(approach, perceiver, exec_monitor)
    env.reset("test", 0)
    env_task = env_test_tasks[0]
    cogman.reset(env_task)
    obs = env_task.init_obs
    act = cogman.step(obs)
    assert env.action_space.contains(act.arr)
    next_obs = env.step(act)
    next_act = cogman.step(next_obs)
    assert not np.allclose(act.arr, next_act.arr)
