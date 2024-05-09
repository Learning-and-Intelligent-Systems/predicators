"""Tests for cogman.py."""

import time
from typing import Any, List

import numpy as np
import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.envs import get_or_create_env
from predicators.envs.cover import CoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver
from predicators.structs import Action, DefaultState


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


def testrun_episode_and_get_observations():
    """Tests for run_episode_and_get_observations()."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    cover_options = get_gt_options(env.get_name())
    task = env.get_task("test", 0)
    approach = create_approach("random_options", env.predicates, cover_options,
                               env.types, env.action_space, train_tasks)
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)
    (states, actions), solved, metrics = run_episode_and_get_observations(
        cogman, env, "test", 0, max_num_steps=5)
    assert not solved
    assert len(states) == 6
    assert len(actions) == 5
    assert "policy_call_time" in metrics
    assert metrics["policy_call_time"] > 0.0
    assert metrics["num_options_executed"] > 0.0

    # Test exceptions_to_break_on.
    def _value_error_policy(_):
        raise ValueError("mock error")

    class _MockApproach:

        def __init__(self, policy):
            self._policy = policy

        def solve(self, task, timeout):
            """Just use the given policy."""
            del task, timeout  # unused
            return self._policy

        def get_execution_monitoring_info(self) -> List[Any]:
            """Just return empty list."""
            return []

    class _CountingMonitor(utils.LoggingMonitor):

        def __init__(self):
            self.num_observations = 0

        def reset(self, train_or_test, task_idx):
            self.num_observations = 0

        def observe(self, obs, action):
            self.num_observations += 1

    approach = _MockApproach(_value_error_policy)
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)

    with pytest.raises(ValueError) as e:
        _, _, _ = run_episode_and_get_observations(cogman,
                                                   env,
                                                   "test",
                                                   0,
                                                   max_num_steps=5)
    assert "mock error" in str(e)

    monitor = _CountingMonitor()
    (states, _), _, _ = run_episode_and_get_observations(
        cogman,
        env,
        "test",
        0,
        max_num_steps=5,
        exceptions_to_break_on={ValueError},
        monitor=monitor)

    assert len(states) == 1
    assert monitor.num_observations == 1

    class _MockEnv:

        @staticmethod
        def reset(train_or_test, task_idx):
            """Reset the mock environment."""
            del train_or_test, task_idx  # unused
            return DefaultState

        @staticmethod
        def step(action):
            """Step the mock environment."""
            del action  # unused
            raise utils.EnvironmentFailure("mock failure")

        def get_observation(self):
            """Gets currrent observation in mock environment."""
            return DefaultState

        def goal_reached(self):
            """Goal never reached."""
            return False

    mock_env = _MockEnv()
    ones_policy = lambda _: Action(np.zeros(1, dtype=np.float32))
    approach = _MockApproach(ones_policy)
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)
    monitor = _CountingMonitor()
    (states, actions), _, _ = run_episode_and_get_observations(
        cogman,
        mock_env,
        "test",
        0,
        max_num_steps=5,
        exceptions_to_break_on={utils.EnvironmentFailure},
        monitor=monitor)
    assert len(states) == 1
    assert len(actions) == 0
    assert monitor.num_observations == 1

    # Test policy call time.
    def _policy(_):
        time.sleep(0.1)
        return Action(env.action_space.sample())

    approach = _MockApproach(_policy)
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)

    _, _, metrics = run_episode_and_get_observations(cogman,
                                                     env,
                                                     "test",
                                                     0,
                                                     max_num_steps=3)
    assert metrics["policy_call_time"] >= 3 * 0.1
    assert metrics["num_options_executed"] == 0

    # Test with monitor in case where an uncaught exception is raised.

    def _policy(_):
        raise ValueError("mock error")

    monitor = _CountingMonitor()
    approach = _MockApproach(_policy)
    cogman = CogMan(approach, perceiver, exec_monitor)
    cogman.reset(task)

    try:
        run_episode_and_get_observations(cogman,
                                         mock_env,
                                         "test",
                                         0,
                                         max_num_steps=3,
                                         monitor=monitor)
    except ValueError:
        pass
    assert monitor.num_observations == 1
