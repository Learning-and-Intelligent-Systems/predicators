"""Test cases for the base environment class."""

import pytest
from test_oracle_approach import ENV_NAME_AND_CLS

from predicators import utils
from predicators.envs import BaseEnv, create_new_env, get_or_create_env


def test_env_creation():
    """Tests for create_new_env() and get_or_create_env()."""
    utils.reset_config({"num_train_tasks": 5, "num_test_tasks": 5})
    for name, _ in ENV_NAME_AND_CLS:
        env = create_new_env(name, do_cache=True, use_gui=False)
        assert isinstance(env, BaseEnv)
        other_env = get_or_create_env(name)
        assert env is other_env
        train_tasks = env.get_train_tasks()
        for idx, train_task in enumerate(train_tasks):
            task = env.get_task("train", idx)
            assert train_task.init.allclose(task.init)
            assert train_task.goal == task.goal
        test_tasks = env.get_test_tasks()
        for idx, test_task in enumerate(test_tasks):
            task = env.get_task("test", idx)
            assert test_task.init.allclose(task.init)
            assert test_task.goal == task.goal
        with pytest.raises(ValueError):
            env.get_task("not a real task category", 0)
    with pytest.raises(NotImplementedError):
        create_new_env("Not a real env")
