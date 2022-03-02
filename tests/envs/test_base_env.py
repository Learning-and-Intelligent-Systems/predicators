"""Test cases for the base environment class."""

import pytest
from predicators.src.envs import BaseEnv, create_env, get_cached_env_instance
from predicators.src import utils


def test_create_env():
    """Tests for create_env() and get_cached_env_instance()."""
    utils.reset_config()
    for name in [
            "cover",
            "cover_typed_options",
            "cover_hierarchical_types",
            "cover_regrasp",
            "cluttered_table",
            "blocks",
            "playroom",
            "painting",
            "tools",
            "repeated_nextto",
            "cover_multistep_options",
            "cover_multistep_options_fixed_tasks",
    ]:
        env = create_env(name)
        assert isinstance(env, BaseEnv)
        other_env = get_cached_env_instance(name)
        assert env is other_env
        train_tasks = env.get_train_tasks()
        for idx, train_task in enumerate(train_tasks):
            task = env.get_task("train", idx)
            assert train_task.init.allclose(task.init)
            task = env.get_task("train", idx)
            assert train_task.goal == task.goal
        test_tasks = env.get_test_tasks()
        for idx, test_task in enumerate(test_tasks):
            task = env.get_task("test", idx)
            assert test_task.init.allclose(task.init)
            task = env.get_task("test", idx)
            assert test_task.goal == task.goal
        with pytest.raises(ValueError):
            env.get_task("not a real task category", 0)
    with pytest.raises(NotImplementedError):
        create_env("Not a real env")
