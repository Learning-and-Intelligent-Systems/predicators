"""Test cases for the base environment class."""

import pytest
from predicators.src.envs import BaseEnv, create_new_env, get_or_create_env
from predicators.src import utils


def test_env_creation():
    """Tests for create_new_env() and get_or_create_env()."""
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
            "pybullet_blocks",
    ]:
        env = create_new_env(name, do_cache=True)
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
