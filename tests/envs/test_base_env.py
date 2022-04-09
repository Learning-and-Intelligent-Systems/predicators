"""Test cases for the base environment class."""

import pytest

from predicators.src import utils
from predicators.src.envs import BaseEnv, create_new_env, get_or_create_env


def test_env_creation():
    """Tests for create_new_env() and get_or_create_env()."""
    utils.reset_config({"num_train_tasks": 5, "num_test_tasks": 5})
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
            "repeated_nextto_single_option",
            "repeated_nextto_painting",
            "cover_multistep_options",
            "cover_multistep_options_fixed_tasks",
            "cover_multistep_options_holding",
            "pybullet_blocks",
            "pddl_blocks_fixed_tasks",
            "pddl_blocks_procedural_tasks",
            "touch_point",
            "four_rooms",
            "four_rooms_generalize",
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
