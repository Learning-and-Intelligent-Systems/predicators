"""Test cases for the base environment class."""

import json
import tempfile
from unittest.mock import patch

import pytest

import predicators.envs
from predicators import utils
from predicators.envs import BaseEnv, create_new_env, get_or_create_env
from tests.approaches.test_oracle_approach import ENV_NAME_AND_CLS

_MODULE_PATH = predicators.envs.__name__


def test_env_creation():
    """Tests for create_new_env() and get_or_create_env()."""
    utils.reset_config({"num_train_tasks": 5, "num_test_tasks": 5})
    for name, _ in ENV_NAME_AND_CLS:
        env = create_new_env(name, do_cache=True, use_gui=False)
        assert isinstance(env, BaseEnv)
        other_env = get_or_create_env(name)
        assert env is other_env
        train_tasks = [t.task for t in env.get_train_tasks()]
        for idx, train_task in enumerate(train_tasks):
            task = env.get_task("train", idx)
            assert train_task.init.allclose(task.init)
            assert train_task.goal == task.goal
        test_tasks = [t.task for t in env.get_test_tasks()]
        for idx, test_task in enumerate(test_tasks):
            task = env.get_task("test", idx)
            assert test_task.init.allclose(task.init)
            assert test_task.goal == task.goal
        with pytest.raises(ValueError):
            env.get_task("not a real task category", 0)
    with pytest.raises(NotImplementedError):
        create_new_env("Not a real env")


@pytest.mark.parametrize("env_name", ("cover", "sandwich"))
def test_load_task_from_json(env_name):
    """Tests for env._load_task_from_json()."""
    # First, generate a task.
    utils.reset_config({"num_train_tasks": 0, "num_test_tasks": 1})
    env = get_or_create_env(env_name)
    task = env.get_test_tasks()[0]
    json_dict = utils.create_json_dict_from_task(task)
    f = tempfile.NamedTemporaryFile(mode="w+", suffix=".json")
    json.dump(json_dict, f)
    f.flush()
    # Now, load the task from the JSON file.
    recovered_task = env._load_task_from_json(f.name)  # pylint: disable=protected-access
    assert task.init.allclose(recovered_task.init)
    assert task.goal == recovered_task.goal
    # Test with a language goal.
    language_json_dict = json_dict.copy()
    del language_json_dict["goal"]
    language_json_dict["language_goal"] = "dummy language goal"
    f = tempfile.NamedTemporaryFile(mode="w+", suffix=".json")
    json.dump(language_json_dict, f)
    f.flush()
    with patch(f"{_MODULE_PATH}.BaseEnv._parse_language_goal_from_json") as m:
        m.return_value = set()
        recovered_task = env._load_task_from_json(f.name)  # pylint: disable=protected-access
    assert task.init.allclose(recovered_task.init)
    assert recovered_task.goal == set()
