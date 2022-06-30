"""Test cases for the base explorer class."""

import pytest

from predicators.src import utils
from predicators.src.envs.cover import CoverEnv
from predicators.src.interaction import BaseExplorer, create_explorer


def test_create_explorer():
    """Tests for create_explorer."""
    utils.reset_config({
        "env": "cover",
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    for name in [
            "random_actions",
    ]:
        utils.reset_config({
            "env": "cover",
            "explorer": name,
        })
        explorer = create_explorer(name, env.predicates, env.options,
                                   env.types, env.action_space, train_tasks)
        assert isinstance(explorer, BaseExplorer)
    with pytest.raises(NotImplementedError):
        create_explorer("Not a real explorer", env.predicates, env.options,
                        env.types, env.action_space, train_tasks)
