"""Test cases for the no explore explorer class."""

import pytest

from predicators.src import utils
from predicators.src.envs.cover import CoverEnv
from predicators.src.explorers import create_explorer


def test_no_explore_explorer():
    """Tests for NoExploreExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "no_explore",
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    task = train_tasks[0]
    explorer = create_explorer("no_explore", env.predicates, env.options,
                               env.types, env.action_space, train_tasks)
    policy, termination_function = explorer.get_exploration_strategy(task, 500)
    assert termination_function(task.init)
    with pytest.raises(RuntimeError) as e:
        policy(task.init)
    assert "The policy for no-explore shouldn't be used." in str(e)
