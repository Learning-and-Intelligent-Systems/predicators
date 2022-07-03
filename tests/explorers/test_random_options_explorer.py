"""Test cases for the random options explorer class."""

from predicators.src import utils
from predicators.src.envs.cover import CoverEnv
from predicators.src.explorers import create_explorer


def test_random_options_explorer():
    """Tests for RandomOptionsExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "random_options",
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    task = train_tasks[0]
    explorer = create_explorer("random_options", env.predicates, env.options,
                               env.types, env.action_space, train_tasks)
    policy, termination_function = explorer.get_exploration_strategy(task, 500)
    assert not termination_function(task.init)
    for _ in range(10):
        act = policy(task.init)
        assert env.action_space.contains(act.arr)
