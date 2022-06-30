"""Test cases for the random actions explorer class."""

from predicators.src import utils
from predicators.src.envs.cover import CoverEnv
from predicators.src.interaction import create_explorer


def test_random_actions_explorer():
    """Tests for RandomActionsExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "random_actions",
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    task = train_tasks[0]
    explorer = create_explorer("random_actions", env.predicates, env.options,
                               env.types, env.action_space, train_tasks)
    policy, termination_function = explorer.get_exploration_strategy(task, 500)
    assert not termination_function(task.init)
    actions = []
    for _ in range(10):
        act = policy(task.init)
        actions.append(act)
        assert env.action_space.contains(act.arr)
