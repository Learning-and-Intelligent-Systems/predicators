"""Test cases for the random actions explorer class."""

from predicators import utils
from predicators.envs.cover import CoverEnv
from predicators.explorers import create_explorer


def test_random_actions_explorer():
    """Tests for RandomActionsExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "random_actions",
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    task_idx = 0
    task = train_tasks[task_idx]
    explorer = create_explorer("random_actions", env.predicates, env.options,
                               env.types, env.action_space, train_tasks)
    policy, termination_function = explorer.get_exploration_strategy(
        task_idx, 500)
    assert not termination_function(task.init)
    for _ in range(10):
        act = policy(task.init)
        assert env.action_space.contains(act.arr)
