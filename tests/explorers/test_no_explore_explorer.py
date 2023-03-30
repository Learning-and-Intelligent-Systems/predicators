"""Test cases for the no explore explorer class."""
import pytest

from predicators import utils
from predicators.envs.cover import CoverEnv
from predicators.explorers import create_explorer
from predicators.ground_truth_models import get_gt_options


def test_no_explore_explorer():
    """Tests for NoExploreExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "no_explore",
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    task_idx = 0
    task = train_tasks[task_idx]
    explorer = create_explorer("no_explore", env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space, train_tasks)
    policy, termination_function = explorer.get_exploration_strategy(
        task_idx, 500)
    assert termination_function(task.init)
    with pytest.raises(RuntimeError) as e:
        policy(task.init)
    assert "The policy for no-explore shouldn't be used." in str(e)
