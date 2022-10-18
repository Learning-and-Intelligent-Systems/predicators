"""Test cases for the random options explorer class."""

import pytest

from predicators import utils
from predicators.envs.cover import CoverEnv
from predicators.explorers import create_explorer
from predicators.structs import ParameterizedOption


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
    # Test case where no applicable option can be found.
    opt = sorted(env.options)[0]
    dummy_opt = ParameterizedOption(opt.name, opt.types, opt.params_space,
                                    opt.policy, lambda _1, _2, _3, _4: False,
                                    opt.terminal)
    explorer = create_explorer("random_options", env.predicates, {dummy_opt},
                               env.types, env.action_space, train_tasks)
    policy, _ = explorer.get_exploration_strategy(task, 500)
    with pytest.raises(utils.RequestActPolicyFailure) as e:
        policy(task.init)
    assert "Random option sampling failed!" in str(e)
