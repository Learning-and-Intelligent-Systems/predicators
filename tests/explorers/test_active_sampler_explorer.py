"""Test cases for the active_sampler explorer class."""
import pytest

from predicators import utils
from predicators.envs.cover import RegionalBumpyCoverEnv
from predicators.explorers import create_explorer
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.option_model import _OracleOptionModel
from predicators.structs import NSRT


def test_active_sampler_explorer():
    """Tests for ActiveSamplerExplorer class."""

    # Test that the explorer starts by solving the task.
    utils.reset_config({
        "explorer": "active_sampler",
        "env": "regional_bumpy_cover",
        "bumpy_cover_init_bumpy_prob": 0.0,  # to make the task trivial
        "strips_learner": "oracle",
        "sampler_learner": "oracle",
    })
    env = RegionalBumpyCoverEnv()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    option_model = _OracleOptionModel(env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    ground_op_hist = {}
    explorer = create_explorer("active_sampler",
                               env.predicates,
                               get_gt_options(env.get_name()),
                               env.types,
                               env.action_space,
                               train_tasks,
                               nsrts,
                               option_model,
                               ground_op_hist=ground_op_hist,
                               max_steps_before_termination=2)
    task_idx = 0
    policy, term_fn = explorer.get_exploration_strategy(task_idx, 500)
    task = train_tasks[0]
    assert len(task.goal) == 1
    # Should be solved in exactly two steps.
    state = task.init.copy()
    assert not term_fn(state)
    state = env.simulate(state, policy(state))
    assert not term_fn(state)
    state = env.simulate(state, policy(state))
    assert task.goal_holds(state)
    assert term_fn(state)  # because of max_steps_before_termination
    # The ground_op_hist should be updated accordingly.
    assert len(ground_op_hist) == 2
    assert all(v == [True] for v in ground_op_hist.values())
