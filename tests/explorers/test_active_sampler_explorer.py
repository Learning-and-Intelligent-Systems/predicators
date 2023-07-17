"""Test cases for the active_sampler explorer class."""
from typing import Dict

import pytest

from predicators import utils
from predicators.envs.cover import RegionalBumpyCoverEnv
from predicators.explorers import create_explorer
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.option_model import _OracleOptionModel
from predicators.structs import NSRT, NSRTSampler


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
    nsrt_to_explorer_sampler: Dict[NSRT, NSRTSampler] = {}
    for nsrt in nsrts:
        nsrt_to_explorer_sampler[nsrt] = nsrt.sampler
    explorer = create_explorer(
        "active_sampler",
        env.predicates,
        get_gt_options(env.get_name()),
        env.types,
        env.action_space,
        train_tasks,
        nsrts,
        option_model,
        ground_op_hist=ground_op_hist,
        max_steps_before_termination=2,
        nsrt_to_explorer_sampler=nsrt_to_explorer_sampler)
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

    # Cover case where we are practicing with an empty ground_op_hist.
    explorer = create_explorer(
        "active_sampler",
        env.predicates,
        get_gt_options(env.get_name()),
        env.types,
        env.action_space,
        train_tasks,
        nsrts,
        option_model,
        ground_op_hist={},
        max_steps_before_termination=2,
        nsrt_to_explorer_sampler=nsrt_to_explorer_sampler)
    task_idx = 0
    policy, _ = explorer.get_exploration_strategy(task_idx, 500)
    with pytest.raises(utils.RequestActPolicyFailure):
        policy(state)

    # Test that the PickFromBumpy operator is tried more than the others when
    # we set the parameters of the environment such that picking is hard.
    utils.reset_config({
        "explorer": "active_sampler",
        "env": "regional_bumpy_cover",
        "bumpy_cover_num_bumps": 3,
        "bumpy_cover_spaces_per_bump": 3,
        "bumpy_cover_init_bumpy_prob": 1.0,  # force pick from bumpy
        "active_sampler_explore_bonus": 0.0,  # disable explore bonus
        "strips_learner": "oracle",
        "sampler_learner": "oracle",
    })
    env = RegionalBumpyCoverEnv()
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    option_model = _OracleOptionModel(env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    ground_op_hist = {}
    nsrt_to_explorer_sampler: Dict[NSRT, NSRTSampler] = {}
    for nsrt in nsrts:
        nsrt_to_explorer_sampler[nsrt] = nsrt.sampler
    explorer = create_explorer(
        "active_sampler",
        env.predicates,
        get_gt_options(env.get_name()),
        env.types,
        env.action_space,
        train_tasks,
        nsrts,
        option_model,
        ground_op_hist=ground_op_hist,
        nsrt_to_explorer_sampler=nsrt_to_explorer_sampler)
    task_idx = 0
    policy, term_fn = explorer.get_exploration_strategy(task_idx, 500)
    task = train_tasks[0]
    state = task.init.copy()
    for _ in range(25):
        assert not term_fn(state)
        state = env.simulate(state, policy(state))
    pick_op = [op for op in ground_op_hist if op.name == "PickFromBumpy"][0]
    assert len(ground_op_hist[pick_op]) > 10
    assert sum(ground_op_hist[pick_op]) < len(ground_op_hist[pick_op])
    # Verify that we had to plan to practice.
    assert len([op for op in ground_op_hist if op.name == "PlaceOnBumpy"]) > 0
