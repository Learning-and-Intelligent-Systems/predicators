"""Test cases for planning algorithms.
"""

import pytest
from predicators.src.approaches import OracleApproach
from predicators.src.approaches.oracle_approach import get_gt_nsrts
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.envs import CoverEnv
from predicators.src.planning import sesame_plan
from predicators.src import utils
from predicators.src.structs import Task, NSRT, ParameterizedOption
from predicators.src.settings import CFG
from predicators.src.option_model import create_option_model


def test_sesame_plan():
    """Tests for sesame_plan().
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    task = env.get_train_tasks()[0]
    option_model = create_option_model(CFG.option_model_name, env.simulate)
    plan = sesame_plan(task, option_model, nsrts, env.predicates, 1, 123)
    assert len(plan) == 2


def test_sesame_plan_failures():
    """Tests for failures in the planner using the OracleApproach on CoverEnv.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    option_model = create_option_model(CFG.option_model_name, env.simulate)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space)
    approach.seed(123)
    task = next(env.train_tasks_generator())[0]
    trivial_task = Task(task.init, set())
    policy = approach.solve(trivial_task, timeout=500)
    with pytest.raises(ApproachFailure):
        policy(task.init)  # plan should get exhausted immediately
    assert utils.policy_solves_task(
        policy, trivial_task, env.simulate, env.predicates)
    assert len(task.goal) == 1
    Covers = next(iter(task.goal)).predicate
    block0 = [obj for obj in task.init if obj.name == "block0"][0]
    target0 = [obj for obj in task.init if obj.name == "target0"][0]
    target1 = [obj for obj in task.init if obj.name == "target1"][0]
    impossible_task = Task(task.init, {Covers([block0, target0]),
                                       Covers([block0, target1])})
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=0.1)  # times out
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=-100)  # times out
    old_max_samples_per_step = CFG.max_samples_per_step
    old_max_skeletons = CFG.max_skeletons_optimized
    CFG.max_samples_per_step = 1
    CFG.max_skeletons_optimized = float("inf")
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=1)  # backtracking occurs
    CFG.max_skeletons_optimized = old_max_skeletons
    with pytest.raises(ApproachFailure):
        approach.solve(impossible_task, timeout=1)  # hits skeleton limit
    CFG.max_samples_per_step = old_max_samples_per_step
    nsrts = get_gt_nsrts(env.predicates, env.options)
    nsrts = {nsrt for nsrt in nsrts if nsrt.name == "Place"}
    with pytest.raises(ApproachFailure):
        # Goal is not dr-reachable, should fail fast.
        sesame_plan(task, option_model, nsrts,
                    env.predicates, timeout=500, seed=123)
    with pytest.raises(ApproachFailure):
        # Goal is not dr-reachable, but we disable that check.
        # Should run out of skeletons.
        sesame_plan(task, option_model, nsrts,
                    env.predicates, timeout=500, seed=123,
                    check_dr_reachable=False)


def test_sesame_plan_uninitiable_option():
    """Tests planning in the presence of an option whose initiation set
    is nontrivial.
    """
    # pylint: disable=protected-access
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    option_model = create_option_model(CFG.option_model_name, env.simulate)
    initiable = lambda s, m, o, p: False
    nsrts = get_gt_nsrts(env.predicates, env.options)
    old_option = next(iter(env.options))
    new_option = ParameterizedOption(
        old_option.name, old_option.types, old_option.params_space,
        old_option._policy, initiable, old_option._terminal)
    new_nsrts = set()
    for nsrt in nsrts:
        new_nsrts.add(NSRT(
            nsrt.name+"UNINITIABLE", nsrt.parameters, nsrt.preconditions,
            nsrt.add_effects, nsrt.delete_effects, new_option,
            nsrt.option_vars, nsrt._sampler))
    task = env.get_train_tasks()[0]
    with pytest.raises(ApproachFailure) as e:
        # Planning should reach max_skeletons_optimized
        sesame_plan(task, option_model, new_nsrts,
                    env.predicates, timeout=500, seed=123)
    assert "Planning reached max_skeletons_optimized!" in str(e.value)
