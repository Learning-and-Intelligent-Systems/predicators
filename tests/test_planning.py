"""Test cases for planning algorithms.
"""

import pytest
from predicators.src.approaches import OracleApproach
from predicators.src.approaches.oracle_approach import get_gt_ops
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.envs import CoverEnv
from predicators.src.planning import sesame_plan
from predicators.src import utils
from predicators.src.structs import Task
from predicators.src.settings import CFG


def test_sesame_plan():
    """Tests for sesame_plan().
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    operators = get_gt_ops(env.predicates, env.options)
    task = env.get_train_tasks()[0]
    plan = sesame_plan(task, env.simulate, operators,
                       env.predicates, 1, 123)
    assert len(plan) == 2


def test_sesame_plan_failures():
    """Tests for failures in the planner using the OracleApproach on CoverEnv.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    approach.seed(123)
    task = env.get_train_tasks()[0]
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
    old_max_skeletons = CFG.max_skeletons
    CFG.max_samples_per_step = 1
    CFG.max_skeletons = float("inf")
    with pytest.raises(ApproachTimeout):
        approach.solve(impossible_task, timeout=1)  # backtracking occurs
    CFG.max_skeletons = old_max_skeletons
    with pytest.raises(ApproachFailure):
        approach.solve(impossible_task, timeout=1)  # hits skeleton limit
    CFG.max_samples_per_step = old_max_samples_per_step
    operators = get_gt_ops(env.predicates, env.options)
    operators = {op for op in operators if op.name == "Place"}
    with pytest.raises(ApproachFailure):
        # Goal is not dr-reachable, should fail fast.
        sesame_plan(task, env.simulate, operators,
                    env.predicates, timeout=500, seed=123)
    with pytest.raises(ApproachFailure):
        # Goal is not dr-reachable, but we disable that check.
        # Should run out of skeletons.
        sesame_plan(task, env.simulate, operators,
                    env.predicates, timeout=500, seed=123,
                    check_dr_reachable=False)
