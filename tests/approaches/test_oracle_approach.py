"""Test cases for the oracle approach class.
"""

import numpy as np
import pytest
from predicators.src.approaches import OracleApproach, ApproachFailure, \
    ApproachTimeout, TAMPApproach
from predicators.src.approaches.oracle_approach import _get_gt_ops
from predicators.src.envs import CoverEnv
from predicators.src.structs import Task, Action
from predicators.src import utils
from predicators.src.settings import CFG


def test_cover_get_gt_ops():
    """Tests for _get_gt_ops in CoverEnv.
    """
    utils.update_config({"env": "cover"})
    # All predicates and options
    env = CoverEnv()
    operators = _get_gt_ops(env.predicates, env.options)
    assert len(operators) == 2
    pick_operator, place_operator = sorted(operators, key=lambda o: o.name)
    assert pick_operator.name == "Pick"
    assert place_operator.name == "Place"
    env.seed(123)
    train_task = env.get_train_tasks()[0]
    state = train_task.init
    block0, _, _, target0, _ = sorted(state.data.keys())
    assert block0.name == "block0"
    assert target0.name == "target0"
    pick0_operator = pick_operator.ground([block0])
    rng = np.random.default_rng(123)
    pick_param = pick0_operator.sampler(state, rng)
    pick_option = pick0_operator.option.ground(pick_param)
    pick_action = pick_option.policy(state)
    assert env.action_space.contains(pick_action.arr)
    state = env.simulate(state, pick_action)
    place0_operator = place_operator.ground([block0, target0])
    place_param = place0_operator.sampler(state, rng)
    place_option = place0_operator.option.ground(place_param)
    place_action = place_option.policy(state)
    assert env.action_space.contains(place_action.arr)
    # Excluded option
    assert _get_gt_ops(env.predicates, set()) == set()
    # Excluded predicate
    predicates = {p for p in env.predicates if p.name != "Holding"}
    operators = _get_gt_ops(predicates, env.options)
    assert len(operators) == 2
    pick_operator, place_operator = sorted(operators, key=lambda o: o.name)
    for atom in pick_operator.preconditions:
        assert atom.predicate.name != "Holding"
    assert len(pick_operator.add_effects) == 0
    for atom in pick_operator.delete_effects:
        assert atom.predicate.name != "Holding"


def test_get_gt_ops():
    """Test get gt ops alone.
    """
    utils.update_config({"env": "not a real environment"})
    with pytest.raises(NotImplementedError):
        _get_gt_ops(set(), set())


def test_oracle_approach_cover():
    """Tests for OracleApproach class with CoverEnv.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    approach.seed(123)
    for task in env.get_train_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(
            policy, task, env.simulate, env.predicates)
        # Test that a random policy fails.
        assert not utils.policy_solves_task(
            lambda s: Action(env.action_space.sample()),
            task, env.simulate, env.predicates)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(
            policy, task, env.simulate, env.predicates)
        # Test that a random policy fails.
        assert not utils.policy_solves_task(
            lambda s: Action(env.action_space.sample()),
            task, env.simulate, env.predicates)


def test_oracle_approach_cover_failures():
    """Tests for failures in the OracleApproach.
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
    operators = _get_gt_ops(env.predicates, env.options)
    operators = {op for op in operators if op.name == "Place"}
    with pytest.raises(ApproachFailure):
        # Goal is not dr-reachable, should fail fast.
        TAMPApproach.sesame_plan(task, env.simulate, operators,
                                 env.predicates, timeout=500, seed=123)
    with pytest.raises(ApproachFailure):
        # Goal is not dr-reachable, but we disable that check.
        # Should run out of skeletons.
        TAMPApproach.sesame_plan(task, env.simulate, operators,
                                 env.predicates, timeout=500, seed=123,
                                 check_dr_reachable=False)
