"""Test cases for the oracle approach class.
"""

import numpy as np
import pytest
from predicators.src.approaches import OracleApproach, ApproachFailure
from predicators.src.approaches.oracle_approach import get_gt_ops
from predicators.src.envs import CoverEnv, CoverEnvTypedOptions, \
    ClutteredTableEnv, EnvironmentFailure
from predicators.src.structs import Action
from predicators.src import utils
from predicators.src.settings import CFG


def test_cover_get_gt_ops():
    """Tests for get_gt_ops in CoverEnv.
    """
    utils.update_config({"env": "cover"})
    # All predicates and options
    env = CoverEnv()
    operators = get_gt_ops(env.predicates, env.options)
    assert len(operators) == 2
    pick_operator, place_operator = sorted(operators, key=lambda o: o.name)
    assert pick_operator.name == "Pick"
    assert place_operator.name == "Place"
    env.seed(123)
    train_task = env.get_train_tasks()[0]
    state = train_task.init
    block0, _, _, target0, _ = list(state)
    assert block0.name == "block0"
    assert target0.name == "target0"
    pick0_operator = pick_operator.ground([block0])
    rng = np.random.default_rng(123)
    pick_option = pick0_operator.sample_option(state, rng)
    pick_action = pick_option.policy(state)
    assert env.action_space.contains(pick_action.arr)
    state = env.simulate(state, pick_action)
    place0_operator = place_operator.ground([block0, target0])
    place_option = place0_operator.sample_option(state, rng)
    place_action = place_option.policy(state)
    assert env.action_space.contains(place_action.arr)
    # Excluded option
    assert get_gt_ops(env.predicates, set()) == set()
    # Excluded predicate
    predicates = {p for p in env.predicates if p.name != "Holding"}
    operators = get_gt_ops(predicates, env.options)
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
        get_gt_ops(set(), set())


def test_oracle_approach_cover():
    """Tests for OracleApproach class with CoverEnv.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in env.get_train_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(
            policy, task, env.simulate, env.predicates)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(
            lambda s: random_action, task, env.simulate, env.predicates)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(
            policy, task, env.simulate, env.predicates)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(
            lambda s: random_action, task, env.simulate, env.predicates)


def test_oracle_approach_cover_typed():
    """Tests for OracleApproach class with CoverEnvTypedOptions.
    """
    utils.update_config({"env": "cover_typed"})
    env = CoverEnvTypedOptions()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in env.get_train_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(
            policy, task, env.simulate, env.predicates)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(
            lambda s: random_action, task, env.simulate, env.predicates)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(
            policy, task, env.simulate, env.predicates)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(
            lambda s: random_action, task, env.simulate, env.predicates)


def test_cluttered_table_get_gt_ops():
    """Tests for get_gt_ops in ClutteredTableEnv.
    """
    utils.update_config({"env": "cluttered_table"})
    # All predicates and options
    env = ClutteredTableEnv()
    operators = get_gt_ops(env.predicates, env.options)
    assert len(operators) == 2
    dump_operator, grasp_operator = sorted(operators, key=lambda o: o.name)
    assert dump_operator.name == "Dump"
    assert grasp_operator.name == "Grasp"
    env.seed(123)
    for train_task in env.get_train_tasks():
        state = train_task.init
        can0, can1, _, can3, _ = list(state)
        assert can0.name == "can0"
        assert can3.name == "can3"
        grasp0_operator = grasp_operator.ground([can0])
        with pytest.raises(AssertionError):
            grasp_operator.ground([])
        rng = np.random.default_rng(123)
        grasp_option = grasp0_operator.sample_option(state, rng)
        grasp_action = grasp_option.policy(state)
        assert env.action_space.contains(grasp_action.arr)
        try:
            state = env.simulate(state, grasp_action)
        except EnvironmentFailure as e:
            assert len(e.offending_objects) == 1
        dump0_operator = dump_operator.ground([can3])
        with pytest.raises(AssertionError):
            dump_operator.ground([can3, can1])
        dump_option = dump0_operator.sample_option(state, rng)
        dump_action = dump_option.policy(state)
        assert env.action_space.contains(dump_action.arr)
        env.simulate(state, dump_action)  # never raises EnvironmentFailure


def test_oracle_approach_cluttered_table():
    """Tests for OracleApproach class with ClutteredTableEnv.
    """
    utils.update_config({"env": "cluttered_table"})
    env = ClutteredTableEnv()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    assert not approach.is_learning_based
    approach.seed(123)
    train_task = env.get_train_tasks()[0]
    policy = approach.solve(train_task, timeout=500)
    assert utils.policy_solves_task(
        policy, train_task, env.simulate, env.predicates)
    test_task = env.get_test_tasks()[0]
    policy = approach.solve(test_task, timeout=500)
    assert utils.policy_solves_task(
        policy, test_task, env.simulate, env.predicates)
