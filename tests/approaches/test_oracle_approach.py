"""Test cases for the oracle approach class.
"""

import numpy as np
import pytest
from predicators.src.approaches import OracleApproach
from predicators.src.approaches.oracle_approach import get_gt_nsrts
from predicators.src.envs import CoverEnv, CoverEnvTypedOptions, \
    CoverEnvHierarchicalTypes, ClutteredTableEnv, EnvironmentFailure, \
    BlocksEnv, PaintingEnv, CoverMultistepOptions, RepeatedNextToEnv
from predicators.src.structs import Action
from predicators.src import utils


def test_cover_get_gt_nsrts():
    """Tests for get_gt_nsrts in CoverEnv.
    """
    utils.update_config({"env": "cover"})
    # All predicates and options
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    assert len(nsrts) == 2
    pick_nsrt, place_nsrt = sorted(nsrts, key=lambda o: o.name)
    assert pick_nsrt.name == "Pick"
    assert place_nsrt.name == "Place"
    env.seed(123)
    train_task = next(env.train_tasks_generator())[0]
    state = train_task.init
    block0, _, _, target0, _ = list(state)
    assert block0.name == "block0"
    assert target0.name == "target0"
    pick0_nsrt = pick_nsrt.ground([block0])
    rng = np.random.default_rng(123)
    pick_option = pick0_nsrt.sample_option(state, rng)
    pick_action = pick_option.policy(state)
    assert env.action_space.contains(pick_action.arr)
    state = env.simulate(state, pick_action)
    place0_nsrt = place_nsrt.ground([block0, target0])
    place_option = place0_nsrt.sample_option(state, rng)
    place_action = place_option.policy(state)
    assert env.action_space.contains(place_action.arr)
    # Excluded option
    assert get_gt_nsrts(env.predicates, set()) == set()
    # Excluded predicate
    predicates = {p for p in env.predicates if p.name != "Holding"}
    nsrts = get_gt_nsrts(predicates, env.options)
    assert len(nsrts) == 2
    pick_nsrt, place_nsrt = sorted(nsrts, key=lambda o: o.name)
    for atom in pick_nsrt.preconditions:
        assert atom.predicate.name != "Holding"
    assert len(pick_nsrt.add_effects) == 0
    for atom in pick_nsrt.delete_effects:
        assert atom.predicate.name != "Holding"


def test_get_gt_nsrts():
    """Test get_gt_nsrts alone.
    """
    utils.update_config({"env": "not a real environment"})
    with pytest.raises(NotImplementedError):
        get_gt_nsrts(set(), set())


def test_oracle_approach_cover():
    """Tests for OracleApproach class with CoverEnv.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
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


def test_oracle_approach_cover_typed_options():
    """Tests for OracleApproach class with CoverEnvTypedOptions.
    """
    utils.update_config({"env": "cover_typed_options"})
    env = CoverEnvTypedOptions()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
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


def test_oracle_approach_cover_hierarchical_types():
    """Tests for OracleApproach class with CoverEnvHierarchicalTypes.
    """
    utils.update_config({"env": "cover_hierarchical_types"})
    env = CoverEnvHierarchicalTypes()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
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


def test_oracle_approach_cover_multistep_options():
    """Tests for OracleApproach class with CoverMultistepOptions.
    """
    utils.update_config({"env": "cover_multistep_options"})
    env = CoverMultistepOptions()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
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


def test_cluttered_table_get_gt_nsrts():
    """Tests for get_gt_nsrts in ClutteredTableEnv.
    """
    utils.update_config({"env": "cluttered_table"})
    # All predicates and options
    env = ClutteredTableEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    assert len(nsrts) == 2
    dump_nsrt, grasp_nsrt = sorted(nsrts, key=lambda o: o.name)
    assert dump_nsrt.name == "Dump"
    assert grasp_nsrt.name == "Grasp"
    env.seed(123)
    for task in next(env.train_tasks_generator()):
        state = task.init
        can0, can1, _, can3, _ = list(state)
        assert can0.name == "can0"
        assert can3.name == "can3"
        grasp0_nsrt = grasp_nsrt.ground([can0])
        with pytest.raises(AssertionError):
            grasp_nsrt.ground([])
        rng = np.random.default_rng(123)
        grasp_option = grasp0_nsrt.sample_option(state, rng)
        grasp_action = grasp_option.policy(state)
        assert env.action_space.contains(grasp_action.arr)
        try:
            state = env.simulate(state, grasp_action)
        except EnvironmentFailure as e:
            assert len(e.offending_objects) == 1
        dump0_nsrt = dump_nsrt.ground([can3])
        with pytest.raises(AssertionError):
            dump_nsrt.ground([can3, can1])
        dump_option = dump0_nsrt.sample_option(state, rng)
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
        env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    train_task = next(env.train_tasks_generator())[0]
    policy = approach.solve(train_task, timeout=500)
    assert utils.policy_solves_task(
        policy, train_task, env.simulate, env.predicates)
    for test_task in env.get_test_tasks()[:5]:
        policy = approach.solve(test_task, timeout=500)
        assert utils.policy_solves_task(
            policy, test_task, env.simulate, env.predicates)


def test_oracle_approach_blocks():
    """Tests for OracleApproach class with BlocksEnv.
    """
    utils.update_config({"env": "blocks"})
    env = BlocksEnv()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    # Test a couple of train tasks so that we get at least one which
    # requires resampling placement poses on the table.
    for train_task in next(env.train_tasks_generator())[:10]:
        policy = approach.solve(train_task, timeout=500)
        assert utils.policy_solves_task(
            policy, train_task, env.simulate, env.predicates)
    test_task = env.get_test_tasks()[0]
    policy = approach.solve(test_task, timeout=500)
    assert utils.policy_solves_task(
        policy, test_task, env.simulate, env.predicates)


def test_oracle_approach_painting():
    """Tests for OracleApproach class with PaintingEnv.
    """
    utils.update_config({"env": "painting"})
    env = PaintingEnv()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    for train_task in next(env.train_tasks_generator())[:2]:
        policy = approach.solve(train_task, timeout=500)
        assert utils.policy_solves_task(
            policy, train_task, env.simulate, env.predicates)
    for test_task in env.get_test_tasks()[:2]:
        policy = approach.solve(test_task, timeout=500)
        assert utils.policy_solves_task(
            policy, test_task, env.simulate, env.predicates)


def test_oracle_approach_repeated_nextto():
    """Tests for OracleApproach class with RepeatedNextToEnv.
    """
    utils.update_config({"env": "repeated_nextto"})
    env = RepeatedNextToEnv()
    env.seed(123)
    approach = OracleApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    for train_task in next(env.train_tasks_generator())[:3]:
        policy = approach.solve(train_task, timeout=500)
        assert utils.policy_solves_task(
            policy, train_task, env.simulate, env.predicates)
    for test_task in env.get_test_tasks()[:3]:
        policy = approach.solve(test_task, timeout=500)
        assert utils.policy_solves_task(
            policy, test_task, env.simulate, env.predicates)
