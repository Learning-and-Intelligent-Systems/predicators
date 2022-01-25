"""Test cases for the oracle approach class."""

from typing import Set
import numpy as np
import pytest
from predicators.src.approaches import OracleApproach
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.envs import CoverEnv, CoverEnvTypedOptions, \
    CoverEnvHierarchicalTypes, ClutteredTableEnv, ClutteredTablePlaceEnv, \
    EnvironmentFailure, BlocksEnv, PaintingEnv, PlayroomEnv, \
    CoverMultistepOptions, CoverMultistepOptionsFixedTasks, RepeatedNextToEnv
from predicators.src.structs import Action, NSRT, Variable
from predicators.src import utils


def test_cover_get_gt_nsrts():
    """Tests for get_gt_nsrts in CoverEnv."""
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
    """Test get_gt_nsrts alone."""
    utils.update_config({"env": "not a real environment"})
    with pytest.raises(NotImplementedError):
        get_gt_nsrts(set(), set())


def _check_nsrt_parameters(nsrts: Set[NSRT]) -> None:
    for nsrt in nsrts:
        effects_vars: Set[Variable] = set()
        precond_vars: Set[Variable] = set()
        for lifted_atom in nsrt.add_effects:
            effects_vars |= set(lifted_atom.variables)
        for lifted_atom in nsrt.delete_effects:
            effects_vars |= set(lifted_atom.variables)
        for lifted_atom in nsrt.preconditions:
            precond_vars |= set(lifted_atom.variables)
        assert set(nsrt.option_vars).issubset(nsrt.parameters), \
            f"Option variables is not a subset of parameters in {nsrt.name}"
        for var in nsrt.parameters:
            assert var in nsrt.option_vars or var in effects_vars, \
                f"Variable {var} not found in effects or option of {nsrt.name}"
        assert set(nsrt.parameters) == (set(nsrt.option_vars) | precond_vars |
                                        effects_vars), \
            f"Set of parameters is not the union of option and operator " \
            f"variables in {nsrt.name}"


def test_check_nsrt_parameters():
    """Checks all of the oracle operators for all envs."""
    envs = {
        "cover": CoverEnv(),
        "cover_typed_options": CoverEnvTypedOptions(),
        "cover_hierarchical_types": CoverEnvHierarchicalTypes(),
        "cluttered_table": ClutteredTableEnv(),
        "blocks": BlocksEnv(),
        "painting": PaintingEnv(),
        "playroom": PlayroomEnv(),
        "cover_multistep_options": CoverMultistepOptions(),
        "repeated_nextto": RepeatedNextToEnv()
    }
    for name, env in envs.items():
        utils.update_config({"env": name})
        nsrts = get_gt_nsrts(env.predicates, env.options)
        _check_nsrt_parameters(nsrts)


def test_oracle_approach_cover():
    """Tests for OracleApproach class with CoverEnv."""
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)


def test_oracle_approach_cover_typed_options():
    """Tests for OracleApproach class with CoverEnvTypedOptions."""
    utils.update_config({"env": "cover_typed_options"})
    env = CoverEnvTypedOptions()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)


def test_oracle_approach_cover_hierarchical_types():
    """Tests for OracleApproach class with CoverEnvHierarchicalTypes."""
    utils.update_config({"env": "cover_hierarchical_types"})
    env = CoverEnvHierarchicalTypes()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)


def test_oracle_approach_cover_multistep_options():
    """Tests for OracleApproach class with CoverMultistepOptions."""
    utils.update_config({"env": "cover_multistep_options"})
    utils.update_config({
        "cover_multistep_use_learned_equivalents": False,
        "cover_multistep_degenerate_oracle_samplers": False,
    })
    env = CoverMultistepOptions()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)
    utils.update_config({"env": "cover_multistep_options"})
    utils.update_config({
        "cover_multistep_use_learned_equivalents": True,
        "cover_multistep_degenerate_oracle_samplers": False,
        "sampler_learner": "neural"
    })
    env = CoverMultistepOptions()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test cover_multistep_degenerate_oracle_samplers.
        utils.update_config({"env": "cover_multistep_options"})
    utils.update_config({
        "cover_multistep_use_learned_equivalents": False,
        "cover_multistep_degenerate_oracle_samplers": True,
    })
    env = CoverMultistepOptions()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)


def test_oracle_approach_cover_multistep_options_fixed_tasks():
    """Tests for OracleApproach class with CoverMultistepOptionsFixedTasks."""
    utils.update_config({"env": "cover_multistep_options_fixed_tasks"})
    utils.update_config({"cover_multistep_use_learned_equivalents": True})
    env = CoverMultistepOptionsFixedTasks()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    random_action = Action(env.action_space.sample())
    approach.seed(123)
    for task in next(env.train_tasks_generator()):
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        assert utils.policy_solves_task(policy, task, env.simulate)
        # Test that a repeated random action fails.
        assert not utils.policy_solves_task(lambda s: random_action, task,
                                            env.simulate)


def test_cluttered_table_get_gt_nsrts(place_version=False):
    """Tests for get_gt_nsrts in ClutteredTableEnv."""
    if not place_version:
        utils.update_config({"env": "cluttered_table"})
        # All predicates and options
        env = ClutteredTableEnv()
    else:
        utils.update_config({"env": "cluttered_table_place"})
        env = ClutteredTablePlaceEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    assert len(nsrts) == 2
    if not place_version:
        dump_nsrt, grasp_nsrt = sorted(nsrts, key=lambda o: o.name)
        assert dump_nsrt.name == "Dump"
        assert grasp_nsrt.name == "Grasp"
    else:
        grasp_nsrt, place_nsrt = sorted(nsrts, key=lambda o: o.name)
        assert grasp_nsrt.name == "Grasp"
        assert place_nsrt.name == "Place"
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
        if not place_version:
            dump0_nsrt = dump_nsrt.ground([can3])
            with pytest.raises(AssertionError):
                dump_nsrt.ground([can3, can1])
            dump_option = dump0_nsrt.sample_option(state, rng)
            dump_action = dump_option.policy(state)
            assert env.action_space.contains(dump_action.arr)
            env.simulate(state, dump_action)  # never raises EnvironmentFailure
        else:
            place3_nsrt = place_nsrt.ground([can3])
            with pytest.raises(AssertionError):
                place_nsrt.ground([can3, can1])
            place_option = place3_nsrt.sample_option(state, rng)
            place_action = place_option.policy(state)
            assert env.action_space.contains(place_action.arr)
            try:
                env.simulate(state, place_action)
            except EnvironmentFailure as e:
                assert len(e.offending_objects) == 1


def test_cluttered_table_place_get_gt_nsrts():
    """Tests for get_gt_nsrts in ClutteredTablePlaceEnv."""
    test_cluttered_table_get_gt_nsrts(place_version=True)


def test_oracle_approach_cluttered_table(place_version=False):
    """Tests for OracleApproach class with ClutteredTableEnv."""
    if not place_version:
        utils.update_config({"env": "cluttered_table"})
        env = ClutteredTableEnv()
    else:
        utils.update_config({
            "env": "cluttered_table_place",
            "cluttered_table_num_cans_train": 3,
            "cluttered_table_num_cans_test": 3
        })
        env = ClutteredTablePlaceEnv()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    train_task = next(env.train_tasks_generator())[0]
    policy = approach.solve(train_task, timeout=500)
    assert utils.policy_solves_task(policy, train_task, env.simulate)
    for test_task in env.get_test_tasks()[:5]:
        policy = approach.solve(test_task, timeout=500)
        assert utils.policy_solves_task(policy, test_task, env.simulate)
    # Reset can settings here so other tests use the defaults
    utils.update_config({
        "cluttered_table_num_cans_train": 5,
        "cluttered_table_num_cans_test": 10
    })


def test_oracle_approach_cluttered_table_place():
    """Tests for OracleApproach class with ClutteredTablePlaceEnv."""
    test_oracle_approach_cluttered_table(place_version=True)


def test_oracle_approach_blocks():
    """Tests for OracleApproach class with BlocksEnv."""
    utils.update_config({"env": "blocks"})
    env = BlocksEnv()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    # Test a couple of train tasks so that we get at least one which
    # requires resampling placement poses on the table.
    for train_task in next(env.train_tasks_generator())[:10]:
        policy = approach.solve(train_task, timeout=500)
        assert utils.policy_solves_task(policy, train_task, env.simulate)
    test_task = env.get_test_tasks()[0]
    policy = approach.solve(test_task, timeout=500)
    assert utils.policy_solves_task(policy, test_task, env.simulate)


def test_oracle_approach_painting():
    """Tests for OracleApproach class with PaintingEnv."""
    utils.update_config({"env": "painting"})
    env = PaintingEnv()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    for train_task in next(env.train_tasks_generator())[:2]:
        policy = approach.solve(train_task, timeout=500)
        assert utils.policy_solves_task(policy, train_task, env.simulate)
    for test_task in env.get_test_tasks()[:2]:
        policy = approach.solve(test_task, timeout=500)
        assert utils.policy_solves_task(policy, test_task, env.simulate)


def test_oracle_approach_playroom():
    """Tests for OracleApproach class with PlayroomEnv."""
    utils.update_config({"env": "playroom"})
    env = PlayroomEnv()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    for train_task in next(env.train_tasks_generator())[:2]:
        policy = approach.solve(train_task, timeout=500)
        assert utils.policy_solves_task(policy, train_task, env.simulate)
    for test_task in env.get_test_tasks()[:2]:
        policy = approach.solve(test_task, timeout=500)
        assert utils.policy_solves_task(policy, test_task, env.simulate)
    # Test MoveDialToDoor for coverage.
    nsrts = get_gt_nsrts(env.predicates, env.options)
    movedialtodoor = [nsrt for nsrt in nsrts \
                      if nsrt.name == "MoveDialToDoor"][0]
    env.seed(123)
    train_task = next(env.train_tasks_generator())[0]
    state = train_task.init
    objs = list(state)
    robot, dial, door5, door6, region6, region7 = objs[17], objs[3], objs[
        8], objs[9], objs[15], objs[16]
    assert robot.name == "robby"
    assert dial.name == "dial"
    assert door5.name == "door5"
    assert door6.name == "door6"
    assert region6.name == "region6"
    assert region7.name == "region7"
    movedialtodoor_nsrt = movedialtodoor.ground([robot, dial, door6, region7])
    rng = np.random.default_rng(123)
    movetodoor_option = movedialtodoor_nsrt.sample_option(state, rng)
    movetodoor_action = movetodoor_option.policy(state)
    assert env.action_space.contains(movetodoor_action.arr)
    assert np.all(movetodoor_action.arr == np.array([110.1, 15, 1, -1, 1],
                                                    dtype=np.float32))
    # Test MoveDoorToTable for coverage.
    movedoortotable = [nsrt for nsrt in nsrts \
                      if nsrt.name == "MoveDoorToTable"][0]
    movedoortotable_nsrt = movedoortotable.ground([robot, door6, region7])
    movedoortotable_option = movedoortotable_nsrt.sample_option(state, rng)
    movedoortotable_action = movedoortotable_option.policy(state)
    assert env.action_space.contains(movedoortotable_action.arr)
    # Test AdvanceThroughDoor (moving left) for coverage.
    state.set(robot, "pose_x", 110.3)
    advancethroughdoor = [nsrt for nsrt in nsrts \
                      if nsrt.name == "AdvanceThroughDoor"][0]
    advancethroughdoor_nsrt = advancethroughdoor.ground(
        [robot, door6, region7, region6])
    movetodoor_option2 = advancethroughdoor_nsrt.sample_option(state, rng)
    movetodoor_action2 = movetodoor_option2.policy(state)
    assert env.action_space.contains(movetodoor_action2.arr)
    # Test MoveDoorToDoor (moving left) for coverage.
    movedoortodoor = [nsrt for nsrt in nsrts \
                      if nsrt.name == "MoveDoorToDoor"][0]
    movedoortodoor_nsrt = movedoortodoor.ground([robot, door6, door5, region6])
    movedoortodoor_option = movedoortodoor_nsrt.sample_option(state, rng)
    movedoortodoor_action = movedoortodoor_option.policy(state)
    assert env.action_space.contains(movedoortodoor_action.arr)


def test_oracle_approach_repeated_nextto():
    """Tests for OracleApproach class with RepeatedNextToEnv."""
    utils.update_config({"env": "repeated_nextto"})
    env = RepeatedNextToEnv()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.types, env.action_space)
    assert not approach.is_learning_based
    approach.seed(123)
    for train_task in next(env.train_tasks_generator())[:3]:
        policy = approach.solve(train_task, timeout=500)
        assert utils.policy_solves_task(policy, train_task, env.simulate)
    for test_task in env.get_test_tasks()[:3]:
        policy = approach.solve(test_task, timeout=500)
        assert utils.policy_solves_task(policy, test_task, env.simulate)
