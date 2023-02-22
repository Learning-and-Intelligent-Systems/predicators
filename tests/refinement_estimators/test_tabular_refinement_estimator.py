"""Test cases for the tabular refinement cost estimator."""

import pytest

from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.refinement_estimators.tabular_refinement_estimator import \
    TabularRefinementEstimator
from predicators.settings import CFG
from predicators.structs import GroundAtom


def test_tabular_refinement_estimator():
    """Test general properties of tabular refinement cost estimator."""
    utils.reset_config({
        "env": "fake_env",
        "refinement_data_failed_refinement_penalty": 3
    })
    estimator = TabularRefinementEstimator()
    assert estimator.get_name() == "tabular"
    assert estimator.is_learning_based
    with pytest.raises(AssertionError):
        sample_state = NarrowPassageEnv().get_train_tasks()[0].init
        estimator.get_cost(sample_state, [], [])
    # Check that train actually runs
    sample_data = [(sample_state, [], [], False, 5)]
    estimator.train(sample_data)
    # Check that the resulting dictionary is correct
    cost_dict = estimator._cost_dict  # pylint: disable=protected-access
    assert cost_dict == {(tuple(), tuple()): 8}
    assert estimator.get_cost(sample_state, [], []) == 8


def test_narrow_passage_tabular_refinement_estimator():
    """Test tabular refinement cost estimator for narrow_passage env."""
    utils.reset_config({
        "env": "narrow_passage",
        "refinement_data_failed_refinement_penalty": 3
    })
    estimator = TabularRefinementEstimator()

    # Get env objects and NSRTs
    env = NarrowPassageEnv()
    DoorIsClosed, DoorIsOpen, TouchedGoal = sorted(env.predicates)
    door_type, _, robot_type, target_type, _ = sorted(env.types)
    sample_state = env.get_train_tasks()[0].init
    door, = sample_state.get_objects(door_type)
    robot, = sample_state.get_objects(robot_type)
    target, = sample_state.get_objects(target_type)
    gt_nsrts = get_gt_nsrts(CFG.env, env.predicates, env.options)
    move_and_open_door_nsrt, move_to_target_nsrt = sorted(gt_nsrts)

    # Ground NSRTs using objects
    ground_move_and_open_door = move_and_open_door_nsrt.ground([robot, door])
    ground_move_to_target = move_to_target_nsrt.ground([robot, target])
    # Ground atoms using objects
    ground_door_is_closed = GroundAtom(DoorIsClosed, [door])
    ground_door_is_open = GroundAtom(DoorIsOpen, [door])
    ground_touched_goal = GroundAtom(TouchedGoal, [robot, target])

    # Make valid test skeletons and atom_sequences
    move_direct_skeleton = [ground_move_to_target]
    move_direct_atoms_seq = [
        {ground_door_is_closed},
        {ground_door_is_closed, ground_touched_goal},
    ]
    move_through_door_skeleton = [
        ground_move_and_open_door,
        ground_move_to_target,
    ]
    move_through_door_atoms_seq = [
        {ground_door_is_closed},
        {ground_door_is_open},
        {ground_door_is_closed, ground_touched_goal},
    ]

    # Create sample data to train using
    sample_data = [
        (sample_state, move_direct_skeleton, move_direct_atoms_seq, True, 4),
        (sample_state, move_through_door_skeleton, move_through_door_atoms_seq,
         True, 2),
        (sample_state, move_direct_skeleton, move_direct_atoms_seq, False, 5),
    ]
    estimator.train(sample_data)

    # Test direct MoveToTarget skeleton
    move_direct_cost = estimator.get_cost(sample_state, move_direct_skeleton,
                                          move_direct_atoms_seq)
    assert move_direct_cost == 6  # average of 2 samples: 4 and 5 + 3

    # Test open door then move skeleton
    move_through_door_cost = estimator.get_cost(sample_state,
                                                move_through_door_skeleton,
                                                move_through_door_atoms_seq)
    assert move_through_door_cost == 2

    # Test an impossible skeleton
    impossible_skeleton = [
        ground_move_and_open_door,
        ground_move_and_open_door,
        ground_move_and_open_door,
        ground_move_to_target,
    ]
    impossible_cost = estimator.get_cost(sample_state, impossible_skeleton, [])
    assert impossible_cost == float('inf')

    # Make sure that sorting the costs makes sense
    assert sorted([
        move_direct_cost, move_through_door_cost, impossible_cost
    ]) == [move_through_door_cost, move_direct_cost, impossible_cost]
