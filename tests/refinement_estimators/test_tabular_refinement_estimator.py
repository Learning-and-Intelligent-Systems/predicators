"""Test cases for the tabular refinement cost estimator."""
import pytest

from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.refinement_estimators.tabular_refinement_estimator import \
    TabularRefinementEstimator
from predicators.settings import CFG
from predicators.structs import GroundAtom


def test_tabular_refinement_estimator():
    """Test general properties of tabular refinement cost estimator."""
    utils.reset_config({
        "env": "narrow_passage",
        "refinement_data_failed_refinement_penalty": 3
    })
    estimator = TabularRefinementEstimator()
    assert estimator.get_name() == "tabular"
    assert estimator.is_learning_based
    with pytest.raises(AssertionError):
        sample_task = NarrowPassageEnv().get_train_tasks()[0].task
        estimator.get_cost(sample_task, [], [])
    # Check that train actually runs
    sample_data = [(sample_task, [], [], False, [], [])]
    estimator.train(sample_data)
    # Check that the resulting dictionary is correct
    cost_dict = estimator._model_dict  # pylint: disable=protected-access
    assert cost_dict == {(tuple(), tuple()): (3, 0)}
    assert estimator.get_cost(sample_task, [], []) == 3


def test_narrow_passage_tabular_refinement_estimator():
    """Test tabular refinement cost estimator for narrow_passage env."""
    utils.reset_config({
        "env": "narrow_passage",
        "refinement_data_failed_refinement_penalty": 3,
        "refinement_data_include_execution_cost": True,
        "refinement_data_low_level_execution_cost": 0.01,
    })
    estimator = TabularRefinementEstimator()

    # Get env objects and NSRTs
    env = NarrowPassageEnv()
    DoorIsClosed, DoorIsOpen, TouchedGoal = sorted(env.predicates)
    door_type, _, robot_type, target_type, _ = sorted(env.types)
    sample_task = env.get_train_tasks()[0].task
    sample_state = sample_task.init
    door, = sample_state.get_objects(door_type)
    robot, = sample_state.get_objects(robot_type)
    target, = sample_state.get_objects(target_type)
    gt_nsrts = get_gt_nsrts(CFG.env, env.predicates,
                            get_gt_options(env.get_name()))
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
        (sample_task, move_direct_skeleton, move_direct_atoms_seq, True, [4],
         [3]),
        (sample_task, move_through_door_skeleton, move_through_door_atoms_seq,
         True, [0.5, 1.5], [3, 5]),
        (sample_task, move_direct_skeleton, move_direct_atoms_seq, False, [5],
         []),
    ]
    estimator.train(sample_data)

    # Test direct MoveToTarget skeleton
    move_direct_cost = estimator.get_cost(sample_task, move_direct_skeleton,
                                          move_direct_atoms_seq)
    assert abs(move_direct_cost - 6.015) < 1e-5  # average of 4.03 and (5 + 3)

    # Test open door then move skeleton
    move_through_door_cost = estimator.get_cost(sample_task,
                                                move_through_door_skeleton,
                                                move_through_door_atoms_seq)
    assert abs(move_through_door_cost - 2.08) < 1e-5

    # Test an impossible skeleton
    impossible_skeleton = [
        ground_move_and_open_door,
        ground_move_and_open_door,
        ground_move_and_open_door,
        ground_move_to_target,
    ]
    impossible_cost = estimator.get_cost(sample_task, impossible_skeleton, [])
    assert impossible_cost == float('inf')

    # Make sure that sorting the costs makes sense
    assert sorted([
        move_direct_cost, move_through_door_cost, impossible_cost
    ]) == [move_through_door_cost, move_direct_cost, impossible_cost]
