"""Test cases for the GNN refinement cost estimator."""

import pytest

from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.refinement_estimators.gnn_refinement_estimator import \
    GNNRefinementEstimator
from predicators.settings import CFG
from predicators.structs import GroundAtom


def test_gnn_refinement_estimator():
    """Test general properties of GNN refinement cost estimator."""
    utils.reset_config({
        "env": "narrow_passage",
        "gnn_num_message_passing": 1,
        "gnn_layer_size": 3,
        "gnn_num_epochs": 1,
    })
    estimator = GNNRefinementEstimator()
    assert estimator.get_name() == "gnn"
    assert estimator.is_learning_based
    with pytest.raises(AssertionError):
        sample_task = NarrowPassageEnv().get_train_tasks()[0].task
        estimator.get_cost(sample_task, [], [])


def test_narrow_passage_gnn_refinement_estimator():
    """Test GNN refinement cost estimator for narrow_passage env."""
    utils.reset_config({
        "env": "narrow_passage",
        "gnn_num_message_passing": 1,
        "gnn_layer_size": 3,
        "gnn_num_epochs": 1,
        "gnn_do_normalization": True,
    })
    estimator = GNNRefinementEstimator()

    # Get env objects and NSRTs
    env = NarrowPassageEnv()
    DoorIsClosed, DoorIsOpen, TouchedGoal = sorted(env.predicates)
    door_type, _, robot_type, target_type, _ = sorted(env.types)
    sample_task = env.get_train_tasks()[0].task
    sample_state = sample_task.init
    door, = sample_state.get_objects(door_type)
    robot, = sample_state.get_objects(robot_type)
    target, = sample_state.get_objects(target_type)
    options = get_gt_options(env.get_name())
    gt_nsrts = get_gt_nsrts(CFG.env, env.predicates, options)
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
        (sample_task, move_direct_skeleton, move_direct_atoms_seq, True, [4]),
        (sample_task, move_through_door_skeleton, move_through_door_atoms_seq,
         True, [0.5, 1.5]),
        (sample_task, move_direct_skeleton, move_direct_atoms_seq, False, [5]),
    ]
    estimator.train(sample_data)

    # Test direct MoveToTarget skeleton returns finite cost
    move_direct_cost = estimator.get_cost(sample_task, move_direct_skeleton,
                                          move_direct_atoms_seq)
    assert move_direct_cost < float('inf')

    # Test open door then move skeleton returns finite cost
    move_through_door_cost = estimator.get_cost(sample_task,
                                                move_through_door_skeleton,
                                                move_through_door_atoms_seq)
    assert move_through_door_cost < float('inf')
