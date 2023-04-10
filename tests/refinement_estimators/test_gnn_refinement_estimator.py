"""Test cases for the GNN refinement cost estimator."""

import os
import shutil
from pathlib import Path
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
from gym.spaces import Box

import predicators.envs.narrow_passage
from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.ground_truth_models.narrow_passage import \
    NarrowPassageGroundTruthNSRTFactory, \
    NarrowPassageGroundTruthOptionFactory
from predicators.refinement_estimators.gnn_refinement_estimator import \
    GNNRefinementEstimator
from predicators.settings import CFG
from predicators.structs import NSRT, Action, GroundAtom, \
    ParameterizedOption, Predicate, Task, Variable

_ENV_MODULE_NAME = predicators.envs.narrow_passage.__name__


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
        "refinement_data_include_execution_cost": True,
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
        (sample_task, move_direct_skeleton, move_direct_atoms_seq, True, [4],
         [3]),
        (sample_task, move_through_door_skeleton, move_through_door_atoms_seq,
         True, [0.5, 1.5], [3, 5]),
        (sample_task, move_direct_skeleton, move_direct_atoms_seq, False, [5],
         []),
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


def test_gnn_refinement_estimator_arities():
    """Test GNN refinement cost estimator on mocked predicate/NSRT sets that
    are 0-arity, unary, and binary."""
    utils.reset_config({
        "env": "narrow_passage",
        "gnn_num_message_passing": 1,
        "gnn_layer_size": 3,
        "gnn_num_epochs": 1,
        "gnn_use_validation_set": False,
    })

    # Get base environment, types, predicates, NSRTs
    env = NarrowPassageEnv()
    _, DoorIsOpen, TouchedGoal = sorted(env.predicates)
    door_type, _, robot_type, target_type, _ = sorted(env.types)
    base_options = get_gt_options(env.get_name())
    gt_nsrts = get_gt_nsrts(CFG.env, env.predicates, base_options)
    move_and_open_door_option, _ = sorted(base_options)
    move_and_open_door_nsrt, _ = sorted(gt_nsrts)

    # Make predicates of all arities
    ZeroArityPred = Predicate("ZeroArityPred", [], lambda s, o: False)
    UnaryPred = DoorIsOpen
    BinaryPred = TouchedGoal

    # Make dummy options and NSRTs of all arities

    _policy = lambda _1, _2, _3, _4: Action(
        np.array([0, 0, 0], dtype=np.float32))
    _initiable = lambda _1, _2, _3, _4: True
    _sampler = lambda _1, _2, rng, _4: np.array([rng.uniform()],
                                                dtype=np.float32)

    ZeroArityOption = ParameterizedOption("ZeroArityOption", [],
                                          Box(0, 1,
                                              (1, )), _policy, _initiable,
                                          lambda _1, _2, _3, _4: True)
    ZeroArityNSRT = NSRT("ZeroArityNSRT", [], set(), set(), set(), set(),
                         ZeroArityOption, [], _sampler)

    UnaryOption = ParameterizedOption("UnaryOption", [robot_type],
                                      Box(0, 1, (1, )), _policy, _initiable,
                                      lambda _1, _2, _3, _4: True)
    robot = Variable("?robot", robot_type)
    UnaryNSRT = NSRT("UnaryNSRT", [robot], set(), set(), set(), set(),
                     UnaryOption, [robot], _sampler)
    BinaryNSRT = move_and_open_door_nsrt

    mock_preds = {ZeroArityPred, UnaryPred, BinaryPred}
    mock_options = {ZeroArityOption, UnaryOption, move_and_open_door_option}
    mock_nsrts = {ZeroArityNSRT, UnaryNSRT, BinaryNSRT}

    with patch(f"{_ENV_MODULE_NAME}.NarrowPassageEnv.predicates",
               new_callable=PropertyMock) as mock_env, \
            patch.object(NarrowPassageGroundTruthOptionFactory,
                         "get_options",
                         return_value=mock_options), \
            patch.object(NarrowPassageGroundTruthNSRTFactory,
                         "get_nsrts",
                         return_value=mock_nsrts):
        mock_env.return_value = mock_preds
        # Test that _setup_fields() works
        estimator = GNNRefinementEstimator()
        estimator2 = GNNRefinementEstimator()

    # Make a test Task with all types of predicates/NSRTs involved
    sample_task = env.get_train_tasks()[0].task
    initial_state = sample_task.init
    robot, = initial_state.get_objects(robot_type)
    door, = initial_state.get_objects(door_type)
    target, = initial_state.get_objects(target_type)
    goal = {
        GroundAtom(ZeroArityPred, []),
        GroundAtom(UnaryPred, [door]),
        GroundAtom(BinaryPred, [robot, target]),
    }
    task = Task(initial_state, goal)

    # Make a test skeleton and atoms_sequence
    skeleton = [ZeroArityNSRT.ground([])]
    atoms_sequence = [goal, goal]

    # Create sample refinement training data
    data = [(task, skeleton, atoms_sequence, False, [5.0], [4])]
    # Check that train() and _graphify_single_input() successfully run
    estimator.train(data)

    # Test that getting a cost returns a finite cost
    test_cost = estimator.get_cost(task, skeleton, atoms_sequence)
    assert test_cost < float('inf')

    # Create fake directory to test saving and loading model
    parent_dir = os.path.dirname(__file__)
    approach_dir = os.path.join(parent_dir, "_fake_approach")
    os.makedirs(approach_dir, exist_ok=True)
    test_approach_path = Path(approach_dir) / "test.estimator"
    estimator.save_model(test_approach_path)
    estimator2.load_model(test_approach_path)

    # Check that the loaded model is the same as the saved one
    test_cost2 = estimator2.get_cost(task, skeleton, atoms_sequence)
    assert test_cost2 == test_cost

    # Remove temp directory
    shutil.rmtree(approach_dir)
