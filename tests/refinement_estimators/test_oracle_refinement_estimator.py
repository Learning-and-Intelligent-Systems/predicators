"""Test cases for the oracle refinement cost estimator."""

import pytest

from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.refinement_estimators.oracle_refinement_estimator import \
    OracleRefinementEstimator
from predicators.settings import CFG
from predicators.structs import Task


def test_oracle_refinement_estimator():
    """Test general properties of oracle refinement cost estimator."""
    utils.reset_config({"env": "cover"})
    estimator = OracleRefinementEstimator()
    assert estimator.get_name() == "oracle"
    assert not estimator.is_learning_based
    with pytest.raises(NotImplementedError):
        sample_task = NarrowPassageEnv().get_train_tasks()[0]
        estimator.get_cost(sample_task, [], [])


def test_narrow_passage_oracle_refinement_estimator():
    """Test oracle refinement cost estimator for narrow_passage env."""
    utils.reset_config({
        "env": "narrow_passage",
        "narrow_passage_door_width_padding_lb": 0.02,
        "narrow_passage_door_width_padding_ub": 0.03,
        "narrow_passage_passage_width_padding_lb": 0.02,
        "narrow_passage_passage_width_padding_ub": 0.03,
    })
    estimator = OracleRefinementEstimator()

    # Get env objects and NSRTs
    env = NarrowPassageEnv()
    door_type, _, robot_type, target_type, _ = sorted(env.types)
    sample_task = env.get_train_tasks()[0]
    sample_state = sample_task.init
    door, = sample_state.get_objects(door_type)
    robot, = sample_state.get_objects(robot_type)
    target, = sample_state.get_objects(target_type)
    gt_nsrts = get_gt_nsrts(CFG.env, env.predicates, env.options)
    move_and_open_door_nsrt, move_to_target_nsrt = sorted(gt_nsrts)
    robot_radius = env.robot_radius

    # Ground NSRTs using objects
    ground_move_and_open_door = move_and_open_door_nsrt.ground([robot, door])
    ground_move_to_target = move_to_target_nsrt.ground([robot, target])

    # 1. Try state where door width > passage width
    sample_state.set(door, "width", (0.1 + robot_radius) * 2)
    task = Task(sample_state, sample_task.goal)

    # Test direct MoveToTarget skeleton
    move_direct_skeleton = [ground_move_to_target]
    move_direct_cost = estimator.get_cost(task, move_direct_skeleton, [])
    assert move_direct_cost == 1

    # Test open door then move skeleton
    move_through_door_skeleton = [
        ground_move_and_open_door,
        ground_move_to_target,
    ]
    move_through_door_cost = estimator.get_cost(task,
                                                move_through_door_skeleton, [])
    assert move_through_door_cost == 0

    # Make sure that sorting the costs considers move_through_door cheaper
    assert sorted([move_direct_cost, move_through_door_cost
                   ]) == [move_through_door_cost, move_direct_cost]

    # 2. Try state where door width < passage width
    sample_state.set(door, "width", (0.02 + robot_radius) * 2)
    task = Task(sample_state, sample_task.goal)

    # Test direct MoveToTarget skeleton
    move_direct_cost = estimator.get_cost(task, move_direct_skeleton, [])
    assert move_direct_cost == 1

    # Test open door then move skeleton
    move_through_door_cost = estimator.get_cost(task,
                                                move_through_door_skeleton, [])
    assert move_through_door_cost == 2

    # Make sure that sorting the costs considers move_direct cheaper
    assert sorted([move_direct_cost, move_through_door_cost
                   ]) == [move_direct_cost, move_through_door_cost]
