"""Test cases for the oracle refinement cost estimator."""

import pytest

from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.refinement_estimators.oracle_refinement_estimator import \
    OracleRefinementEstimator
from predicators.settings import CFG


def test_oracle_refinement_estimator():
    """Test general properties of oracle refinement cost estimator."""
    utils.reset_config({"env": "non-existent env"})
    estimator = OracleRefinementEstimator()
    assert estimator.get_name() == "oracle"
    with pytest.raises(NotImplementedError):
        estimator.get_cost([], [])


def test_narrow_passage_oracle_refinement_estimator():
    """Test oracle refinement cost estimator for narrow_passage env."""
    utils.reset_config({"env": "narrow_passage"})
    estimator = OracleRefinementEstimator()

    # Get env objects and NSRTs
    env = NarrowPassageEnv()
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

    # Test direct MoveToTarget skeleton
    move_direct_skeleton = [ground_move_to_target]
    move_direct_cost = estimator.get_cost(move_direct_skeleton, [])
    assert move_direct_cost == 3

    # Test open door then move skeleton
    move_through_door_skeleton = [
        ground_move_and_open_door,
        ground_move_to_target,
    ]
    move_through_door_cost = estimator.get_cost(move_through_door_skeleton, [])
    assert move_through_door_cost == 1 + 1

    # Test open door multiple times then move skeleton
    move_door_multiple_skeleton = [
        ground_move_and_open_door,
        ground_move_and_open_door,
        ground_move_and_open_door,
        ground_move_to_target,
    ]
    move_door_multiple_cost = estimator.get_cost(move_door_multiple_skeleton,
                                                 [])
    assert move_door_multiple_cost == 4

    # Make sure that sorting the costs makes sense
    assert sorted([
        move_direct_cost, move_through_door_cost, move_door_multiple_cost
    ]) == [move_through_door_cost, move_direct_cost, move_door_multiple_cost]
