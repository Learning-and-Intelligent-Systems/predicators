"""Test cases for the cluttered table environment."""
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.cluttered_table import ClutteredTableEnv, \
    ClutteredTablePlaceEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import Action, GroundAtom


def test_cluttered_table(place_version=False):
    """Tests for ClutteredTableEnv class."""
    if not place_version:
        utils.reset_config({"env": "cluttered_table"})
        env = ClutteredTableEnv()
    else:
        utils.reset_config({"env": "cluttered_table_place"})
        env = ClutteredTablePlaceEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {HandEmpty, Holding, Untrashed}.
    assert len(env.predicates) == 3
    # Goal predicates should be {Holding}.
    assert {pred.name for pred in env.goal_predicates} == {"Holding"}
    # Options should be {Grasp, Dump}. If place version, {Grasp, Place}.
    assert len(get_gt_options(env.get_name())) == 2
    # Types should be {can}
    assert len(env.types) == 1
    # Action space should be 4-dimensional.
    if not place_version:
        assert env.action_space == Box(0, 1, (4, ))
    else:
        assert env.action_space == Box(np.array([0., 0., 0., 0.]),
                                       np.array([1., 1., 1., 1.]))
    HandEmpty = [pred for pred in env.predicates
                 if pred.name == "HandEmpty"][0]
    Untrashed = [pred for pred in env.predicates
                 if pred.name == "Untrashed"][0]
    Holding = [pred for pred in env.predicates if pred.name == "Holding"][0]
    # Test init state and simulate()
    for i, task in enumerate(env.get_test_tasks()):
        state = task.init
        for can1 in state:
            pose_x1 = state.get(can1, "pose_x")
            pose_y1 = state.get(can1, "pose_y")
            rad1 = state.get(can1, "radius")
            for can2 in state:
                if can1 == can2:
                    continue
                pose_x2 = state.get(can2, "pose_x")
                pose_y2 = state.get(can2, "pose_y")
                rad2 = state.get(can2, "radius")
                assert np.linalg.norm([pose_y2 - pose_y1, pose_x2 - pose_x1
                                       ]) > rad1 + rad2
        can = list(state)[0]
        act = Action(env.action_space.sample())
        if i == 0:
            env.render_state(state, task, act)
        try:
            env.simulate(state, act)
        except utils.EnvironmentFailure:  # pragma: no cover
            pass
        if not place_version:
            atoms = utils.abstract(state, env.predicates)
            assert GroundAtom(HandEmpty, []) in atoms
            for can1 in state:
                assert Untrashed([can1]) in atoms
            state.set(can, "is_grasped", 1.0)
            pose_x = state.get(can, "pose_x")
            pose_y = state.get(can, "pose_y")
            act = Action(np.array([0.0, 0.0, pose_x, pose_y],
                                  dtype=np.float32))
            next_state = env.simulate(state,
                                      act)  # grasp while already grasping
            assert state.allclose(next_state)
            atoms = utils.abstract(state, env.predicates)
            assert GroundAtom(HandEmpty, []) not in atoms
            assert Holding([can]) in atoms
            for can1 in state:
                assert Untrashed([can1]) in atoms
            # Additionally test when grasp is not on top of can coordinates
            act = Action(np.array([0.0, 0.0, 0.9, 0.9], dtype=np.float32))
            next_state = env.simulate(state, act)
            assert state.allclose(next_state)
        if i == 0:
            env.render_state(state, task, act, "caption")


def test_cluttered_table_place():
    """Tests for ClutteredTablePlaceEnv class."""
    test_cluttered_table(place_version=True)
