"""Test cases for the cluttered table environment."""

import pytest
import numpy as np
from gym.spaces import Box
from predicators.src.envs import ClutteredTableEnv, ClutteredTablePlaceEnv
from predicators.src import utils
from predicators.src.structs import Action
from predicators.src.envs import EnvironmentFailure


def test_cluttered_table(place_version=False):
    """Tests for ClutteredTableEnv class."""
    if not place_version:
        utils.update_config({"env": "cluttered_table"})
        env = ClutteredTableEnv()
    else:
        utils.update_config({"env": "cluttered_table_place"})
        env = ClutteredTablePlaceEnv()
    env.seed(123)
    train_tasks_gen = env.train_tasks_generator()
    for task in next(train_tasks_gen):
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    with pytest.raises(StopIteration):
        next(train_tasks_gen)
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {HandEmpty, Holding, Untrashed}.
    assert len(env.predicates) == 3
    # Goal predicates should be {Holding}.
    assert {pred.name for pred in env.goal_predicates} == {"Holding"}
    # Options should be {Grasp, Dump}. If place version, {Grasp, Place}.
    assert len(env.options) == 2
    # Types should be {can}
    assert len(env.types) == 1
    # Action space should be 4-dimensional.
    if not place_version:
        assert env.action_space == Box(0, 1, (4, ))
    else:
        assert env.action_space == Box(np.array([0, 0, 0, 0]),
                                       np.array([0.2, 0.2, 1, 1]))
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
            env.render(state, task, act)
        try:
            env.simulate(state, act)
        except EnvironmentFailure:  # pragma: no cover
            pass
        state.set(can, "is_grasped", 1.0)
        pose_x = state.get(can, "pose_x")
        pose_y = state.get(can, "pose_y")
        act = Action(np.array([0.0, 0.0, pose_x, pose_y], dtype=np.float32))
        if not place_version:
            next_state = env.simulate(state,
                                      act)  # grasp while already grasping
            assert state.allclose(next_state)
        if i == 0:
            env.render(state, task, act)


def test_cluttered_table_place():
    """Tests for ClutteredTablePlaceEnv class."""
    test_cluttered_table(place_version=True)
