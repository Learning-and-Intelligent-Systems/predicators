"""Test cases for the cluttered table environment.
"""

import pytest
import numpy as np
from gym.spaces import Box
from predicators.src.envs import ClutteredTableEnv
from predicators.src import utils
from predicators.src.structs import Action
from predicators.src.envs import EnvironmentFailure


def test_cluttered_table():
    """Tests for ClutteredTableEnv class.
    """
    utils.update_config({"env": "cluttered_table"})
    env = ClutteredTableEnv()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {HandEmpty, Holding}.
    assert len(env.predicates) == 2
    # Options should be {Grasp, Dump}.
    assert len(env.options) == 2
    # Types should be {can}
    assert len(env.types) == 1
    # Action space should be 4-dimensional.
    assert env.action_space == Box(0, 1, (4,))
    # Test init state and simulate()
    for task in env.get_test_tasks():
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
                assert np.linalg.norm(
                    [pose_y2-pose_y1, pose_x2-pose_x1]) > rad1+rad2
        can = list(state)[0]
        act = Action(env.action_space.sample())
        try:
            env.simulate(state, act)
        except EnvironmentFailure:
            pass
        state.set(can, "is_grasped", 1.0)
        pose_x = state.get(can, "pose_x")
        pose_y = state.get(can, "pose_y")
        act = Action(np.array([0.0, 0.0, pose_x, pose_y], dtype=np.float32))
        next_state = env.simulate(state, act)  # grasp while already grasping
        assert all(np.all(next_state[can] == state[can]) for can in state)
        with pytest.raises(NotImplementedError):
            env.render(state)
