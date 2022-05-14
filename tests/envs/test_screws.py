"""Test cases for the cover environment."""

import numpy as np
import pytest
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs.screws import ScrewsEnv
from predicators.src.structs import Action, Task


def test_screws():
    utils.reset_config({"seed": 0})
    env = ScrewsEnv()
    train_tasks = env.get_train_tasks()
    task = train_tasks[0]
    env.render_state(task.init, task)
    # Goal predicates should be {ScrewInReceptacle}.
    goal_preds = env.goal_predicates
    assert len(goal_preds) == 1
    assert {pred.name for pred in env.goal_predicates} == {"ScrewInReceptacle"}
    # Check that moving the robot beyond the bounds in the
    # x-direction doesn't allow it to move out of bounds.
    extreme_left_state = env.simulate(
        task.init,
        Action(
            np.array([-(env.rz_x_ub - env.rz_x_lb), 0.0, 0.0],
                     dtype=np.float32)))
    assert extreme_left_state.get(
        env._robot, "pose_x") == env.rz_x_lb + (env._gripper_width / 2.0)
    extreme_right_state = env.simulate(
        task.init,
        Action(
            np.array([(env.rz_x_ub - env.rz_x_lb), 0.0, 0.0],
                     dtype=np.float32)))
    assert extreme_right_state.get(
        env._robot, "pose_x") == env.rz_x_ub - (env._gripper_width / 2.0)
    # Check that moving the robot beyond the bounds in the
    # y-direction doesn't allow it to move out of bounds.
    extreme_down_state = env.simulate(
        task.init,
        Action(
            np.array([0.0, -(env.rz_y_ub - env.rz_y_lb), 0.0],
                     dtype=np.float32)))
    assert extreme_down_state.get(env._robot, "pose_y") == env.rz_y_lb
    extreme_up_state = env.simulate(
        task.init,
        Action(
            np.array([0.0, (env.rz_y_ub - env.rz_y_lb), 0.0],
                     dtype=np.float32)))
    assert extreme_up_state.get(env._robot, "pose_y") == env.rz_y_ub

    # Check that picking up something and then dropping it immediately
    # works (this is never done during normal optimal planning.)
    goal_screw = list(task.goal)[0].objects[0]
    move_to_screw_action = env._MoveToScrew_policy(task.init, {},
                                                   [env._robot, goal_screw],
                                                   np.zeros(1))
    above_screw_state = env.simulate(task.init, move_to_screw_action)
    holding_screw_state = env.simulate(
        above_screw_state, Action(np.array([0.0, 0.0, 1.0], dtype=np.float32)))
    screw_putdown_state = env.simulate(
        holding_screw_state, Action(np.array([0.0, 0.0, 0.0],
                                             dtype=np.float32)))
    assert screw_putdown_state.allclose(above_screw_state)
