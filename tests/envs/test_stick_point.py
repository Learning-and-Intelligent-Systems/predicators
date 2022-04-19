"""Test cases for the stick point environment."""

import numpy as np

from predicators.src import utils
from predicators.src.envs.stick_point import StickPointEnv
from predicators.src.structs import Action

def test_stick_point():
    """Tests for StickPointEnv()."""
    utils.reset_config({
        "env": "stick_point",
        "stick_point_num_points_train": [2],
    })
    env = StickPointEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 6
    assert len(env.goal_predicates) == 1
    assert {pred.name for pred in env.goal_predicates} == {"Touched"}
    assert len(env.options) == 1
    assert len(env.types) == 3
    point_type, robot_type, stick_type = sorted(env.types)
    assert point_type.name == "point"
    assert robot_type.name == "robot"
    assert stick_type.name == "stick"
    assert env.action_space.shape == (3, )
    # Create a custom initial state, with the robot in the middle, one point
    # reachable on the left, one point out of the reachable zone in the middle,
    # and the stick on the right at a 45 degree angle.
    state = env.get_train_tasks()[0].init.copy()
    robot, = state.get_objects(robot_type)
    stick, = state.get_objects(stick_type)
    points = state.get_objects(point_type)
    assert len(points) == 2
    state.set(robot, "x", (env.rz_x_ub + env.rz_x_lb)/2)
    state.set(robot, "y", (env.rz_y_ub + env.rz_y_lb)/2)
    state.set(robot, "theta", np.pi/2)
    reachable_point, unreachable_point = points
    state.set(reachable_point, "x", (env.rz_x_ub + env.rz_x_lb)/4)
    state.set(reachable_point, "y", (env.rz_y_ub + env.rz_y_lb)/2)
    state.set(unreachable_point, "x", (env.rz_x_ub + env.rz_x_lb)/2)
    unreachable_y = 0.75 * env.y_ub
    assert not env.rz_y_lb <= unreachable_y <= env.rz_y_ub
    state.set(unreachable_point, "y", unreachable_y)
    state.set(stick, "x", 3*(env.rz_x_ub + env.rz_x_lb)/4)
    state.set(stick, "y", (env.rz_y_ub + env.rz_y_lb)/4)
    state.set(stick, "theta", np.pi/4)


    img, = env.render_state(state, task)
    # Uncomment to visualize for debugging.
    # TODO comment out.
    import imageio; imageio.imsave("/tmp/stick_point_test.png", img)
