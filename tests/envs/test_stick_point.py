"""Test cases for the stick point environment."""

import numpy as np

from predicators.src import utils
from predicators.src.envs.stick_point import StickPointEnv
from predicators.src.structs import Action, Task


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
    assert env.action_space.shape == (4, )
    # Create a custom initial state, with the robot in the middle, one point
    # reachable on the left, one point out of the reachable zone in the middle,
    # and the stick on the right at a 45 degree angle.
    state = env.get_train_tasks()[0].init.copy()
    robot, = state.get_objects(robot_type)
    stick, = state.get_objects(stick_type)
    points = state.get_objects(point_type)
    assert len(points) == 2
    robot_x = (env.rz_x_ub + env.rz_x_lb)/2
    state.set(robot, "x", robot_x)
    state.set(robot, "y", (env.rz_y_ub + env.rz_y_lb)/2)
    state.set(robot, "theta", np.pi/2)
    reachable_point, unreachable_point = points
    reachable_x = (env.rz_x_ub + env.rz_x_lb)/4
    state.set(reachable_point, "x", reachable_x)
    state.set(reachable_point, "y", (env.rz_y_ub + env.rz_y_lb)/2)
    state.set(unreachable_point, "x", robot_x)
    unreachable_y = 0.75 * env.y_ub
    assert not env.rz_y_lb <= unreachable_y <= env.rz_y_ub
    state.set(unreachable_point, "y", unreachable_y)
    state.set(stick, "x", 3*(env.rz_x_ub + env.rz_x_lb)/4)
    state.set(stick, "y", (env.rz_y_ub + env.rz_y_lb)/4)
    state.set(stick, "theta", np.pi/4)
    task = Task(state, task.goal)
    env.render_state(state, task)
    
    # Test for going to touch the reachable point.
    num_steps_to_left = int(np.ceil((robot_x - reachable_x) / env.max_speed))
    action_arrs = [np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32)
                             for _ in range(num_steps_to_left)]

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-2].get(reachable_point, "touched") < 0.5
    assert traj.states[-1].get(reachable_point, "touched") > 0.5

    # Test for going to pick up the stick.
    action_arrs.extend([np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)])
    # TODO

    # TODO remove
    policy = utils.action_arrs_to_policy(action_arrs)
    monitor = utils.SimulateVideoMonitor(task, env.render_state)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs),
                                           monitor=monitor)
    video = monitor.get_video()
    outfile = "hardcoded_actions_stick_point.mp4"
    utils.save_video(outfile, video)
    # TODO end remove

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
