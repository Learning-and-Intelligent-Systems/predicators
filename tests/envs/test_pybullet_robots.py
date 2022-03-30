"""Test cases for pybullet_robots."""

import numpy as np
import pybullet as p
import pytest

from predicators.src.pybullet_utils.robots import create_single_arm_pybullet_robot
from predicators.src.pybullet_utils.robots.fetch import FetchPyBulletRobot
from predicators.src.settings import CFG


def test_fetch_pybullet_robot():
    """Tests for FetchPyBulletRobot()."""
    physics_client_id = p.connect(p.DIRECT)

    ee_home_pose = (1.35, 0.75, 0.75)
    open_fingers = 0.04
    closed_fingers = 0.01
    finger_tol = 0.0001
    robot = FetchPyBulletRobot(ee_home_pose, open_fingers, closed_fingers,
                               finger_tol, physics_client_id)

    robot_state = np.array(ee_home_pose + (open_fingers, ), dtype=np.float32)
    robot.reset_state(robot_state)
    recovered_state = robot.get_state()
    assert np.allclose(robot_state, recovered_state, atol=1e-3)

    ee_delta = (-0.01, 0.0, 0.01)
    f_delta = -0.01
    robot.set_motors(ee_delta, f_delta)
    for _ in range(CFG.pybullet_sim_steps_per_action):
        p.stepSimulation(physicsClientId=physics_client_id)
    expected_state = np.add(robot_state, ee_delta + (f_delta, ))
    recovered_state = robot.get_state()
    # IK is currently not precise enough to increase this tolerance.
    assert np.allclose(expected_state, recovered_state, atol=1e-2)


def test_create_single_arm_pybullet_robot():
    """Tests for create_single_arm_pybullet_robot()."""
    physics_client_id = p.connect(p.DIRECT)
    ee_home_pose = (1.35, 0.75, 0.75)
    open_fingers = 0.04
    closed_fingers = 0.01
    finger_tol = 0.0001
    robot = create_single_arm_pybullet_robot("fetch", ee_home_pose,
                                             open_fingers, closed_fingers,
                                             finger_tol, physics_client_id)
    assert isinstance(robot, FetchPyBulletRobot)
    with pytest.raises(NotImplementedError):
        create_single_arm_pybullet_robot("not a real robot", ee_home_pose,
                                         open_fingers, closed_fingers,
                                         finger_tol, physics_client_id)
