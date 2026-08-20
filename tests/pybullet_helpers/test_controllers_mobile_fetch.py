"""Tests for mobile Fetch controller utilities."""

import numpy as np
import pybullet as p

from predicators import utils
from predicators.pybullet_helpers.controllers import \
    get_move_end_effector_to_pose_with_base_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots.mobile_fetch import \
    MobileFetchPyBulletRobot


def test_mobile_fetch_base_follows_y_motion(physics_client_id):
    """Ensure base translates in y with minimal arm motion."""
    utils.reset_config({"pybullet_control_mode": "reset"})
    ee_home_position = (1.35, 0.75, 0.75)
    ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    ee_home_pose = Pose(ee_home_position, ee_orn)
    base_pose = Pose((0.75, 0.7441, 0.0))
    robot = MobileFetchPyBulletRobot(physics_client_id, ee_home_pose,
                                     base_pose)

    current_joint_positions = robot.get_joints()
    state = robot.get_state()
    current_pose = Pose(state[:3], state[3:7])
    delta_y = -0.1
    target = Pose(
        (current_pose.position[0], current_pose.position[1] + delta_y,
         current_pose.position[2]),
        current_pose.orientation,
    )
    action = get_move_end_effector_to_pose_with_base_action(
        robot=robot,
        current_joint_positions=current_joint_positions,
        current_pose=current_pose,
        target_pose=target,
        finger_status="open",
        max_vel_norm=0.2,
        finger_action_nudge_magnitude=1e-3,
        max_base_vel_norm=0.2,
        _max_base_rot_vel=0.5,
        _arm_reach_radius=0.8,
        validate=False,
    )
    assert action.arr.shape[0] == len(robot.arm_joints) + 3
    base_delta = action.arr[-3:]
    assert np.isclose(base_delta[1], delta_y, atol=1e-3)
    assert abs(base_delta[0]) < 1e-3
    assert abs(base_delta[2]) < 1e-6
    joint_delta = action.arr[:len(robot.arm_joints)] - \
        np.array(current_joint_positions)
    assert np.linalg.norm(joint_delta) < 1e-2


def test_mobile_fetch_base_follows_x_motion(physics_client_id):
    """Ensure base translates in x with minimal arm motion."""
    utils.reset_config({"pybullet_control_mode": "reset"})
    ee_home_position = (1.35, 0.75, 0.75)
    ee_orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    ee_home_pose = Pose(ee_home_position, ee_orn)
    base_pose = Pose((0.75, 0.7441, 0.0))
    robot = MobileFetchPyBulletRobot(physics_client_id, ee_home_pose,
                                     base_pose)

    current_joint_positions = robot.get_joints()
    state = robot.get_state()
    current_pose = Pose(state[:3], state[3:7])

    delta_x = 0.1
    target = Pose(
        (current_pose.position[0] + delta_x, current_pose.position[1],
         current_pose.position[2]),
        current_pose.orientation,
    )
    action = get_move_end_effector_to_pose_with_base_action(
        robot=robot,
        current_joint_positions=current_joint_positions,
        current_pose=current_pose,
        target_pose=target,
        finger_status="open",
        max_vel_norm=0.2,
        finger_action_nudge_magnitude=1e-3,
        max_base_vel_norm=0.2,
        _max_base_rot_vel=0.5,
        _arm_reach_radius=0.8,
        validate=False,
    )
    base_delta = action.arr[-3:]
    assert np.isclose(base_delta[0], delta_x, atol=1e-3)
    assert abs(base_delta[1]) < 1e-3
    assert abs(base_delta[2]) < 1e-6
    joint_delta = action.arr[:len(robot.arm_joints)] - \
        np.array(current_joint_positions)
    assert np.linalg.norm(joint_delta) < 1e-2
