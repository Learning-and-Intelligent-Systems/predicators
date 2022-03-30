"""Interfaces to PyBullet robots."""

import abc
from typing import Sequence

import numpy as np
import pybullet as p

from predicators.src import utils
from predicators.src.pybullet_utils.robots import _SingleArmPyBulletRobot
from predicators.src.pybullet_utils.utils import get_kinematic_chain, \
    inverse_kinematics
from predicators.src.structs import Array, Pose3D


class FetchPyBulletRobot(_SingleArmPyBulletRobot):
    """A Fetch robot with a fixed base and only one arm in use."""

    # Parameters that aren't important enough to need to clog up settings.py
    _base_pose: Pose3D = (0.75, 0.7441, 0.0)
    _base_orientation: Sequence[float] = [0., 0., 0., 1.]
    _ee_orientation: Sequence[float] = [1., 0., -1., 0.]
    _finger_action_nudge_magnitude: float = 0.001

    def _initialize(self) -> None:
        self._fetch_id = p.loadURDF(
            utils.get_env_asset_path("urdf/fetch/robots/fetch.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id)

        # Extract IDs for individual robot links and joints.
        joint_names = [
            p.getJointInfo(
                self._fetch_id, i,
                physicsClientId=self._physics_client_id)[1].decode("utf-8")
            for i in range(
                p.getNumJoints(self._fetch_id,
                               physicsClientId=self._physics_client_id))
        ]
        self._ee_id = joint_names.index('gripper_axis')
        self._arm_joints = get_kinematic_chain(
            self._fetch_id,
            self._ee_id,
            physics_client_id=self._physics_client_id)
        self._left_finger_id = joint_names.index("l_gripper_finger_joint")
        self._right_finger_id = joint_names.index("r_gripper_finger_joint")
        self._arm_joints.append(self._left_finger_id)
        self._arm_joints.append(self._right_finger_id)

        self._initial_joint_values = inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            self._ee_home_pose,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id)

    @property
    def robot_id(self) -> int:
        return self._fetch_id

    @property
    def end_effector_id(self) -> int:
        return self._ee_id

    @property
    def left_finger_id(self) -> int:
        return self._left_finger_id

    @property
    def right_finger_id(self) -> int:
        return self._right_finger_id

    def reset_state(self, robot_state: Array) -> None:
        rx, ry, rz, rf = robot_state
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id)
        assert np.allclose((rx, ry, rz), self._ee_home_pose)
        joint_values = self._initial_joint_values
        for joint_id, joint_val in zip(self._arm_joints, joint_values):
            p.resetJointState(self._fetch_id,
                              joint_id,
                              joint_val,
                              physicsClientId=self._physics_client_id)
        for finger_id in [self._left_finger_id, self._right_finger_id]:
            p.resetJointState(self._fetch_id,
                              finger_id,
                              rf,
                              physicsClientId=self._physics_client_id)

    def get_state(self) -> Array:
        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       physicsClientId=self._physics_client_id)
        rx, ry, rz = ee_link_state[4]
        rf = p.getJointState(self._fetch_id,
                             self._left_finger_id,
                             physicsClientId=self._physics_client_id)[0]
        # pose_x, pose_y, pose_z, fingers
        return np.array([rx, ry, rz, rf], dtype=np.float32)

    def set_motors(self, ee_delta: Pose3D, f_delta: float) -> None:
        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       physicsClientId=self._physics_client_id)
        current_position = ee_link_state[4]
        target_position = np.add(current_position, ee_delta).tolist()

        # We assume that the robot is already close enough to the target
        # position that IK will succeed with one call, so validate is False.
        # Furthermore, updating the state of the robot during simulation, which
        # validate=True would do, is risky and discouraged by PyBullet.
        joint_values = inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            target_position,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id,
            validate=False)

        # Set arm joint motors.
        for joint_idx, joint_val in zip(self._arm_joints, joint_values):
            p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                    jointIndex=joint_idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_val,
                                    physicsClientId=self._physics_client_id)

        # Set finger joint motors.
        for finger_id in [self._left_finger_id, self._right_finger_id]:
            current_val = p.getJointState(
                self._fetch_id,
                finger_id,
                physicsClientId=self._physics_client_id)[0]
            # Fingers drift if left alone. If the finger action is near zero,
            # nudge the fingers toward being open or closed, based on which end
            # of the spectrum they are currently closer to.
            if abs(f_delta) < self._finger_action_tol:
                assert self._open_fingers > self._closed_fingers
                if abs(current_val -
                       self._open_fingers) < abs(current_val -
                                                 self._closed_fingers):
                    nudge = self._finger_action_nudge_magnitude
                else:
                    nudge = -self._finger_action_nudge_magnitude
                target_val = current_val + nudge
            else:
                target_val = current_val + f_delta
            p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                    jointIndex=finger_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_val,
                                    physicsClientId=self._physics_client_id)
