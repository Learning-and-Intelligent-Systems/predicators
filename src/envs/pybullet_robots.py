"""Interfaces to PyBullet robots."""

import abc
from typing import ClassVar, List, Sequence

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs.pybullet_utils import get_kinematic_chain, \
    inverse_kinematics
from predicators.src.structs import Array, Pose3D


class _SingleArmPyBulletRobot(abc.ABC):
    """A single-arm fixed-base PyBullet robot with a two-finger gripper.

    The action space for the robot is 4D. The first three dimensions are
    a change in the (x, y, z) of the end effector. The last dimension is
    a change in the finger joint(s), which are constrained to be
    symmetric.
    """

    def __init__(self, ee_home_pose: Pose3D, open_fingers: float,
                 closed_fingers: float, max_vel_norm: float,
                 physics_client_id: int) -> None:
        # Initial position for the end effector.
        self._ee_home_pose = ee_home_pose
        # The value at which the finger joints should be open.
        self._open_fingers = open_fingers
        # The value at which the finger joints should be closed.
        self._closed_fingers = closed_fingers
        # Used for the action space.
        self._max_vel_norm = max_vel_norm
        self._physics_client_id = physics_client_id
        # These get overridden in initialize(), but type checking needs to be
        # aware that it exists.
        self._initial_joint_values: List[float] = []
        self._initialize()

    @property
    def initial_joint_values(self) -> List[float]:
        """The joint values for the robot in its home pose."""
        return self._initial_joint_values

    @property
    def action_space(self) -> Box:
        """The action space for the robot."""
        # This is a temporary implementation that will soon be replaced with
        # the robot's joint space.
        # dimensions: [dx, dy, dz, finger value]
        return Box(low=-self._max_vel_norm,
                   high=self._max_vel_norm,
                   shape=(4, ),
                   dtype=np.float32)

    @abc.abstractmethod
    def _initialize(self) -> None:
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def robot_id(self) -> int:
        """The PyBullet ID for the robot."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def end_effector_id(self) -> int:
        """The PyBullet ID for the end effector."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def left_finger_id(self) -> int:
        """The PyBullet ID for the left finger."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def right_finger_id(self) -> int:
        """The PyBullet ID for the right finger."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def left_finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the left finger."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def right_finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the right finger."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def reset_state(self, robot_state: Array) -> None:
        """Reset the robot state to match the input state.

        The robot_state corresponds to the State vector for the robot
        object.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_state(self) -> Array:
        """Get the robot state vector based on the current PyBullet state.

        This corresponds to the State vector for the robot object.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_joints(self) -> Sequence[float]:
        """Get the joint states from the current PyBullet state."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def set_motors(self, ee_delta: Pose3D, f_value: float) -> None:
        """Update the motors to execute the given action in PyBullet given a
        delta on the end effector and finger joint(s)."""
        raise NotImplementedError("Override me!")


class FetchPyBulletRobot(_SingleArmPyBulletRobot):
    """A Fetch robot with a fixed base and only one arm in use."""

    # Parameters that aren't important enough to need to clog up settings.py
    _base_pose: ClassVar[Pose3D] = (0.75, 0.7441, 0.0)
    _base_orientation: ClassVar[Sequence[float]] = [0., 0., 0., 1.]
    _ee_orientation: ClassVar[Sequence[float]] = [1., 0., -1., 0.]
    _finger_action_nudge_magnitude: ClassVar[float] = 0.001

    def _initialize(self) -> None:
        self._fetch_id = p.loadURDF(
            utils.get_env_asset_path("urdf/robots/fetch.urdf"),
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
        # The initial joint values for the fingers should be open. IK may
        # return anything for them.
        self._initial_joint_values[-2] = self._open_fingers
        self._initial_joint_values[-1] = self._open_fingers

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

    @property
    def left_finger_joint_idx(self) -> int:
        return len(self._arm_joints) - 2

    @property
    def right_finger_joint_idx(self) -> int:
        return len(self._arm_joints) - 1

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

    def get_joints(self) -> Sequence[float]:
        joint_state = []
        for joint_idx in self._arm_joints:
            joint_val = p.getJointState(
                self._fetch_id,
                joint_idx,
                physicsClientId=self._physics_client_id)[0]
            joint_state.append(joint_val)
        return joint_state

    def set_motors(self, ee_delta: Pose3D, f_value: float) -> None:
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
            p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                    jointIndex=finger_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=f_value,
                                    physicsClientId=self._physics_client_id)


def create_single_arm_pybullet_robot(
        robot_name: str, ee_home_pose: Pose3D, open_fingers: float,
        closed_fingers: float, max_vel_norm: float,
        physics_client_id: int) -> _SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    if robot_name == "fetch":
        return FetchPyBulletRobot(ee_home_pose, open_fingers, closed_fingers,
                                  max_vel_norm, physics_client_id)
    raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")
