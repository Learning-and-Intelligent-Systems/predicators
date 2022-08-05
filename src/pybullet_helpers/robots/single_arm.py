"""Abstract class for single armed manipulators with Pybullet helper
functions."""
import abc
from typing import ClassVar, Sequence

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs.pybullet_robots import get_kinematic_chain, \
    pybullet_inverse_kinematics
from predicators.src.settings import CFG
from predicators.src.structs import Array, JointsState, Pose3D


class SingleArmPyBulletRobot(abc.ABC):
    """A single-arm fixed-base PyBullet robot with a two-finger gripper."""

    def __init__(self, ee_home_pose: Pose3D, ee_orientation: Sequence[float],
                 physics_client_id: int) -> None:
        # Initial position for the end effector.
        self._ee_home_pose = ee_home_pose
        # Orientation for the end effector.
        self._ee_orientation = ee_orientation
        self._physics_client_id = physics_client_id
        # These get overridden in initialize(), but type checking needs to be
        # aware that it exists.
        self._initial_joints_state: JointsState = []
        self._initialize()

    @property
    def initial_joints_state(self) -> JointsState:
        """The joint values for the robot in its home pose."""
        return self._initial_joints_state

    @property
    def action_space(self) -> Box:
        """The action space for the robot.

        Represents position control of the arm and finger joints.
        """
        return Box(np.array(self.joint_lower_limits, dtype=np.float32),
                   np.array(self.joint_upper_limits, dtype=np.float32),
                   dtype=np.float32)

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the robot."""
        raise NotImplementedError("Override me!")

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

    @property
    @abc.abstractmethod
    def joint_lower_limits(self) -> JointsState:
        """Lower bound on the arm joint limits."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def joint_upper_limits(self) -> JointsState:
        """Upper bound on the arm joint limits."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def open_fingers(self) -> float:
        """The value at which the finger joints should be open."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def closed_fingers(self) -> float:
        """The value at which the finger joints should be closed."""
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
    def get_joints(self) -> JointsState:
        """Get the joints state from the current PyBullet state."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def set_joints(self, joints_state: JointsState) -> None:
        """Directly set the joint states.

        Outside of resetting to an initial state, this should not be
        used with the robot that uses stepSimulation(); it should only
        be used for motion planning, collision checks, etc., in a robot
        that does not maintain state.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def set_motors(self, joints_state: JointsState) -> None:
        """Update the motors to move toward the given joints state."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def forward_kinematics(self, joints_state: JointsState) -> Pose3D:
        """Compute the end effector position that would result if the robot arm
        joints state was equal to the input joints_state.

        WARNING: This method will make use of resetJointState(), and so it
        should NOT be used during simulation.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def inverse_kinematics(self, end_effector_pose: Pose3D,
                           validate: bool) -> JointsState:
        """Compute a joints state from a target end effector position.

        The target orientation is always self._ee_orientation.

        If validate is True, guarantee that the returned joints state
        would result in end_effector_pose if run through
        forward_kinematics.

        WARNING: if validate is True, physics may be overridden, and so it
        should not be used within simulation.
        """
        raise NotImplementedError("Override me!")


class FetchPyBulletRobot(SingleArmPyBulletRobot):
    """A Fetch robot with a fixed base and only one arm in use."""

    # Parameters that aren't important enough to need to clog up settings.py
    _base_pose: ClassVar[Pose3D] = (0.75, 0.7441, 0.0)
    _base_orientation: ClassVar[Sequence[float]] = [0., 0., 0., 1.]

    @classmethod
    def get_name(cls) -> str:
        return "fetch"

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

        self._initial_joints_state = self.inverse_kinematics(
            self._ee_home_pose, validate=True)
        # The initial joint values for the fingers should be open. IK may
        # return anything for them.
        self._initial_joints_state[-2] = self.open_fingers
        self._initial_joints_state[-1] = self.open_fingers
        # Establish the lower and upper limits for the arm joints.
        self._joint_lower_limits = []
        self._joint_upper_limits = []
        for i in self._arm_joints:
            info = p.getJointInfo(self._fetch_id,
                                  i,
                                  physicsClientId=self._physics_client_id)
            lower_limit = info[8]
            upper_limit = info[9]
            # Per PyBullet documentation, values ignored if upper < lower.
            if upper_limit < lower_limit:
                self._joint_lower_limits.append(-np.inf)
                self._joint_upper_limits.append(np.inf)
            else:
                self._joint_lower_limits.append(lower_limit)
                self._joint_upper_limits.append(upper_limit)

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

    @property
    def joint_lower_limits(self) -> JointsState:
        return self._joint_lower_limits

    @property
    def joint_upper_limits(self) -> JointsState:
        return self._joint_upper_limits

    @property
    def open_fingers(self) -> float:
        return 0.04

    @property
    def closed_fingers(self) -> float:
        return 0.01

    def reset_state(self, robot_state: Array) -> None:
        rx, ry, rz, rf = robot_state
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id)
        # First, reset the joint values to self._initial_joints_state,
        # so that IK is consistent (less sensitive to initialization).
        self.set_joints(self._initial_joints_state)
        # Now run IK to get to the actual starting rx, ry, rz. We use
        # validate=True to ensure that this initialization works.
        joints_state = self.inverse_kinematics((rx, ry, rz), validate=True)
        self.set_joints(joints_state)
        # Handle setting the robot finger joints.
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

    def get_joints(self) -> JointsState:
        joints_state = []
        for joint_idx in self._arm_joints:
            joint_val = p.getJointState(
                self._fetch_id,
                joint_idx,
                physicsClientId=self._physics_client_id)[0]
            joints_state.append(joint_val)
        return joints_state

    def set_joints(self, joints_state: JointsState) -> None:
        assert len(joints_state) == len(self._arm_joints)
        for joint_id, joint_val in zip(self._arm_joints, joints_state):
            p.resetJointState(self._fetch_id,
                              joint_id,
                              targetValue=joint_val,
                              targetVelocity=0,
                              physicsClientId=self._physics_client_id)

    def set_motors(self, joints_state: JointsState) -> None:
        assert len(joints_state) == len(self._arm_joints)

        # Set arm joint motors.
        if CFG.pybullet_control_mode == "position":
            p.setJointMotorControlArray(
                bodyUniqueId=self._fetch_id,
                jointIndices=self._arm_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joints_state,
                physicsClientId=self._physics_client_id)
        elif CFG.pybullet_control_mode == "reset":
            self.set_joints(joints_state)
        else:
            raise NotImplementedError("Unrecognized pybullet_control_mode: "
                                      f"{CFG.pybullet_control_mode}")

    def forward_kinematics(self, joints_state: JointsState) -> Pose3D:
        self.set_joints(joints_state)
        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       computeForwardKinematics=True,
                                       physicsClientId=self._physics_client_id)
        position = ee_link_state[4]
        return position

    def inverse_kinematics(self, end_effector_pose: Pose3D,
                           validate: bool) -> JointsState:
        return pybullet_inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            end_effector_pose,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id,
            validate=validate)
