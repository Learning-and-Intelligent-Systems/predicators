import abc
from functools import cached_property
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box
from pybullet_tools.utils import (
    Pose,
    euler_from_quat,
    link_from_name,
    draw_pose,
    wait_for_user,
)

from predicators.src import utils
from predicators.src.pybullet_helpers.ikfast import IKFastInfo
from predicators.src.pybullet_helpers.ikfast.utils import closest_inverse_kinematics
from predicators.src.pybullet_helpers.utils import (
    get_kinematic_chain,
    pybullet_inverse_kinematics,
)
from predicators.src.settings import CFG
from predicators.src.structs import Array, JointsState, Pose3D


class SingleArmPyBulletRobot(abc.ABC):
    """A single-arm fixed-base PyBullet robot with a two-finger gripper."""

    def __init__(
        self,
        ee_home_pose: Pose3D,
        ee_orientation: Sequence[float],
        physics_client_id: int,
        base_pose: Pose3D = (0, 0, 0),
        base_orientation: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
    ) -> None:
        # Initial position for the end effector.
        self._ee_home_pose = ee_home_pose
        # Orientation for the end effector.
        self._ee_orientation = ee_orientation
        self._physics_client_id = physics_client_id

        # Position and orientation of base of robot
        self._base_pose = base_pose
        self._base_orientation = base_orientation

        # Load the robot and reset base position and orientation.
        self.robot_id = p.loadURDF(
            self.urdf_path(),
            useFixedBase=True,
            physicsClientId=self._physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self.robot_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id,
        )

        # Robot initially at home pose
        self.go_home()

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the robot."""
        raise NotImplementedError("Override me!")

    @classmethod
    @abc.abstractmethod
    def urdf_path(cls) -> str:
        """Get the path to the URDF file for the robot."""
        raise NotImplementedError("Override me!")

    @property
    def action_space(self) -> Box:
        """
        The action space for the robot.
        Represents position control of the arm and finger joints.
        """
        return Box(
            np.array(self.joint_lower_limits, dtype=np.float32),
            np.array(self.joint_upper_limits, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    @abc.abstractmethod
    def end_effector_name(self) -> str:
        """The name of the end effector."""
        raise NotImplementedError("Override me!")

    @property
    def end_effector_id(self) -> int:
        """The PyBullet joint ID for the end effector."""
        return self.joint_names.index(self.end_effector_name)

    @property
    @abc.abstractmethod
    def tool_link_name(self) -> str:
        """The name of the end effector link (i.e., the tool link)."""
        raise NotImplementedError("Override me!")

    @cached_property
    def arm_joints(self) -> List[int]:
        """
        The Pybullet joint IDs of the joints of the robot arm.
        Note these are joint indices not body IDs.
        """
        joint_ids = get_kinematic_chain(
            self.robot_id,
            self.end_effector_id,
            physics_client_id=self._physics_client_id,
        )
        # NOTE: pybullet tools assumes sorted arm joints.
        joint_ids = sorted(joint_ids)
        joint_ids.extend([self.left_finger_id, self.right_finger_id])
        return joint_ids

    @cached_property
    def num_joints(self) -> int:
        """The number of joints in the robot."""
        return p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)

    @cached_property
    def joint_names(self) -> List[str]:
        """Get the names of the joints in the robot."""
        joint_names = [
            p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)[
                1
            ].decode("utf-8")
            for i in range(self.num_joints)
        ]
        return joint_names

    @property
    @abc.abstractmethod
    def left_finger_joint_name(self) -> str:
        """The name of the left finger joint."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def right_finger_joint_name(self) -> str:
        """The name of the right finger joint."""
        raise NotImplementedError("Override me!")

    @cached_property
    def left_finger_id(self) -> int:
        """The PyBullet joint ID for the left finger."""
        return self.joint_names.index(self.left_finger_joint_name)

    @cached_property
    def right_finger_id(self) -> int:
        """The PyBullet joint ID for the right finger."""
        return self.joint_names.index(self.right_finger_joint_name)

    @cached_property
    def left_finger_joint_idx(self) -> int:
        """
        The index into the joints corresponding to the left finger.
        Note this is not the joint ID, but the index of the joint within the list of arm joints.
        """
        return self.arm_joints.index(self.left_finger_id)

    @cached_property
    def right_finger_joint_idx(self) -> int:
        """
        The index into the joints corresponding to the right finger.
        Note this is not the joint ID, but the index of the joint within the list of arm joints.
        """
        return self.arm_joints.index(self.right_finger_id)

    @cached_property
    def _joint_limits(self) -> Tuple[JointsState, JointsState]:
        """Return the lower and upper joint limits."""
        joint_lower_limits, joint_upper_limits = [], []

        for i in self.arm_joints:
            info = p.getJointInfo(
                self.robot_id, i, physicsClientId=self._physics_client_id
            )
            lower_limit = info[8]
            upper_limit = info[9]
            # Per PyBullet documentation, values ignored if upper < lower.
            if upper_limit < lower_limit:
                joint_lower_limits.append(-np.inf)
                joint_upper_limits.append(np.inf)
            else:
                joint_lower_limits.append(lower_limit)
                joint_upper_limits.append(upper_limit)

        return joint_lower_limits, joint_upper_limits

    @property
    def joint_lower_limits(self) -> JointsState:
        """Lower bound on the arm joint limits."""
        return self._joint_limits[0]

    @property
    def joint_upper_limits(self) -> JointsState:
        """Upper bound on the arm joint limits."""
        return self._joint_limits[1]

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

    @cached_property
    def initial_joints_state(self) -> JointsState:
        """The joint values for the robot in its home pose."""
        initial_joints_state = self.inverse_kinematics(
            self._ee_home_pose, validate=True
        )
        # The initial joint values for the fingers should be open. IK may
        # return anything for them.
        initial_joints_state[self.left_finger_joint_idx] = self.open_fingers
        initial_joints_state[self.right_finger_joint_idx] = self.open_fingers
        return initial_joints_state

    def reset_state(self, robot_state: Array) -> None:
        """Reset the robot state to match the input state.

        The robot_state corresponds to the State vector for the robot
        object.
        """
        rx, ry, rz, rf = robot_state
        p.resetBasePositionAndOrientation(
            self.robot_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id,
        )
        # First, reset the joint values to initial joint state,
        # so that IK is consistent (less sensitive to initialization).
        self.set_joints(self.initial_joints_state)

        # Now run IK to get to the actual starting rx, ry, rz. We use
        # validate=True to ensure that this initialization works.
        joint_values = self.inverse_kinematics((rx, ry, rz), validate=True)
        self.set_joints(joint_values)

        # Handle setting the robot finger joints.
        for finger_id in [self.left_finger_id, self.right_finger_id]:
            p.resetJointState(
                self.robot_id, finger_id, rf, physicsClientId=self._physics_client_id
            )

    def get_state(self) -> Array:
        """Get the robot state vector based on the current PyBullet state.

        This corresponds to the State vector for the robot object.
        """
        ee_link_state = p.getLinkState(
            self.robot_id, self.end_effector_id, physicsClientId=self._physics_client_id
        )
        rx, ry, rz = ee_link_state[4]
        # Note: we assume both left and right gripper have the same joint position
        rf = p.getJointState(
            self.robot_id,
            self.left_finger_id,
            physicsClientId=self._physics_client_id,
        )[0]
        # pose_x, pose_y, pose_z, fingers
        return np.array([rx, ry, rz, rf], dtype=np.float32)

    def get_joints(self) -> JointsState:
        """Get the joints state from the current PyBullet state."""
        joint_state = [
            joint_info[0]  # extract joint position only
            for joint_info in p.getJointStates(
                self.robot_id, self.arm_joints, physicsClientId=self._physics_client_id
            )
        ]
        return joint_state

    def set_joints(self, joints_state: JointsState) -> None:
        """Directly set the joint states.

        Outside of resetting to an initial state, this should not be
        used with the robot that uses stepSimulation(); it should only
        be used for motion planning, collision checks, etc., in a robot
        that does not maintain state.
        """
        assert len(joints_state) == len(self.arm_joints)
        for joint_id, joint_val in zip(self.arm_joints, joints_state):
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=joint_val,
                targetVelocity=0,
                physicsClientId=self._physics_client_id,
            )

    def set_motors(self, joints_state: JointsState) -> None:
        """Update the motors to move toward the given joints state."""
        assert len(joints_state) == len(self.arm_joints)

        # Set arm joint motors.
        if CFG.pybullet_control_mode == "position":
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.arm_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joints_state,
                physicsClientId=self._physics_client_id,
            )
        elif CFG.pybullet_control_mode == "reset":
            self.set_joints(joints_state)
        else:
            raise NotImplementedError(
                f"Unrecognized pybullet_control_mode: {CFG.pybullet_control_mode}"
            )

    def go_home(self) -> None:
        """Move the robot to its home end-effector pose."""
        self.set_motors(self.initial_joints_state)

    def forward_kinematics(self, joints_state: JointsState) -> Pose3D:
        """Compute the end effector position that would result if the robot arm
        joints state was equal to the input joints_state.

        WARNING: This method will make use of resetJointState(), and so it
        should NOT be used during simulation.
        """
        self.set_joints(joints_state)
        ee_link_state = p.getLinkState(
            self.robot_id,
            self.end_effector_id,
            computeForwardKinematics=True,
            physicsClientId=self._physics_client_id,
        )
        position = ee_link_state[4]
        return position

    def _validate_joints_state(self, joints_state: JointsState, target_pose: Pose3D):
        """Validate that the given joint states matches the target pose."""
        # Store current joint positions so we can reset
        initial_joint_states = self.get_joints()

        # Set joint states, forward kinematics to determine end-effector position
        self.set_joints(joints_state)
        ee_pos = self.get_state()[:3]
        target_pos = target_pose
        pos_is_close = np.allclose(ee_pos, target_pos, atol=CFG.pybullet_ik_tol)

        # Reset joint positions before returning/raising error
        self.set_joints(initial_joint_states)

        if not pos_is_close:
            raise ValueError(
                f"Joint states do not match target pose {target_pos} from {ee_pos}"
            )

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        """
        IKFastInfo for this robot. If this is specified, then IK will use IKFast.
        """
        return None

    def _ikfast_inverse_kinematics(self, end_effector_pose: Pose3D) -> JointsState:
        tool_link = link_from_name(self.robot_id, self.tool_link_name)
        world_from_target = Pose(
            end_effector_pose, euler_from_quat(self._ee_orientation)
        )
        sols = closest_inverse_kinematics(
            self,
            tool_link,
            world_from_target,
            max_time=CFG.ikfast_max_time,
            max_candidates=CFG.ikfast_max_candidates,
        )
        sol = next(sols, None)
        if sol is None:
            raise ValueError(
                f"No IK solution found for target pose {end_effector_pose} using IKFast"
            )

        # Add fingers to state
        final_joint_state = list(sol)
        first_finger_idx, second_finger_idx = sorted(
            [self.left_finger_joint_idx, self.right_finger_joint_idx]
        )
        final_joint_state.insert(first_finger_idx, self.open_fingers)
        final_joint_state.insert(second_finger_idx, self.open_fingers)
        return final_joint_state

    def inverse_kinematics(
        self, end_effector_pose: Pose3D, validate: bool
    ) -> JointsState:
        """
        Compute a joints state from a target end effector position.
        Uses IKFast if the robot has IKFast info specified.

        The target orientation is always self._ee_orientation.

        If validate is True, guarantee that the returned joints state
        would result in end_effector_pose if run through
        forward_kinematics.

        WARNING: if validate is True, physics may be overridden, and so it
        should not be used within simulation.
        """
        if self.ikfast_info():
            final_joint_state = self._ikfast_inverse_kinematics(end_effector_pose)
            if validate:
                self._validate_joints_state(
                    final_joint_state, target_pose=end_effector_pose
                )
            return final_joint_state
        else:
            return pybullet_inverse_kinematics(
                self.robot_id,
                self.end_effector_id,
                end_effector_pose,
                self._ee_orientation,
                self.arm_joints,
                physics_client_id=self._physics_client_id,
                validate=validate,
            )


_ROBOT_TO_BASE_POSE: Dict[str, Pose3D] = {
    "fetch": (0.75, 0.7441, 0.0),
    "panda": (0.8, 0.7441, 0.25),
}


def create_single_arm_pybullet_robot(
    robot_name: str,
    ee_home_pose: Pose3D,
    ee_orientation: Sequence[float],
    physics_client_id: int,
) -> SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    available_robots = set()
    # TODO: fix this bad hack
    from .fetch import FetchPyBulletRobot
    from .panda import PandaPyBulletRobot

    for cls in utils.get_all_concrete_subclasses(SingleArmPyBulletRobot):
        available_robots.add(cls.get_name())
        if cls.get_name() == robot_name:
            base_pose = _ROBOT_TO_BASE_POSE.get(cls.get_name(), (0, 0, 0))
            robot = cls(
                ee_home_pose, ee_orientation, physics_client_id, base_pose=base_pose
            )
            break
    else:
        raise NotImplementedError(
            f"Unrecognized robot name: {robot_name}. "
            f"Supported robots: {', '.join(available_robots)}"
        )
    return robot
