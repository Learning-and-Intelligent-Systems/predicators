"""Generic controllers for the robots."""
import logging
from typing import Callable, Dict, Sequence, Set, Tuple, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose, Quaternion
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type
import pybullet as p

_SUPPORTED_ROBOTS: Set[str] = {"fetch", "panda"}

class AlwaysInitiable:
    def __call__(self, *args, **kwargs) -> bool:
        return True

class MoveEndEffectorToPosePolicy:
    def __init__(
        self,
        robot: SingleArmPyBulletRobot,
        get_current_and_target_pose_and_finger_status: Callable[
            [State, Sequence[Object], Array], Tuple[Pose, Pose, str]],
        max_vel_norm: float,
        finger_action_nudge_magnitude: float,
        max_rot_speed: float,
    ):
        self.robot = robot
        self.get_current_and_target_pose_and_finger_status = get_current_and_target_pose_and_finger_status
        self.max_vel_norm = max_vel_norm
        self.finger_action_nudge_magnitude = finger_action_nudge_magnitude
        self.max_rot_speed = max_rot_speed

    def __call__(
        self,
        state: State,
        memory: Dict,
        objects: Sequence[Object],
        params: Array
    ) -> Action:
        del memory  # unused
        # Sync the joints.
        assert isinstance(state, utils.PyBulletState)
        self.robot.set_joints(state.joint_positions)
        # First handle the main arm joints.
        current_pose, target_pose, finger_status = \
            self.get_current_and_target_pose_and_finger_status(
            state, objects, params)

        # # This option currently assumes a fixed end effector orientation.
        # assert np.allclose(current_pose.orientation, target_pose.orientation)

        current_pos, current_rot = current_pose
        target_pos, target_rot = target_pose

        ee_pos_delta = np.subtract(target_pos, current_pos)
        ee_rot_delta = quaternion_difference(target_rot, current_rot)

        # Reduce the target to conform to the max velocity constraint.
        ee_pos_norm = np.linalg.norm(ee_pos_delta)
        if ee_pos_norm > self.max_vel_norm:
            ee_pos_delta = ee_pos_delta * self.max_vel_norm / ee_pos_norm

        # Reduce the target to conform to the max rotational velocity constraint.
        axis, angle = p.getAxisAngleFromQuaternion(ee_rot_delta)
        if angle >= np.pi:
            angle -= np.pi * 2
        if np.abs(angle) > self.max_rot_speed:
            angle = angle / np.abs(angle) * self.max_rot_speed
            ee_rot_delta = p.getQuaternionFromAxisAngle(axis, angle)

        dx, dy, dz = np.add(current_pose.position, ee_pos_delta)
        orn = Pose((0, 0, 0), ee_rot_delta).multiply(Pose((0, 0, 0), current_rot)).orientation
        ee_action = Pose((dx, dy, dz), orn)

        # Keep validate as False because validate=True would update the
        # state of the robot during simulation, which overrides physics.
        try:
            # For the panda, always set the joints after running IK because
            # IKFast is very sensitive to initialization, and it's easier to
            # find good solutions on subsequent calls if we are already near
            # a solution from the previous call. The fetch robot does not
            # use IKFast, and in fact gets screwed up if we set joints here.
            joint_positions = self.robot.inverse_kinematics(ee_action,
                                                       validate=False,
                                                       set_joints=True)
        except InverseKinematicsError:
            raise utils.OptionExecutionFailure("Inverse kinematics failed.")
        # Handle the fingers. Fingers drift if left alone.
        # When the fingers are not explicitly being opened or closed, we
        # nudge the fingers toward being open or closed according to the
        # finger status.
        if finger_status == "open":
            finger_delta = self.finger_action_nudge_magnitude
        else:
            assert finger_status == "closed"
            finger_delta = -self.finger_action_nudge_magnitude
        # Extract the current finger state.
        state = cast(utils.PyBulletState, state)
        finger_position = state.joint_positions[self.robot.left_finger_joint_idx]
        # The finger action is an absolute joint position for the fingers.
        f_action = finger_position + finger_delta
        # Override the meaningless finger values in joint_action.
        joint_positions[self.robot.left_finger_joint_idx] = f_action
        joint_positions[self.robot.right_finger_joint_idx] = f_action
        action_arr = np.array(joint_positions, dtype=np.float32)
        # This clipping is needed sometimes for the joint limits.
        action_arr = np.clip(action_arr, self.robot.action_space.low,
                             self.robot.action_space.high)
        assert self.robot.action_space.contains(action_arr)
        return Action(action_arr)

class MoveEndEffectorToPoseTerminal:
    def __init__(
        self,
        get_current_and_target_pose_and_finger_status: Callable[
            [State, Sequence[Object], Array], Tuple[Pose, Pose, str]],
        move_to_pos_tol: float,
        move_to_rot_tol: float = 2e-3,
    ):
        self.get_current_and_target_pose_and_finger_status = get_current_and_target_pose_and_finger_status
        self.move_to_pos_tol = move_to_pos_tol
        self.move_to_rot_tol = move_to_rot_tol
    def __call__(
        self,
        state: State,
        memory: Dict,
        objects: Sequence[Object],
        params: Array
    ) -> bool:
        del memory  # unused
        current_pose, target_pose, _ = \
            self.get_current_and_target_pose_and_finger_status(
                state, objects, params)
        current_pos, current_rot = current_pose
        target_pos, target_rot = target_pose
        squared_dist = np.sqrt(np.square(np.subtract(current_pos, target_pos)).mean())
        _, angle = p.getAxisAngleFromQuaternion(quaternion_difference(target_rot, current_rot))
        return squared_dist < self.move_to_pos_tol and min(angle, np.pi * 2 - angle) < self.move_to_rot_tol

def create_move_end_effector_to_pose_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_pose_and_finger_status: Callable[
        [State, Sequence[Object], Array], Tuple[Pose, Pose, str]],
    move_to_pos_tol: float,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
    max_rot_speed: float = 10,
    move_to_rot_tol: float = 2e-3,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for moving the end
    effector to a target pose, given a function that takes in the current
    state, objects, and parameters, and returns the current pose and target
    pose of the end effector, and the finger status."""

    robot_name = robot.get_name()
    assert robot_name in _SUPPORTED_ROBOTS, (
        "Move end effector to pose option " +
        f"not implemented for robot {robot_name}.")

    return ParameterizedOption(name,
                               types=types,
                               params_space=params_space,
                               policy=MoveEndEffectorToPosePolicy(
                                   robot = robot,
                                   get_current_and_target_pose_and_finger_status = get_current_and_target_pose_and_finger_status,
                                   max_vel_norm = max_vel_norm,
                                   finger_action_nudge_magnitude = finger_action_nudge_magnitude,
                                   max_rot_speed = max_rot_speed
                               ),
                               initiable=AlwaysInitiable(),
                               terminal=MoveEndEffectorToPoseTerminal(
                                   get_current_and_target_pose_and_finger_status = get_current_and_target_pose_and_finger_status,
                                   move_to_pos_tol = move_to_pos_tol,
                                   move_to_rot_tol = move_to_rot_tol,
                               ))


class CreateChangeFingersOptionPolicy:
    def __init__(
        self,
        robot: SingleArmPyBulletRobot,
        get_current_and_target_val: Callable[[State, Sequence[Object], Array],
                                            Tuple[float, float]],
        max_vel_norm: float,
    ):
        self.robot = robot
        self.get_current_and_target_val = get_current_and_target_val
        self.max_vel_norm = max_vel_norm

    def __call__(
        self,
        state: State,
        memory: Dict,
        objects: Sequence[Object],
        params: Array
    ) -> Action:
        del memory  # unused
        current_val, target_val = self.get_current_and_target_val(
            state, objects, params)
        f_delta = target_val - current_val
        f_delta = np.clip(f_delta, -self.max_vel_norm, self.max_vel_norm)
        f_action = current_val + f_delta
        # Don't change the rest of the joints.
        state = cast(utils.PyBulletState, state)
        target = np.array(state.joint_positions, dtype=np.float32)
        target[self.robot.left_finger_joint_idx] = f_action
        target[self.robot.right_finger_joint_idx] = f_action
        # This clipping is needed sometimes for the joint limits.
        target = np.clip(target, self.robot.action_space.low,
                         self.robot.action_space.high)
        assert self.robot.action_space.contains(target)
        return Action(target)

class CreateChangeFingersOptionTerminal:
    def __init__(
        self,
        get_current_and_target_val: Callable[[State, Sequence[Object], Array],
                                         Tuple[float, float]],
        grasp_tol: float,
    ):
        self.get_current_and_target_val = get_current_and_target_val
        self.grasp_tol = grasp_tol

    def __call__(
        self,
        state: State,
        memory: Dict,
        objects: Sequence[Object],
        params: Array
    ) -> bool:
        del memory  # unused
        current_val, target_val = self.get_current_and_target_val(
            state, objects, params)
        squared_dist = (target_val - current_val)**2
        return squared_dist < self.grasp_tol

def create_change_fingers_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_val: Callable[[State, Sequence[Object], Array],
                                         Tuple[float, float]],
    max_vel_norm: float,
    grasp_tol: float,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for changing the
    robot fingers, given a function that takes in the current state, objects,
    and parameters, and returns the current and target finger joint values."""

    assert robot.get_name() in _SUPPORTED_ROBOTS, (
        "Change fingers option not " +
        f"implemented for robot {robot.get_name()}.")

    return ParameterizedOption(name,
                               types=types,
                               params_space=params_space,
                               policy=CreateChangeFingersOptionPolicy(
                                   robot = robot,
                                   get_current_and_target_val = get_current_and_target_val,
                                   max_vel_norm = max_vel_norm,
                               ),
                               initiable=AlwaysInitiable(),
                               terminal=CreateChangeFingersOptionTerminal(
                                    get_current_and_target_val = get_current_and_target_val,
                                    grasp_tol = grasp_tol,
                               ))

def quaternion_difference(q1: Quaternion, q2: Quaternion) -> Quaternion:
    return Pose((0, 0, 0), q1).multiply(Pose((0, 0, 0), q2).invert()).orientation