"""Generic controllers for the robots."""
from typing import Callable, Dict, Sequence, Set, Tuple, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type

_SUPPORTED_ROBOTS: Set[str] = {"fetch", "panda"}


def create_move_end_effector_to_pose_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_pose_and_finger_status: Callable[
        [State, Sequence[Object], Array], Tuple[Pose3D, Pose3D, str]],
    move_to_pose_tol: float,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for moving the end
    effector to a target pose, given a function that takes in the current
    state, objects, and parameters, and returns the current pose and target
    pose of the end effector, and the finger status."""

    assert robot.get_name() in _SUPPORTED_ROBOTS, (
        "Move end effector to pose option " +
        f"not implemented for robot {robot.get_name()}.")

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory  # unused
        # First handle the main arm joints.
        current, target, finger_status = \
            get_current_and_target_pose_and_finger_status(
            state, objects, params)
        # Run IK to determine the target joint positions.
        ee_delta = np.subtract(target, current)
        # Reduce the target to conform to the max velocity constraint.
        ee_norm = np.linalg.norm(ee_delta)
        if ee_norm > max_vel_norm:
            ee_delta = ee_delta * max_vel_norm / ee_norm
        ee_action = np.add(current, ee_delta)
        # Keep validate as False because validate=True would update the
        # state of the robot during simulation, which overrides physics.
        try:
            joint_positions = robot.inverse_kinematics(
                (ee_action[0], ee_action[1], ee_action[2]), validate=False)
        except InverseKinematicsError:
            raise utils.OptionExecutionFailure("Inverse kinematics failed.")
        # Handle the fingers. Fingers drift if left alone.
        # When the fingers are not explicitly being opened or closed, we
        # nudge the fingers toward being open or closed according to the
        # finger status.
        if finger_status == "open":
            finger_delta = finger_action_nudge_magnitude
        else:
            assert finger_status == "closed"
            finger_delta = -finger_action_nudge_magnitude
        # Extract the current finger state.
        state = cast(utils.PyBulletState, state)
        finger_position = state.joint_positions[robot.left_finger_joint_idx]
        # The finger action is an absolute joint position for the fingers.
        f_action = finger_position + finger_delta
        # Override the meaningless finger values in joint_action.
        joint_positions[robot.left_finger_joint_idx] = f_action
        joint_positions[robot.right_finger_joint_idx] = f_action
        action_arr = np.array(joint_positions, dtype=np.float32)
        # This clipping is needed sometimes for the joint limits.
        action_arr = np.clip(action_arr, robot.action_space.low,
                             robot.action_space.high)
        assert robot.action_space.contains(action_arr)
        return Action(action_arr)

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del memory  # unused
        current, target, _ = \
            get_current_and_target_pose_and_finger_status(
                state, objects, params)
        squared_dist = np.sum(np.square(np.subtract(current, target)))
        return squared_dist < move_to_pose_tol

    return ParameterizedOption(name,
                               types=types,
                               params_space=params_space,
                               policy=_policy,
                               initiable=lambda _1, _2, _3, _4: True,
                               terminal=_terminal)


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

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory  # unused
        current_val, target_val = get_current_and_target_val(
            state, objects, params)
        f_delta = target_val - current_val
        f_delta = np.clip(f_delta, -max_vel_norm, max_vel_norm)
        f_action = current_val + f_delta
        # Don't change the rest of the joints.
        state = cast(utils.PyBulletState, state)
        target = np.array(state.joint_positions, dtype=np.float32)
        target[robot.left_finger_joint_idx] = f_action
        target[robot.right_finger_joint_idx] = f_action
        # This clipping is needed sometimes for the joint limits.
        target = np.clip(target, robot.action_space.low,
                         robot.action_space.high)
        assert robot.action_space.contains(target)
        return Action(target)

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del memory  # unused
        current_val, target_val = get_current_and_target_val(
            state, objects, params)
        squared_dist = (target_val - current_val)**2
        return squared_dist < grasp_tol

    return ParameterizedOption(name,
                               types=types,
                               params_space=params_space,
                               policy=_policy,
                               initiable=lambda _1, _2, _3, _4: True,
                               terminal=_terminal)
