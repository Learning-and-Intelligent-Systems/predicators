"""Generic controllers for the robots."""
from typing import Callable, Dict, Sequence, Set, Tuple, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type

_SUPPORTED_ROBOTS: Set[str] = {"fetch", "panda"}


def get_move_end_effector_to_pose_action(
    robot: SingleArmPyBulletRobot,
    current_joint_positions: JointPositions,
    current_pose: Pose,
    target_pose: Pose,
    finger_status: str,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
) -> Action:
    """Get an action for moving the end effector to a target pose.

    See create_move_end_effector_to_pose_option() for more info.
    """
    # Sync the joints.
    robot.set_joints(current_joint_positions)
    # First handle the main arm joints.
    # Currently assumes a fixed end effector orientation. (TODO fix)
    assert np.allclose(current_pose.orientation, target_pose.orientation)
    orn = current_pose.orientation
    current = current_pose.position
    target = target_pose.position
    # Run IK to determine the target joint positions.
    ee_delta = np.subtract(target, current)
    # Reduce the target to conform to the max velocity constraint.
    ee_norm = np.linalg.norm(ee_delta)
    if ee_norm > max_vel_norm:
        ee_delta = ee_delta * max_vel_norm / ee_norm
    dx, dy, dz = np.add(current, ee_delta)
    ee_action = Pose((dx, dy, dz), orn)
    # Keep validate as False because validate=True would update the
    # state of the robot during simulation, which overrides physics.
    try:
        # For the panda, always set the joints after running IK because
        # IKFast is very sensitive to initialization, and it's easier to
        # find good solutions on subsequent calls if we are already near
        # a solution from the previous call. The fetch robot does not
        # use IKFast, and in fact gets screwed up if we set joints here.
        joint_positions = robot.inverse_kinematics(
            ee_action,
            validate=True,  # TODO turn back to false? or make fetch-specific
            set_joints=True)
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
    finger_position = current_joint_positions[robot.left_finger_joint_idx]
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


def create_move_end_effector_to_pose_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_pose_and_finger_status: Callable[
        [State, Sequence[Object], Array], Tuple[Pose, Pose, str]],
    move_to_pose_tol: float,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for moving the end
    effector to a target pose, given a function that takes in the current
    state, objects, and parameters, and returns the current pose and target
    pose of the end effector, and the finger status."""

    robot_name = robot.get_name()
    assert robot_name in _SUPPORTED_ROBOTS, (
        "Move end effector to pose option " +
        f"not implemented for robot {robot_name}.")

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory  # unused
        current_pose, target_pose, finger_status = get_current_and_target_pose_and_finger_status(
            state, objects, params)
        assert isinstance(state, utils.PyBulletState)
        current_joint_positions = state.joint_positions
        return get_move_end_effector_to_pose_action(
            robot,
            current_joint_positions,
            current_pose,
            target_pose,
            finger_status,
            max_vel_norm,
            finger_action_nudge_magnitude,
        )

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del memory  # unused
        current_pose, target_pose, _ = \
            get_current_and_target_pose_and_finger_status(
                state, objects, params)
        # This option currently assumes a fixed end effector orientation.
        assert np.allclose(current_pose.orientation, target_pose.orientation)
        current = current_pose.position
        target = target_pose.position
        squared_dist = np.sum(np.square(np.subtract(current, target)))
        return squared_dist < move_to_pose_tol

    return ParameterizedOption(name,
                               types=types,
                               params_space=params_space,
                               policy=_policy,
                               initiable=lambda _1, _2, _3, _4: True,
                               terminal=_terminal)


def get_change_fingers_action(robot: SingleArmPyBulletRobot,
                              current_joint_positions: JointPositions,
                              current_val: float, target_val: float,
                              max_vel_norm: float) -> Action:
    f_delta = target_val - current_val
    f_delta = np.clip(f_delta, -max_vel_norm, max_vel_norm)
    f_action = current_val + f_delta
    # Don't change the rest of the joints.
    target = np.array(current_joint_positions, dtype=np.float32)
    target[robot.left_finger_joint_idx] = f_action
    target[robot.right_finger_joint_idx] = f_action
    # This clipping is needed sometimes for the joint limits.
    target = np.clip(target, robot.action_space.low, robot.action_space.high)
    assert robot.action_space.contains(target)
    return Action(target)


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
        return get_change_fingers_action(robot, current_val, target_val,
                                         max_vel_norm)

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
