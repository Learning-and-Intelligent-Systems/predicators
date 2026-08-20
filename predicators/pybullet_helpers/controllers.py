"""Generic controllers for the robots."""
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, cast

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.robots.mobile_fetch import \
    MobileFetchPyBulletRobot
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.structs import Action, Array, Object, \
    ParameterizedInitiable, ParameterizedOption, ParameterizedTerminal, \
    State, Type

_SUPPORTED_ROBOTS: Set[str] = {"fetch", "mobile_fetch", "panda"}


def _get_base_action_dim(robot: SingleArmPyBulletRobot) -> int:
    base_dim = getattr(robot, "base_action_dim", 0)
    return int(base_dim) if isinstance(base_dim, int) else 0


def _robot_supports_base_action(robot: SingleArmPyBulletRobot) -> bool:
    return _get_base_action_dim(robot) > 0 and \
        hasattr(robot, "get_base_pose") and hasattr(robot, "set_base_pose")


def _compute_ee_action_pose(current_pose: Pose, target_pose: Pose,
                            max_vel_norm: float) -> Pose:
    orn = target_pose.orientation
    current = np.array(current_pose.position, dtype=np.float32)
    target = np.array(target_pose.position, dtype=np.float32)
    ee_delta = np.subtract(target, current)
    ee_norm = np.linalg.norm(ee_delta)
    if ee_norm > max_vel_norm:
        ee_delta = ee_delta * max_vel_norm / ee_norm
    dx, dy, dz = np.add(current, ee_delta)
    return Pose((dx, dy, dz), orn)


def _compute_arm_joint_positions(robot: SingleArmPyBulletRobot,
                                 current_joint_positions: JointPositions,
                                 ee_action: Pose,
                                 validate: bool) -> JointPositions:
    robot.set_joints(current_joint_positions)
    if robot.get_name() == "panda":
        validate = False
    return robot.inverse_kinematics(ee_action,
                                    validate=validate,
                                    set_joints=True)


def _build_action_from_joints(
        robot: SingleArmPyBulletRobot,
        joint_positions: JointPositions,
        base_delta: Optional[np.ndarray] = None) -> Action:
    action_arr = np.array(joint_positions, dtype=np.float32)
    if _robot_supports_base_action(robot):
        base_dim = _get_base_action_dim(robot)
        if base_delta is None:
            base_delta = np.zeros(base_dim, dtype=np.float32)
        base_delta = np.asarray(base_delta, dtype=np.float32)
        if base_delta.shape[0] != base_dim:
            raise ValueError(
                f"Expected base_delta dim {base_dim}, got {base_delta.shape}")
        action_arr = np.concatenate([action_arr, base_delta])
    action_arr = np.clip(action_arr, robot.action_space.low,
                         robot.action_space.high)
    assert robot.action_space.contains(action_arr)
    return Action(action_arr)


def get_move_end_effector_to_pose_action(
    robot: SingleArmPyBulletRobot,
    current_joint_positions: JointPositions,
    current_pose: Pose,
    target_pose: Pose,
    finger_status: str,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
    validate: bool = True,
    move_base: bool = True,
) -> Action:
    """Get an action for moving the end effector to a target pose.

    See create_move_end_effector_to_pose_option() for more info.

    For mobile-base robots the base is also driven toward the target by
    default. Callers that position the base separately (e.g. the skill
    factories' ``_maybe_drive_base``) should pass ``move_base=False`` to keep
    this purely an arm motion -- otherwise the base would drift during delicate
    incremental-IK phases such as a switch push.
    """
    if move_base and _robot_supports_base_action(robot):
        max_base_vel_norm = getattr(robot, "default_base_vel_norm",
                                    max_vel_norm)
        max_base_rot_vel = getattr(robot, "default_base_rot_vel", max_vel_norm)
        arm_reach_radius = getattr(robot, "default_arm_reach_radius", 0.8)
        return get_move_end_effector_to_pose_with_base_action(
            robot=robot,
            current_joint_positions=current_joint_positions,
            current_pose=current_pose,
            target_pose=target_pose,
            finger_status=finger_status,
            max_vel_norm=max_vel_norm,
            finger_action_nudge_magnitude=finger_action_nudge_magnitude,
            max_base_vel_norm=max_base_vel_norm,
            _max_base_rot_vel=max_base_rot_vel,
            _arm_reach_radius=arm_reach_radius,
            validate=validate,
        )

    ee_action = _compute_ee_action_pose(current_pose, target_pose,
                                        max_vel_norm)
    try:
        joint_positions = _compute_arm_joint_positions(
            robot, current_joint_positions, ee_action, validate)
    except InverseKinematicsError:
        raise utils.OptionExecutionFailure("Inverse kinematics failed.")
    # Handle the fingers. Fingers drift if left alone.
    # When the fingers are not explicitly being opened or closed, we
    # nudge the fingers toward being open or closed according to the
    # finger status. "hold" keeps the current width (e.g. retreating
    # from a partial-open release without re-pinching the placed
    # object or sweeping wider next to its neighbors).
    if finger_status == "open":
        finger_delta = finger_action_nudge_magnitude
    elif finger_status == "hold":
        finger_delta = 0.0
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
    return _build_action_from_joints(robot, joint_positions)


def get_move_end_effector_to_pose_with_base_action(
    robot: SingleArmPyBulletRobot,
    current_joint_positions: JointPositions,
    current_pose: Pose,
    target_pose: Pose,
    finger_status: str,
    max_vel_norm: float,
    finger_action_nudge_magnitude: float,
    max_base_vel_norm: float,
    _max_base_rot_vel: float,
    _arm_reach_radius: float,
    validate: bool = True,
) -> Action:
    """Get a combined arm + base action for a mobile-base robot."""
    if not _robot_supports_base_action(robot):
        raise ValueError("Robot does not support base actions.")

    mobile_robot = cast(MobileFetchPyBulletRobot, robot)

    ee_action = _compute_ee_action_pose(current_pose, target_pose,
                                        max_vel_norm)

    base_pose = mobile_robot.get_base_pose()
    ee_delta = np.subtract(ee_action.position, current_pose.position)
    base_delta_xy = np.array(ee_delta[:2], dtype=np.float32)
    base_delta_norm = np.linalg.norm(base_delta_xy)
    if base_delta_norm > max_base_vel_norm:
        base_delta_xy = base_delta_xy * max_base_vel_norm / base_delta_norm
    base_delta = np.array([base_delta_xy[0], base_delta_xy[1], 0.0],
                          dtype=np.float32)

    moved_base_pose = None
    if not np.allclose(base_delta, 0.0):
        current_yaw = p.getEulerFromQuaternion(base_pose.orientation)[2]
        new_yaw = current_yaw + float(base_delta[2])
        moved_base_pose = Pose(
            (base_pose.position[0] + float(base_delta[0]),
             base_pose.position[1] + float(base_delta[1]),
             base_pose.position[2]),
            p.getQuaternionFromEuler([0.0, 0.0, new_yaw]),
        )
        mobile_robot.set_base_pose(moved_base_pose)

    try:
        joint_positions = _compute_arm_joint_positions(
            robot, current_joint_positions, ee_action, validate)
    except InverseKinematicsError:
        if moved_base_pose is not None:
            mobile_robot.set_base_pose(base_pose)
        raise utils.OptionExecutionFailure("Inverse kinematics failed.")
    # Handle the fingers. Fingers drift if left alone. "hold" keeps the
    # current width (see get_move_end_effector_to_pose_action).
    if finger_status == "open":
        finger_delta = finger_action_nudge_magnitude
    elif finger_status == "hold":
        finger_delta = 0.0
    else:
        assert finger_status == "closed"
        finger_delta = -finger_action_nudge_magnitude
    finger_position = current_joint_positions[robot.left_finger_joint_idx]
    f_action = finger_position + finger_delta
    joint_positions[robot.left_finger_joint_idx] = f_action
    joint_positions[robot.right_finger_joint_idx] = f_action
    return _build_action_from_joints(robot, joint_positions, base_delta)


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
    initiable: ParameterizedInitiable = lambda _1, _2, _3, _4: True,
    terminal: Optional[ParameterizedTerminal] = None,
    validate: bool = True,
    stall_limit: Optional[int] = None,
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
        # Sync the joints.
        assert isinstance(state, utils.PyBulletState)
        robot.set_joints(state.joint_positions)
        # First handle the main arm joints.
        current_pose, target_pose, finger_status = \
            get_current_and_target_pose_and_finger_status(
            state, objects, params)
        # This option currently assumes a fixed end effector orientation.
        # assert np.allclose(current_pose.orientation, target_pose.orientation)
        action = get_move_end_effector_to_pose_action(
            robot=robot,
            current_joint_positions=state.joint_positions,
            current_pose=current_pose,
            target_pose=target_pose,
            finger_status=finger_status,
            max_vel_norm=max_vel_norm,
            finger_action_nudge_magnitude=finger_action_nudge_magnitude,
            validate=validate,
        )
        return action

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        current_pose, target_pose, _ = \
            get_current_and_target_pose_and_finger_status(
                state, objects, params)
        # This option currently assumes a fixed end effector orientation.
        # assert np.allclose(current_pose.orientation, target_pose.orientation)
        current = current_pose.position
        target = target_pose.position
        squared_dist = np.sum(np.square(np.subtract(current, target)))
        if squared_dist < move_to_pose_tol:
            return True
        # When opted in via ``stall_limit``, also terminate once the end
        # effector has frozen: incremental IK can plateau a couple of cm
        # short of a reach-edge target under the fixed wrist orientation,
        # otherwise burning the whole option horizon. The near-target gate
        # keeps this from masking genuine far-from-goal failures.
        if stall_limit is not None:
            last = memory.get("_stall_last_pos")
            if last is not None and \
                    np.sum(np.square(np.subtract(current, last))) < 1e-8:
                memory["_stall_count"] = memory.get("_stall_count", 0) + 1
            else:
                memory["_stall_count"] = 0
            memory["_stall_last_pos"] = current
            if memory["_stall_count"] >= stall_limit and squared_dist < 0.01:
                return True
        return False

    return ParameterizedOption(
        name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=initiable,
        terminal=_terminal if terminal is None else terminal)


def get_change_fingers_action(robot: SingleArmPyBulletRobot,
                              current_joint_positions: JointPositions,
                              current_val: float, target_val: float,
                              max_vel_norm: float) -> Action:
    """Get change fingers action."""
    f_delta = target_val - current_val
    f_delta = np.clip(f_delta, -max_vel_norm, max_vel_norm)
    f_action = current_val + f_delta
    # Don't change the rest of the joints.
    target = np.array(current_joint_positions, dtype=np.float32)
    target[robot.left_finger_joint_idx] = f_action
    target[robot.right_finger_joint_idx] = f_action
    return _build_action_from_joints(robot, list(target))


def create_change_fingers_option(
    robot: SingleArmPyBulletRobot,
    name: str,
    types: Sequence[Type],
    params_space: Box,
    get_current_and_target_val: Callable[[State, Sequence[Object], Array],
                                         Tuple[float, float]],
    max_vel_norm: float,
    grasp_tol: float,
    terminal: Optional[ParameterizedTerminal] = None,
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
        state = cast(utils.PyBulletState, state)
        return get_change_fingers_action(robot, state.joint_positions,
                                         current_val, target_val, max_vel_norm)

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del memory  # unused
        current_val, target_val = get_current_and_target_val(
            state, objects, params)
        squared_dist = (target_val - current_val)**2
        # logging.debug(
        #     f"[terminal] current_val: {current_val}, "
        #     f"target_val: {target_val}, "
        #     f"squared_dist: {squared_dist}, "
        #     f"grasp_tol: {grasp_tol}")
        return squared_dist < grasp_tol

    return ParameterizedOption(
        name,
        types=types,
        params_space=params_space,
        policy=_policy,
        initiable=lambda _1, _2, _3, _4: True,
        terminal=_terminal if terminal is None else terminal)
