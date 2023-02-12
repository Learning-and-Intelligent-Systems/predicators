"""Generic controllers for the robots."""
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type

_SUPPORTED_ROBOTS: Set[str] = {"fetch", "panda"}


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
    mode: str = "direct",  #  "direct" or "motion_planning"
    get_collision_bodies: Optional[Callable[[State, Sequence[Object]],
                                            Set[int]]] = None,
) -> ParameterizedOption:
    """A generic utility that creates a ParameterizedOption for moving the end
    effector to a target pose, given a function that takes in the current
    state, objects, and parameters, and returns the current pose and target
    pose of the end effector, and the finger status.

    There are two possible modes:
    1. The "direct" mode assumes that the end effector orientation is fixed
       and that the robot can move in a straight line (in ee position space)
       to the target pose without worrying about collisions. This mode is
       useful for simple up / down / left / right movements.
    2. The "motion_planning" mode does not make any assumptions about end
       effector orientation and performs full motion planning. It uses the
       get_collision_bodies() function to determine what pairs of objects
       should be collision checked during motion planning.

    In both cases, the option should respect max_vel_norm, i.e., should never
    move more than that magnitude in a single action, and should use the
    finger_action_nudge_magnitude to make sure that the fingers do not drift.
    """
    robot_name = robot.get_name()
    assert robot_name in _SUPPORTED_ROBOTS, (
        "Move end effector to pose option " +
        f"not implemented for robot {robot_name}.")

    if mode == "direct":
        assert get_collision_bodies is None
    else:
        assert mode == "motion_planning"
        assert get_collision_bodies is not None

    def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        if mode == "direct":
            return True
        # Run motion planning.
        initial_positions = state.simulator_state
        # Assume that the target pose won't change during execution.
        _, target_pose, _ = get_current_and_target_pose_and_finger_status(
            state, objects, params)
        target_positions = robot.inverse_kinematics(target_pose, validate=True)
        collision_bodies = get_collision_bodies(state, objects)
        plan = run_motion_planning(robot,
                                   initial_positions,
                                   target_positions,
                                   collision_bodies,
                                   seed=CFG.seed,
                                   physics_client_id=robot.physics_client_id)
        if plan is None:
            # Motion planning failed, option not initiable.
            return False
        # Store the motion plan.
        memory["plan"] = plan
        return True

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        current_pose, target_pose, finger_status = \
            get_current_and_target_pose_and_finger_status(
            state, objects, params)
        if mode == "direct":
            # This mode currently assumes a fixed end effector orientation.
            orn = current_pose.orientation
            # assert np.allclose(orn, target_pose.orientation)
            current = current_pose.position
            target = target_pose.position
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
                joint_positions = robot.inverse_kinematics(ee_action,
                                                           validate=False,
                                                           set_joints=True)
            except InverseKinematicsError:
                raise utils.OptionExecutionFailure(
                    "Inverse kinematics failed.")
        else:
            assert mode == "motion_planning"
            # Execute the motion plan open loop.
            joint_positions = memory["plan"].pop(0)
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
        current_pose, target_pose, _ = \
            get_current_and_target_pose_and_finger_status(
                state, objects, params)
        # This option currently assumes a fixed end effector orientation.
        # assert np.allclose(current_pose.orientation, target_pose.orientation)
        current = current_pose.position
        target = target_pose.position
        squared_dist = np.sum(np.square(np.subtract(current, target)))
        return squared_dist < move_to_pose_tol

    return ParameterizedOption(name,
                               types=types,
                               params_space=params_space,
                               policy=_policy,
                               initiable=_initiable,
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
