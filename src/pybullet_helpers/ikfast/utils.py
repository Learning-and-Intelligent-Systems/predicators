from __future__ import annotations

import logging
import random
import time
from itertools import chain, islice
from numbers import Number
from typing import TYPE_CHECKING, Callable, Generator, List, Sequence, Tuple, \
    Union

import numpy as np
from pybullet_tools.utils import get_ordered_ancestors

from predicators.src.pybullet_helpers.ikfast import IKFastInfo
from predicators.src.pybullet_helpers.ikfast.load import import_ikfast
from predicators.src.pybullet_helpers.utils import Pose, get_joint_info, \
    get_joint_infos, get_joint_limits, get_joint_lower_limits, \
    get_joint_upper_limits, get_link_from_name, get_link_parent, \
    get_link_pose, get_relative_link_pose, matrix_from_quat, multiply_poses
from predicators.src.settings import CFG
from predicators.src.structs import JointsState

"""
Note: modified from pybullet-planning for our purposes.
"""

if TYPE_CHECKING:
    from predicators.src.pybullet_helpers.robots import SingleArmPyBulletRobot


def get_length(vec: Union[np.ndarray, List[Number]], norm=2) -> float:
    return np.linalg.norm(vec, ord=norm)


def prune_fixed_joints(body: int, joints: List[int],
                       physics_client_id: int) -> List[int]:
    return [
        joint for joint in joints
        if get_joint_info(body, joint, physics_client_id).is_movable()
    ]


# Joint index == link index in pybullet
parent_joint_from_link = lambda joint: joint


def violates_limit(body: int, joint: int, value: float,
                   physics_client_id: int) -> bool:
    # TODO: custom limits
    if get_joint_info(body, joint, physics_client_id).is_circular():
        return False
    lower, upper = get_joint_limits(body, [joint], physics_client_id)
    assert len(lower) == len(upper) == 1
    return (value < lower[0]) or (upper[0] < value)


def violates_limits(body: int, joints: List[int], values: JointsState,
                    physics_client_id: int) -> bool:
    return any(
        violates_limit(body, joint, value, physics_client_id)
        for joint, value in zip(joints, values))


def _get_base_from_ee(
    robot: SingleArmPyBulletRobot,
    ikfast_info: IKFastInfo,
    tool_link: int,
    world_from_target: Pose,
) -> Pose:
    ee_link = get_link_from_name(robot.robot_id, ikfast_info.ee_link,
                                 robot.physics_client_id)
    tool_from_ee = get_relative_link_pose(robot.robot_id, ee_link, tool_link,
                                          robot.physics_client_id)
    world_from_base = get_link_pose(
        robot.robot_id,
        get_link_from_name(robot.robot_id, ikfast_info.base_link,
                           robot.physics_client_id),
        robot.physics_client_id,
    )

    base_from_ee = multiply_poses(world_from_base.invert(), world_from_target,
                                  tool_from_ee)
    return base_from_ee


# TODO: type hints here below


def get_difference_fn(
    body: int, joints: List[int], physics_client_id: int
) -> Callable[[JointsState, JointsState], Tuple[float]]:
    circular_joints = [
        joint_info.is_circular()
        for joint_info in get_joint_infos(body, joints, physics_client_id)
    ]

    def fn(q2: JointsState, q1: JointsState) -> Tuple[float]:
        return tuple(
            circular_difference(value2, value1) if circular else (value2 -
                                                                  value1)
            for circular, value2, value1 in zip(circular_joints, q2, q1))

    return fn


def circular_interval(lower=-np.pi):
    return (lower, lower + 2 * np.pi)


def wrap_interval(value, interval=(0., 1.)):
    lower, upper = interval
    if (lower == -np.pi) and (+np.pi == upper):
        return value
    assert -np.pi < lower <= upper < +np.pi
    return (value - lower) % (upper - lower) + lower


def circular_difference(theta2, theta1, **kwargs) -> float:
    interval = circular_interval(**kwargs)
    #extent = get_interval_extent(interval) # TODO: combine with motion_planners
    extent = get_aabb_extent(interval)
    diff_interval = (-extent / 2, +extent / 2)
    difference = wrap_interval(theta2 - theta1, interval=diff_interval)
    #difference = interval_difference(theta2, theta1, interval=interval)
    return difference


def get_aabb_extent(aabb):
    lower, upper = aabb
    return np.array(upper) - np.array(lower)


def convex_combination(x, y, w=0.5):
    return (1 - w) * np.array(x) + w * np.array(y)


def interval_generator(lower, upper):
    assert len(lower) == len(upper)
    assert np.less_equal(lower, upper).all()
    if np.equal(lower, upper).all():
        return iter([lower])
    return (convex_combination(lower, upper, w=weights)
            for weights in unit_generator(d=len(lower)))


def unit_generator(d):
    return uniform_generator(d)


def uniform_generator(d):
    while True:
        yield np.random.uniform(size=d)


def get_ik_joints(robot: SingleArmPyBulletRobot, ikfast_info: IKFastInfo,
                  tool_link: int) -> List[int]:
    """Returns the joint IDs of the robot's joints that are used in IKFast."""
    robot_id = robot.robot_id
    physics_client_id = robot.physics_client_id

    # Get joints between base and ee, and ensure no joints between ee and tool
    # Ensure no joints between ee and tool
    base_link = get_link_from_name(robot_id, ikfast_info.base_link,
                                   physics_client_id)
    ee_link = get_link_from_name(robot_id, ikfast_info.ee_link,
                                 physics_client_id)

    ee_ancestors = get_ordered_ancestors(robot_id, ee_link)
    tool_ancestors = get_ordered_ancestors(robot_id, tool_link)
    [first_joint] = [
        parent_joint_from_link(link) for link in tool_ancestors
        if get_link_parent(robot_id, parent_joint_from_link(link),
                           physics_client_id) == base_link
    ]
    assert prune_fixed_joints(robot_id, ee_ancestors,
                              robot.physics_client_id) == prune_fixed_joints(
                                  robot_id, tool_ancestors,
                                  robot.physics_client_id)
    # assert base_link in ee_ancestors # base_link might be -1
    ik_joints = prune_fixed_joints(
        robot_id, ee_ancestors[ee_ancestors.index(first_joint):],
        robot.physics_client_id)
    free_joints = robot.joints_from_names(ikfast_info.free_joints)
    assert set(free_joints) <= set(ik_joints)
    assert len(ik_joints) == 6 + len(free_joints)
    return ik_joints


def ikfast_inverse_kinematics(
        robot: SingleArmPyBulletRobot,
        world_from_target: Pose,
        tool_link: int,
        fixed_joints: Sequence[int] = (),
) -> Generator[JointsState, None, None]:
    """Run IKFast to compute joint positions for given target pose specified in
    the world frame.

    Note that this will automatically compile IKFast for the given robot
    if it hasn't been compiled already when this function is called for
    the first time.
    """
    ikfast_info = robot.ikfast_info()
    ikfast = import_ikfast(ikfast_info)

    max_time = CFG.ikfast_max_time
    max_distance = CFG.ikfast_max_distance
    max_attempts = CFG.ikfast_max_attempts
    norm = CFG.ikfast_norm_ord

    og_robot = robot
    robot: int = robot.robot_id
    ik_joints = get_ik_joints(og_robot, ikfast_info, tool_link)
    free_joints = og_robot.joints_from_names(ikfast_info.free_joints)

    base_from_ee = _get_base_from_ee(og_robot, ikfast_info, tool_link,
                                     world_from_target)
    difference_fn = get_difference_fn(robot, ik_joints,
                                      og_robot.physics_client_id)

    current_conf = og_robot.get_joints(ik_joints)
    current_positions = og_robot.get_joints(free_joints)

    # TODO: handle circular joints
    # TODO: use norm=INF to limit the search for free values
    free_deltas = np.array([
        0.0 if joint in fixed_joints else max_distance for joint in free_joints
    ])
    lower_limits = np.maximum(
        get_joint_lower_limits(robot, free_joints, og_robot.physics_client_id),
        current_positions - free_deltas)
    upper_limits = np.minimum(
        get_joint_upper_limits(robot, free_joints, og_robot.physics_client_id),
        current_positions + free_deltas)
    generator = chain(
        [current_positions],  # TODO: sample from a truncated Gaussian nearby
        interval_generator(lower_limits, upper_limits),
    )
    if max_attempts < np.inf:
        generator = islice(generator, max_attempts)

    start_time = time.perf_counter()

    for free_positions in generator:
        # Exceeded time to generate an IK solution
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time >= max_time:
            break

        # Get IK solutions
        rot_matrix = matrix_from_quat(base_from_ee.quat_xyzw,
                                      og_robot.physics_client_id).tolist()

        ik_candidates = ikfast.get_ik(rot_matrix, list(base_from_ee.position),
                                      list(free_positions))
        if ik_candidates is None:
            continue

        random.shuffle(ik_candidates)

        for conf in ik_candidates:
            difference = difference_fn(current_conf, conf)
            if not violates_limits(robot, ik_joints, conf,
                                   og_robot.physics_client_id) and (get_length(
                                       difference, norm=norm) <= max_distance):
                yield conf


def ikfast_closest_inverse_kinematics(
        robot: SingleArmPyBulletRobot, tool_link: int,
        world_from_target: Pose) -> List[JointsState]:
    """Runs IKFast and returns the solutions sorted in order of closets
    distance to the robot's current joint positions.

    Parameters
    ----------
    robot: SingleArmPyBulletRobot
    tool_link: pybullet link ID of the tool link of the robot
    world_from_target: target pose in the world frame

    Returns
    -------
    A list of joint states that satisfy the given arguments.
    If no solutions are found, an empty list is returned.
    """
    ikfast_info = robot.ikfast_info()
    robot_id = robot.robot_id
    start_time = time.perf_counter()

    norm = CFG.ikfast_norm_ord

    ik_joints = get_ik_joints(robot, ikfast_info, tool_link)
    generator = ikfast_inverse_kinematics(
        robot,
        world_from_target,
        tool_link,
    )

    # Only use up to the max candidates specified
    candidate_solutions = list(islice(generator, CFG.ikfast_max_candidates))
    if not candidate_solutions:
        return []

    # Sort the solutions by distance to the current joint positions
    current_conf = robot.get_joints(ik_joints)

    # TODO: relative to joint limits
    difference_fn = get_difference_fn(
        robot_id, ik_joints, robot.physics_client_id)  # get_distance_fn
    solutions = sorted(
        candidate_solutions,
        key=lambda q: get_length(difference_fn(q, current_conf), norm=norm),
    )
    verbose = True
    if verbose:
        min_distance = min([np.inf] + [
            get_length(difference_fn(q, current_conf), norm=norm)
            for q in solutions
        ])
        elapsed_time = time.perf_counter() - start_time
        logging.debug(
            "Identified {} IK solutions with minimum distance of {:.3f} in {:.3f} seconds"
            .format(len(solutions), min_distance, elapsed_time))

    return solutions
