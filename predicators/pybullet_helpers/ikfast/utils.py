"""This module wraps around IKFast so we can run inverse kinematics for our
desired pose.

The code here is based off the pybullet-planning repository by Caelan
Garrett, but has been heavily modified and simplified to suit our more
minimal use case.

https://github.com/caelan/pybullet-planning/blob/master/pybullet_tools/ikfast/ikfast.py
"""
from __future__ import annotations

import logging
import random
import time
from itertools import islice
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, \
    Sequence, Tuple, Union

import numpy as np

from predicators.pybullet_helpers.geometry import Pose, matrix_from_quat, \
    multiply_poses
from predicators.pybullet_helpers.ikfast.load import import_ikfast
from predicators.pybullet_helpers.joint import JointInfo, JointPositions, \
    get_joint_lower_limits, get_joint_positions, get_joint_upper_limits
from predicators.pybullet_helpers.link import get_link_pose, \
    get_relative_link_pose
from predicators.settings import CFG

if TYPE_CHECKING:
    from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot


def get_difference_fn(
    joint_infos: List[JointInfo]
) -> Callable[[JointPositions, JointPositions], JointPositions]:
    """Determine the difference between two joint positions.

    Note: we do not support circular joints.
    """
    if any(joint_info.is_circular for joint_info in joint_infos):
        raise ValueError("Circular joints are not supported yet")

    def fn(q2: JointPositions, q1: JointPositions) -> JointPositions:
        if not len(q2) == len(q1) == len(joint_infos):
            raise ValueError("q2, q1, and joint infos must be the same length")
        diff = list((value2 - value1) for value2, value1 in zip(q2, q1))
        return diff

    return fn


def violates_joint_limits(joint_infos: List[JointInfo],
                          conf: JointPositions) -> bool:
    """Check if the given configuration violate the joint limits."""
    if len(joint_infos) != len(conf):
        raise ValueError("Joint Infos and values must be the same length")
    return any(
        joint_info.violates_limit(value)
        for joint_info, value in zip(joint_infos, conf))


def get_ordered_ancestors(robot: SingleArmPyBulletRobot,
                          link: int) -> List[int]:
    """Get the ancestors of the given link in order from ancestor to the given
    link itself.

    The returned link ordering excludes the base link, but includes the
    given link as the last element.
    """
    ikfast_info = robot.ikfast_info()
    if ikfast_info is None:
        # Keep mypy happy
        raise ValueError(f"Robot {robot.get_name()} has no IKFast info")

    # Mapping of link ID to parent link ID for each link in the robot
    link_to_parent_link: Dict[int, int] = {
        info.jointIndex: info.parentIndex
        for info in robot.joint_infos
    }
    base_link = robot.link_from_name(ikfast_info.base_link)

    # Get ancestors of given link
    current_link = link
    ancestors_reversed = [current_link]
    while link_to_parent_link[current_link] != base_link:
        current_link = link_to_parent_link[current_link]
        ancestors_reversed.append(current_link)

    # Return ordering with ancestor -> ... -> grandparent -> parent -> link
    ancestors = list(reversed(ancestors_reversed))
    return ancestors


def get_base_from_ee(
    robot: SingleArmPyBulletRobot,
    tool_link: int,
    world_from_target: Pose,
) -> Pose:
    """Transform the target tool link pose from the world frame into the pose
    of the end-effector link in the base link frame."""
    ikfast_info = robot.ikfast_info()
    if ikfast_info is None:
        # Keep mypy happy
        raise ValueError(f"Robot {robot.get_name()} has no IKFast info")

    ee_link = robot.link_from_name(ikfast_info.ee_link)
    base_link = robot.link_from_name(ikfast_info.base_link)

    # Pose of end effector in the tool link frame
    tool_from_ee = get_relative_link_pose(robot.robot_id, ee_link, tool_link,
                                          robot.physics_client_id)
    # Pose of base link in the world frame
    world_from_base = get_link_pose(
        robot.robot_id,
        base_link,
        robot.physics_client_id,
    )
    # Pose of end effector in the base link frame
    base_from_ee = multiply_poses(world_from_base.invert(), world_from_target,
                                  tool_from_ee)
    return base_from_ee


def get_ikfast_joints(
        robot: SingleArmPyBulletRobot
) -> Tuple[List[JointInfo], List[JointInfo]]:
    """Determines the joints that are used by IKFast for the given robot.

    Assumptions:
    - There are no joints between the end effector and the tool.
    - Parent of the first link is the base link.

    Returns a tuple where the first element is the list of IK joints for IKFast,
    and the second element is the list of free joints to sample over.
    """
    ikfast_info = robot.ikfast_info()
    if ikfast_info is None:
        # Keep mypy happy
        raise ValueError(f"Robot {robot.get_name()} has no IKFast info")

    ee_link = robot.link_from_name(ikfast_info.ee_link)
    # Get the ancestors of the end effector link (excluding base link)
    ee_ancestors = get_ordered_ancestors(robot, ee_link)

    # Prune out the fixed joints
    ik_joints = [
        joint_info for joint_info in robot.joint_infos
        if joint_info.jointIndex in ee_ancestors and not joint_info.is_fixed
    ]
    free_joints = [
        robot.joint_info_from_name(joint) for joint in ikfast_info.free_joints
    ]
    assert len(ik_joints) == 6 + len(free_joints)

    return ik_joints, free_joints


def free_joints_generator(
    robot: SingleArmPyBulletRobot,
    free_joint_infos: List[JointInfo],
    max_distance: float,
    fixed_joints: Sequence[int] = ()
) -> Iterator[Union[JointPositions, np.ndarray]]:
    """A generator that samples configurations for free joints in the given
    robot that are within the joint limits."""
    free_joints = [joint_info.jointIndex for joint_info in free_joint_infos]
    current_positions = get_joint_positions(robot.robot_id, free_joints,
                                            robot.physics_client_id)
    # Maximum distance between each free joint from current position
    free_deltas = np.array([
        0.0 if joint in fixed_joints else max_distance for joint in free_joints
    ])

    # Determine lower and upper limits
    lower_limits = np.maximum(
        get_joint_lower_limits(robot.robot_id, free_joints,
                               robot.physics_client_id),
        current_positions - free_deltas,
    )
    upper_limits = np.minimum(
        get_joint_upper_limits(robot.robot_id, free_joints,
                               robot.physics_client_id),
        current_positions + free_deltas,
    )
    assert np.less_equal(lower_limits, upper_limits).all()

    # First return the current free joint positions as it may
    # already satisfy the constraints
    yield current_positions

    if np.equal(lower_limits, upper_limits).all():
        # No need to sample if all limits are the same
        yield lower_limits
    else:
        # Note: Caelan used convex combination to sample, but uniform
        # sampling is sufficient for our use case.
        while True:
            yield np.random.uniform(lower_limits, upper_limits)


def ikfast_inverse_kinematics(
    robot: SingleArmPyBulletRobot,
    world_from_target: Pose,
    max_time: float,
    max_distance: float,
    max_attempts: int,
    norm: int,
) -> Iterator[JointPositions]:
    """Run IKFast to compute joint positions for given target pose specified in
    the world frame.

    Note that this will automatically compile IKFast for the given robot
    if it hasn't been compiled already when this function is called for
    the first time.
    """
    ikfast_info = robot.ikfast_info()
    if ikfast_info is None:
        # Keep mypy happy
        raise ValueError(f"Robot {robot.get_name()} has no IKFast info")

    # Get the IKFast module for this robot
    ikfast = import_ikfast(ikfast_info)

    ik_joint_infos, free_joint_infos = get_ikfast_joints(robot)
    ik_joints = [joint_info.jointIndex for joint_info in ik_joint_infos]

    tool_link = robot.tool_link_id

    # Get the desired pose of the end-effector in the base frame
    base_from_ee = get_base_from_ee(robot, tool_link, world_from_target)
    position = list(base_from_ee.position)
    rot_matrix = matrix_from_quat(base_from_ee.orientation).tolist()

    # Sampler for free joints
    generator = free_joints_generator(robot, free_joint_infos, max_distance)
    if max_attempts < np.inf:
        generator = islice(generator, max_attempts)

    difference_fn = get_difference_fn(ik_joint_infos)
    current_conf = get_joint_positions(robot.robot_id, ik_joints,
                                       robot.physics_client_id)

    start_time = time.perf_counter()
    for free_positions in generator:
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time >= max_time:
            logging.warning("Max time reached. No IKFast solution found.")
            break

        # Call IKFast to compute candidates for sampled free joint positions
        ik_candidates: Optional[List[
            List[float]]] = ikfast.get_ik(  # type: ignore
                rot_matrix, position, list(free_positions))
        if ik_candidates is None:
            continue

        # Shuffle the candidates to avoid any biases
        random.shuffle(ik_candidates)

        # Check candidates are valid
        for conf in ik_candidates:
            difference = difference_fn(current_conf, conf)
            if not violates_joint_limits(ik_joint_infos, conf) and (
                    np.linalg.norm(difference, ord=norm) <= max_distance):
                yield conf


def ikfast_closest_inverse_kinematics(
        robot: SingleArmPyBulletRobot,
        world_from_target: Pose) -> List[JointPositions]:
    """Runs IKFast and returns the solutions sorted in order of closets
    distance to the robot's current joint positions.

    Parameters
    ----------
    robot: SingleArmPyBulletRobot
    world_from_target: target pose of the tool link in the world frame

    Returns
    -------
    A list of joint states that satisfy the given arguments.
    If no solutions are found, an empty list is returned.
    """
    start_time = time.perf_counter()
    ik_joint_infos, _ = get_ikfast_joints(robot)
    ik_joints = [joint_info.jointIndex for joint_info in ik_joint_infos]
    current_conf = get_joint_positions(robot.robot_id, ik_joints,
                                       robot.physics_client_id)

    generator = ikfast_inverse_kinematics(
        robot,
        world_from_target,
        max_time=CFG.ikfast_max_time,
        max_distance=CFG.ikfast_max_distance,
        max_attempts=CFG.ikfast_max_attempts,
        norm=CFG.ikfast_norm,
    )

    # Only use up to the max candidates specified
    if CFG.ikfast_max_candidates < np.inf:
        generator = islice(generator, CFG.ikfast_max_candidates)

    # Sort solutions by distance to current joint positions
    candidate_solutions = list(generator)
    difference_fn = get_difference_fn(ik_joint_infos)
    solutions = sorted(
        candidate_solutions,
        key=lambda conf: np.linalg.norm(  # type: ignore
            difference_fn(current_conf, conf),
            ord=CFG.ikfast_norm))
    elapsed_time = time.perf_counter() - start_time

    if solutions:
        min_distance = np.linalg.norm(difference_fn(current_conf,
                                                    solutions[0]),
                                      ord=CFG.ikfast_norm)
        logging.debug(
            f"Identified {len(solutions)} IK solutions with minimum distance "
            f"of {min_distance:.3f} in {elapsed_time:.3f} seconds")
    else:
        logging.warning(f"No IK solutions found in {elapsed_time:.3f} seconds")

    return solutions
