from __future__ import annotations

import logging
import random
import time
from itertools import chain, islice
from typing import TYPE_CHECKING, Generator, List, Sequence

import numpy as np
from pybullet_tools.ikfast.ikfast import get_base_from_ee
from pybullet_tools.ikfast.utils import compute_inverse_kinematics
from pybullet_tools.utils import INF, get_difference_fn, get_joint_positions, \
    get_length, get_max_limits, get_min_limits, get_ordered_ancestors, \
    interval_generator, joints_from_names, parent_joint_from_link, \
    parent_link_from_joint, point_from_pose, prune_fixed_joints, randomize, \
    violates_limits

from predicators.src.pybullet_helpers.ikfast import IKFastInfo
from predicators.src.pybullet_helpers.ikfast.load import import_ikfast
from predicators.src.pybullet_helpers.utils import get_link_from_name, \
    matrix_from_quat
from predicators.src.settings import CFG
from predicators.src.structs import JointsState, Pose

"""
Note: I copied this from Caelan's stuff and hacked it to get it working for us
Should discuss how we want to do things in terms of pybullet utils.
"""

if TYPE_CHECKING:
    from predicators.src.pybullet_helpers.robots import SingleArmPyBulletRobot


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
    robot = robot.robot_id
    ik_joints = get_ik_joints(robot, ikfast_info, tool_link,
                              og_robot.physics_client_id)
    free_joints = joints_from_names(robot, ikfast_info.free_joints)
    og_world_from_target = world_from_target
    world_from_target = (world_from_target.position,
                         world_from_target.quat_xyzw)
    base_from_ee = get_base_from_ee(robot, ikfast_info, tool_link,
                                    world_from_target)
    difference_fn = get_difference_fn(robot, ik_joints)
    current_conf = get_joint_positions(robot, ik_joints)
    current_positions = get_joint_positions(robot, free_joints)

    # TODO: handle circular joints
    # TODO: use norm=INF to limit the search for free values
    free_deltas = np.array([
        0.0 if joint in fixed_joints else max_distance for joint in free_joints
    ])
    lower_limits = np.maximum(get_min_limits(robot, free_joints),
                              current_positions - free_deltas)
    upper_limits = np.minimum(get_max_limits(robot, free_joints),
                              current_positions + free_deltas)
    generator = chain(
        [current_positions],  # TODO: sample from a truncated Gaussian nearby
        interval_generator(lower_limits, upper_limits),
    )
    if max_attempts < INF:
        generator = islice(generator, max_attempts)

    start_time = time.perf_counter()

    for free_positions in generator:
        # Exceeded time to generate an IK solution
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time >= max_time:
            break

        # Get IK solutions
        rot_matrix = matrix_from_quat(base_from_ee[1],
                                      og_robot.physics_client_id).tolist()

        ik_candidates = ikfast.get_ik(rot_matrix, list(base_from_ee[0]),
                                      list(free_positions))
        if ik_candidates is None:
            continue

        random.shuffle(ik_candidates)

        for conf in ik_candidates:
            difference = difference_fn(current_conf, conf)
            if not violates_limits(robot, ik_joints, conf) and (get_length(
                    difference, norm=norm) <= max_distance):
                # set_joint_positions(robot, ik_joints, conf)
                yield conf


def ikfast_closest_inverse_kinematics(
        robot: SingleArmPyBulletRobot, tool_link: int,
        world_from_target: Pose) -> List[JointsState]:
    """Runs IKFast and returns the solutions sorted in order of closets
    distance to the robot's current joint positions.

    Parameters
    ----------
    ikfast_info: IKFastInfo for the robot
    robot_id: pybullet body ID of the robot
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

    ik_joints = get_ik_joints(robot_id, ikfast_info, tool_link,
                              robot.physics_client_id)
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
    current_conf = get_joint_positions(robot_id, ik_joints)

    # TODO: relative to joint limits
    difference_fn = get_difference_fn(robot_id, ik_joints)  # get_distance_fn
    solutions = sorted(
        candidate_solutions,
        key=lambda q: get_length(difference_fn(q, current_conf), norm=norm),
    )
    verbose = True
    if verbose:
        min_distance = min([INF] + [
            get_length(difference_fn(q, current_conf), norm=norm)
            for q in solutions
        ])
        elapsed_time = time.perf_counter() - start_time
        logging.debug(
            "Identified {} IK solutions with minimum distance of {:.3f} in {:.3f} seconds"
            .format(len(solutions), min_distance, elapsed_time))

    return solutions


def get_ik_joints(robot: int, ikfast_info: IKFastInfo, tool_link: int,
                  physics_client_id: int) -> List[int]:
    """Returns the joint IDs of the robot's joints that are used in IKFast."""
    # Get joints between base and ee, and ensure no joints between ee and tool
    # Ensure no joints between ee and tool
    base_link = get_link_from_name(robot, ikfast_info.base_link,
                                   physics_client_id)
    ee_link = get_link_from_name(robot, ikfast_info.ee_link, physics_client_id)

    ee_ancestors = get_ordered_ancestors(robot, ee_link)
    tool_ancestors = get_ordered_ancestors(robot, tool_link)
    [first_joint] = [
        parent_joint_from_link(link)
        for link in tool_ancestors if parent_link_from_joint(
            robot, parent_joint_from_link(link)) == base_link
    ]
    assert prune_fixed_joints(robot, ee_ancestors) == prune_fixed_joints(
        robot, tool_ancestors)
    # assert base_link in ee_ancestors # base_link might be -1
    ik_joints = prune_fixed_joints(
        robot, ee_ancestors[ee_ancestors.index(first_joint):])
    free_joints = joints_from_names(robot, ikfast_info.free_joints)
    assert set(free_joints) <= set(ik_joints)
    assert len(ik_joints) == 6 + len(free_joints)
    return ik_joints
