import logging
import time
from itertools import chain, islice
from typing import TYPE_CHECKING, Any, Generator, Iterator, List, Sequence

import numpy as np
from pybullet_tools.ikfast.ikfast import get_base_from_ee, get_ik_joints
from pybullet_tools.ikfast.utils import compute_inverse_kinematics
from pybullet_tools.utils import INF, Pose, elapsed_time, get_difference_fn, \
    get_joint_positions, get_length, get_max_limits, get_min_limits, \
    interval_generator, joints_from_names, randomize, violates_limits

from predicators.src.pybullet_helpers.ikfast import IKFastInfo
from predicators.src.pybullet_helpers.ikfast.load import import_ikfast
from predicators.src.settings import CFG
from predicators.src.structs import JointsState, Pose

"""
Note: I copied this from Caelan's stuff and hacked it to get it working for us
Should discuss how we want to do things in terms of pybullet utils.
"""


def ikfast_inverse_kinematics(
    ikfast_info: IKFastInfo,
    robot: int,
    world_from_target: Pose,
    tool_link: int,
    fixed_joints: Sequence[int] = (),
    max_attempts: int = INF,
    max_distance: float = INF,
    max_time: float = INF,
    norm: float = INF,
) -> Generator[JointsState, None, None]:
    """Run IKFast to compute joint positions for given target pose specified in
    the world frame.

    Note that this will automatically compile IKFast for the given robot
    if it hasn't been compiled already when this function is called for
    the first time.
    """
    ikfast = import_ikfast(ikfast_info)

    max_time = CFG.ikfast_max_time
    max_candidates = CFG.ikfast_max_candidates

    ik_joints = get_ik_joints(robot, ikfast_info, tool_link)
    free_joints = joints_from_names(robot, ikfast_info.free_joints)
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
    start_time = time.time()
    for free_positions in generator:
        if max_time < elapsed_time(start_time):
            break
        for conf in randomize(
                compute_inverse_kinematics(ikfast.get_ik, base_from_ee,
                                           free_positions)):
            # solution(robot, ik_joints, conf, tool_link, world_from_target)
            difference = difference_fn(current_conf, conf)
            if not violates_limits(robot, ik_joints, conf) and (get_length(
                    difference, norm=norm) <= max_distance):
                # set_joint_positions(robot, ik_joints, conf)
                yield conf


def ikfast_closest_inverse_kinematics(
        ikfast_info: IKFastInfo, robot_id: int, tool_link: int,
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
    start_time = time.perf_counter()
    norm = INF
    ik_joints = get_ik_joints(robot_id, ikfast_info, tool_link)
    current_conf = get_joint_positions(robot_id, ik_joints)
    generator = ikfast_inverse_kinematics(
        ikfast_info,
        robot_id,
        world_from_target,
        tool_link,
        norm=norm,
        max_time=CFG.ikfast_max_time,
    )

    # Only use up to the max candidates specified
    candidate_solutions = list(islice(generator, CFG.ikfast_max_candidates))
    if not candidate_solutions:
        return []

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
        logging.debug(
            "Identified {} IK solutions with minimum distance of {:.3f} in {:.3f} seconds"
            .format(len(solutions), min_distance, elapsed_time(start_time)))

    return solutions
