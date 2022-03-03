"""Generic utilities for Pybullet."""

from typing import List, Sequence, Tuple
import numpy as np
import pybullet as p


def get_kinematic_chain(robot_id: int, end_effector_id: int,
                        physics_client_id: int) -> List[int]:
    """Get all of the free joints from robot base to end effector.

    Includes the end effector.
    """
    kinematic_chain = []
    while end_effector_id > 0:
        joint_info = p.getJointInfo(robot_id,
                                    end_effector_id,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            kinematic_chain.append(end_effector_id)
        end_effector_id = joint_info[-1]
    return kinematic_chain


def get_joint_ranges(
    body_id: int, joint_indices: Sequence[int], physics_client_id: int
) -> Tuple[Sequence[float], Sequence[float], Sequence[float], Sequence[float]]:
    """Returns lower limits, upper limits, joint ranges, and rest poses."""
    lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)

    for i in range(num_joints):
        joint_info = p.getJointInfo(body_id,
                                    i,
                                    physicsClientId=physics_client_id)

        # Fixed joint so ignore
        qIndex = joint_info[3]
        if qIndex <= -1:
            continue

        ll, ul = -2., 2.
        jr = 2.

        # For simplicity, assume resting state == initial state
        rp = p.getJointState(body_id, i, physicsClientId=physics_client_id)[0]

        # Fix joints that we don't want to move
        if i not in joint_indices:
            ll, ul = rp - 1e-8, rp + 1e-8
            jr = 1e-8

        lower_limits.append(ll)
        upper_limits.append(ul)
        joint_ranges.append(jr)
        rest_poses.append(rp)

    return lower_limits, upper_limits, joint_ranges, rest_poses


def inverse_kinematics(body_id: int, end_effector_id: int,
                       target_position: Sequence[float],
                       target_orientation: Sequence[float],
                       joint_indices: Sequence[int],
                       physics_client_id: int) -> Sequence[float]:
    """Runs IK and returns joint values."""
    lls, uls, jrs, rps = get_joint_ranges(body_id,
                                          joint_indices,
                                          physics_client_id=physics_client_id)

    all_joint_poses = p.calculateInverseKinematics(
        body_id,
        end_effector_id,
        target_position,
        targetOrientation=target_orientation,
        lowerLimits=lls,
        upperLimits=uls,
        jointRanges=jrs,
        restPoses=rps,
        physicsClientId=physics_client_id)

    # Find the free joints.
    free_joint_indices = []

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    for idx in range(num_joints):
        joint_info = p.getJointInfo(body_id,
                                    idx,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            free_joint_indices.append(idx)

    # Find the poses for the joints that we want to move.
    joint_poses = []

    for idx in joint_indices:
        free_joint_idx = free_joint_indices.index(idx)
        joint_pose = all_joint_poses[free_joint_idx]
        joint_poses.append(joint_pose)

    return joint_poses
