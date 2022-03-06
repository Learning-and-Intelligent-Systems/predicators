"""Generic utilities for Pybullet."""

from typing import List, Sequence, Tuple
import pybullet as p

Pose3D = Tuple[float, float, float]


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


def inverse_kinematics(body_id: int, end_effector_id: int,
                       target_position: Sequence[float],
                       target_orientation: Sequence[float],
                       joint_indices: Sequence[int],
                       physics_client_id: int) -> Sequence[float]:
    """Runs IK and returns joint values for the given joint indices.

    The joint_indices are assumed to be a subset of the free joints.
    """
    free_joint_poses = p.calculateInverseKinematics(
        body_id,
        end_effector_id,
        target_position,
        targetOrientation=target_orientation,
        physicsClientId=physics_client_id)

    # Figure out which joint each dimension of free_joint_poses corresponds to.
    # Note that the kinematic chain for the end effector is a subset of the
    # free joints.
    free_joint_indices = []
    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    for idx in range(num_joints):
        joint_info = p.getJointInfo(body_id,
                                    idx,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            free_joint_indices.append(idx)
    assert len(free_joint_indices) == len(free_joint_poses)
    assert set(joint_indices).issubset(set(free_joint_indices))

    # Find the poses for the joints that we want to move.
    joint_poses = []

    for idx in joint_indices:
        free_joint_idx = free_joint_indices.index(idx)
        joint_pose = free_joint_poses[free_joint_idx]
        joint_poses.append(joint_pose)

    return joint_poses

