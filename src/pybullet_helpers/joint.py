"""Pybullet helper class for joint utilities."""
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators.src.pybullet_helpers.geometry import Pose3D, Quaternion


class JointInfo(NamedTuple):
    """Joint Information from Pybullet."""
    jointIndex: int
    jointName: str
    jointType: int
    qIndex: int
    uIndex: int
    flags: int
    jointDamping: float
    jointFriction: float
    jointLowerLimit: float
    jointUpperLimit: float
    jointMaxForce: float
    jointMaxVelocity: float
    linkName: str
    jointAxis: Sequence[float]
    parentFramePos: Pose3D
    parentFrameOrn: Quaternion
    parentIndex: int

    def is_circular(self) -> bool:
        """Whether the joint is circular or not."""
        if self.is_fixed():
            return False
        return self.jointUpperLimit < self.jointLowerLimit

    def is_fixed(self) -> bool:
        """Whether the joint is fixed or not."""
        return self.jointType == p.JOINT_FIXED


def get_num_joints(body: int, physics_client_id: int) -> int:
    """Get the number of joints for a body."""
    return p.getNumJoints(body, physicsClientId=physics_client_id)


def get_joints(body: int, physics_client_id: int) -> List[int]:
    """Get joint indices for a body."""
    return list(range(get_num_joints(body, physics_client_id)))


def get_joint_info(body: int, joint: int, physics_client_id: int) -> JointInfo:
    """Get the info for the given joint for a body."""
    raw_joint_info: List = list(
        p.getJointInfo(body, joint, physicsClientId=physics_client_id))
    # Decode the byte strings for joint name and link name
    raw_joint_info[1] = raw_joint_info[1].decode("UTF-8")
    raw_joint_info[12] = raw_joint_info[12].decode("UTF-8")

    joint_info = JointInfo(*raw_joint_info)
    return joint_info


def get_joint_infos(body: int, joints: List[int],
                    physics_client_id: int) -> List[JointInfo]:
    """Get the infos the given joints for a body."""
    return [
        get_joint_info(body, joint_id, physics_client_id)
        for joint_id in joints
    ]


def get_joint_limits(
        body: int, joints: List[int],
        physics_client_id: int) -> Tuple[List[float], List[float]]:
    """Get the joint limits for the given joints for a body. Circular joints do
    not have limits (represented by ±np.pi).

    Returns
    -------
    Tuple with the lower limits as a list, and the upper limits as list.
    """
    joint_infos = get_joint_infos(body, joints, physics_client_id)
    lower_limits = [
        joint.jointLowerLimit if not joint.is_circular() else -np.inf
        for joint in joint_infos
    ]
    upper_limits = [
        joint.jointUpperLimit if not joint.is_circular() else np.inf
        for joint in joint_infos
    ]
    return lower_limits, upper_limits


def get_joint_lower_limits(body: int, joints: List[int],
                           physics_client_id: int) -> List[float]:
    """Get the lower joint limits for the given joints for a body."""
    return get_joint_limits(body, joints, physics_client_id)[0]


def get_joint_upper_limits(body: int, joints: List[int],
                           physics_client_id: int) -> List[float]:
    """Get the upper joint limits for the given joints for a body."""
    return get_joint_limits(body, joints, physics_client_id)[1]
