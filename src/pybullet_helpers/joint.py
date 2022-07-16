from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators.src.structs import Pose3D, Quaternion


class JointInfo(NamedTuple):
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
        if self.is_fixed():
            return False
        return self.jointUpperLimit < self.jointLowerLimit

    def is_fixed(self) -> bool:
        return self.jointType == p.JOINT_FIXED

    def is_movable(self) -> bool:
        return not self.is_fixed()


def get_num_joints(body, physics_client_id: int):
    """Get the number of joints for a body."""
    return p.getNumJoints(body, physicsClientId=physics_client_id)


def get_joints(body, physics_client_id: int) -> List[int]:
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


def is_circular_joint(body: int, joint: int, physics_client_id: int) -> bool:
    """Check if the given joint is circular."""
    return get_joint_info(body, joint, physics_client_id).is_circular()


def is_fixed_joint(body: int, joint: int, physics_client_id: int) -> bool:
    """Check if the given joint is fixed."""
    return get_joint_info(body, joint, physics_client_id).is_fixed()


def is_movable_joint(body: int, joint: int, physics_client_id: int) -> bool:
    """Check if the given joint is movable."""
    return get_joint_info(body, joint, physics_client_id).is_movable()
