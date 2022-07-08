from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
from pybullet_utils.transformations import euler_from_quaternion, \
    quaternion_from_euler

from predicators.src.structs import Array, Pose3D, Quaternion, RollPitchYaw

_BASE_LINK = -1


class Pose(NamedTuple):
    """Pose which is a position (translation) and rotation.

    We use a NamedTuple as it supports retrieving by 0-indexing and most
    closely follows the pybullet API.
    """

    position: Pose3D
    quat_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_rpy(cls, translation: Pose3D, rpy: RollPitchYaw) -> "Pose":
        return cls(translation, quaternion_from_euler(*rpy))

    @property
    def quat(self) -> Quaternion:
        """The default quaternion representation is xyzw as followed by
        pybullet."""
        return self.quat_xyzw

    @property
    def quat_wxyz(self) -> Quaternion:
        return (
            self.quat_xyzw[3],
            self.quat_xyzw[0],
            self.quat_xyzw[1],
            self.quat_xyzw[2],
        )

    @property
    def rpy(self) -> RollPitchYaw:
        return euler_from_quaternion(self.quat_xyzw)

    @classmethod
    def identity(cls) -> "Pose":
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def multiply(self, *poses: "Pose") -> "Pose":
        """Multiplies poses (i.e., rigid transforms) together."""
        return multiply_poses(self, *poses)

    def invert(self) -> "Pose":
        pos, quat = p.invertTransform(self.position, self.quat_xyzw)
        return Pose(pos, quat)


def multiply_poses(*poses: Pose) -> Pose:
    """Multiplies poses together."""
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose.position, pose.quat_xyzw,
                                    next_pose.position, next_pose.quat_xyzw)
        pose = Pose(pose[0], pose[1])
    return pose


def matrix_from_quat(quat: Sequence[float], physics_client_id: int) -> Array:
    return np.array(
        p.getMatrixFromQuaternion(quat,
                                  physicsClientId=physics_client_id)).reshape(
                                      3, 3)


def get_pose(body: int, physics_client_id: int) -> Pose:
    pybullet_pose = p.getBasePositionAndOrientation(
        body, physicsClientId=physics_client_id)
    return Pose(pybullet_pose[0], pybullet_pose[1])


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
        if self.jointType == p.JOINT_FIXED:
            return False
        return self.jointUpperLimit < self.jointLowerLimit


def get_joint_info(body: int, joint: int, physics_client_id: int) -> JointInfo:
    # Decode the byte strings for joint name and link name
    raw_joint_info = list(
        p.getJointInfo(body, joint, physicsClientId=physics_client_id))
    raw_joint_info[1] = raw_joint_info[1].decode("UTF-8")
    raw_joint_info[12] = raw_joint_info[12].decode("UTF-8")

    joint_info = JointInfo(*raw_joint_info)
    return joint_info


def get_joint_infos(body: int, joint: List[int],
                    physics_client_id: int) -> List[JointInfo]:
    return [
        get_joint_info(body, joint_id, physics_client_id) for joint_id in joint
    ]


def get_joint_limits(
        body: int, joints: List[int],
        physics_client_id: int) -> Tuple[List[float], List[float]]:
    joint_infos = get_joint_infos(body, joints, physics_client_id)
    lower_limits = []
    upper_limits = []

    for joint_info in joint_infos:
        if joint_info.is_circular():
            lower_limits.append(-np.inf)
            upper_limits.append(np.inf)
        else:
            lower_limits.append(joint_info.jointLowerLimit)
            upper_limits.append(joint_info.jointUpperLimit)

    return lower_limits, upper_limits


def get_joint_lower_limits(body: int, joints: List[int],
                           physics_client_id: int) -> List[float]:
    return get_joint_limits(body, joints, physics_client_id)[0]


def get_joint_upper_limits(body: int, joints: List[int],
                           physics_client_id: int) -> List[float]:
    return get_joint_limits(body, joints, physics_client_id)[1]


def get_link_from_name(body: int, name: str, physics_client_id: int) -> int:
    """Get the link ID from the name of the link."""
    base_info = p.getBodyInfo(body, physicsClientId=physics_client_id)
    base_name = base_info[0].decode(encoding="UTF-8")
    if name == base_name:
        return -1  # base link
    for joint in range(p.getNumJoints(body,
                                      physicsClientId=physics_client_id)):
        joint_info = get_joint_info(body, joint, physics_client_id)
        if joint_info.linkName == name:
            # Note: link index is the same as joint index in pybullet
            link = joint
            return link
    raise ValueError(f"Body {body} has no link with name {name}.")


def get_link_pose(body: int, link: int, physics_client_id: int) -> Pose:
    """Get the position and orientation for a link."""
    if link == _BASE_LINK:
        return get_pose(body, physics_client_id)
    link_state = p.getLinkState(body, link, physicsClientId=physics_client_id)
    return Pose(link_state[0], link_state[1])


def get_link_parent(body: int, link: int,
                    physics_client_id: int) -> Optional[int]:
    """Get the parent link index of the given link."""
    if link == _BASE_LINK:
        return None
    joint_info = get_joint_info(body, link, physics_client_id)
    return joint_info.parentIndex


def get_relative_link_pose(body: int, link1: int, link2: int,
                           physics_client_id: int) -> Pose:
    """Get the pose of one link relative to another link on the same body."""
    world_from_link1 = get_link_pose(body, link1, physics_client_id)
    world_from_link2 = get_link_pose(body, link2, physics_client_id)
    link2_from_link1 = multiply_poses(world_from_link2.invert(),
                                      world_from_link1)
    return link2_from_link1


def get_kinematic_chain(robot: int, end_effector: int,
                        physics_client_id: int) -> List[int]:
    """Get all of the free joints from robot base to end effector.

    Includes the end effector.
    """
    kinematic_chain = []
    while end_effector > -1:
        joint_info = p.getJointInfo(robot,
                                    end_effector,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            kinematic_chain.append(end_effector)
        end_effector = joint_info[-1]
    return kinematic_chain
