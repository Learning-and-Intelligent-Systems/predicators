from typing import NamedTuple, Optional

import pybullet as p

from predicators.src.pybullet_helpers.geometry import Pose, get_pose, \
    multiply_poses
from predicators.src.pybullet_helpers.joint import get_joint_info, \
    get_joint_infos, get_joints
from predicators.src.structs import Pose3D, Quaternion

BASE_LINK: int = -1


class LinkState(NamedTuple):
    linkWorldPosition: Pose3D
    linkWorldOrientation: Quaternion
    localInertialFramePosition: Pose3D
    localInertialFrameOrientation: Quaternion
    worldLinkFramePosition: Pose3D
    worldLinkFrameOrientation: Quaternion

    def cartesian_pose(self) -> Pose:
        """Cartesian (center of mass) pose of link."""
        return Pose(self.linkWorldPosition, self.linkWorldOrientation)


def get_link_from_name(body: int, name: str, physics_client_id: int) -> int:
    """Get the link ID from the name of the link."""
    base_info = p.getBodyInfo(body, physicsClientId=physics_client_id)
    base_name = base_info[0].decode(encoding="UTF-8")
    if name == base_name:
        return -1  # base link

    joints = get_joints(body, physics_client_id)
    joint_infos = get_joint_infos(body, joints, physics_client_id)

    for joint, joint_info in zip(joints, joint_infos):
        if joint_info.linkName == name:
            # Note: link index is the same as joint index in pybullet
            link = joint
            return link

    raise ValueError(f"Body {body} has no link with name {name}.")


def get_link_state(body: int, link: int, physics_client_id: int) -> LinkState:
    """Get the state of a link in a given body."""
    link_state = p.getLinkState(body, link, physicsClientId=physics_client_id)
    return LinkState(*link_state)


def get_link_pose(body: int, link: int, physics_client_id: int) -> Pose:
    """Get the position and orientation for a link."""
    if link == BASE_LINK:
        return get_pose(body, physics_client_id)

    link_state = get_link_state(body, link, physics_client_id)
    return link_state.cartesian_pose()


def get_link_parent(body: int, link: int,
                    physics_client_id: int) -> Optional[int]:
    """Get the parent link index of the given link."""
    if link == BASE_LINK:
        return None
    joint_info = get_joint_info(body, link, physics_client_id)
    return joint_info.parentIndex


def get_parent_joint_from_link(link: int) -> int:
    """Get the joint ID of the parent joint of a link."""
    # Note: Parent joint index == link index in pybullet
    return link


def get_link_ancestors(body: int, link: int, physics_client_id: int) -> list:
    """Get the ancestors of a link in order of depth (deepest to shallowest).

    Ancestors do not include the link itself.
    """
    parent = get_link_parent(body, link, physics_client_id)
    if parent is None:
        return []
    return get_link_ancestors(body, parent, physics_client_id) + [parent]


def get_relative_link_pose(body: int, link1: int, link2: int,
                           physics_client_id: int) -> Pose:
    """Get the pose of one link relative to another link on the same body."""
    world_from_link1 = get_link_pose(body, link1, physics_client_id)
    world_from_link2 = get_link_pose(body, link2, physics_client_id)
    link2_from_link1 = multiply_poses(world_from_link2.invert(),
                                      world_from_link1)
    return link2_from_link1
