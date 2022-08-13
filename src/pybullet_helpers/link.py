"""Pybullet helper class for link utilities."""
from typing import NamedTuple

import pybullet as p

from predicators.src.pybullet_helpers.geometry import Pose, Pose3D, \
    Quaternion, get_pose

BASE_LINK: int = -1


class LinkState(NamedTuple):
    """Link state from Pybullet."""
    linkWorldPosition: Pose3D
    linkWorldOrientation: Quaternion
    localInertialFramePosition: Pose3D
    localInertialFrameOrientation: Quaternion
    worldLinkFramePosition: Pose3D
    worldLinkFrameOrientation: Quaternion

    def com_pose(self) -> Pose:
        """Center of mass (COM) pose of link."""
        return Pose(self.linkWorldPosition, self.linkWorldOrientation)

    def pose(self) -> Pose:
        """Pose of link in world frame."""
        return Pose(self.worldLinkFramePosition,
                    self.worldLinkFrameOrientation)


def get_link_state(body: int, link: int, physics_client_id: int) -> LinkState:
    """Get the state of a link in a given body."""
    link_state = p.getLinkState(body, link, physicsClientId=physics_client_id)
    return LinkState(*link_state)


def get_link_pose(body: int, link: int, physics_client_id: int) -> Pose:
    """Get the position and orientation for a link."""
    if link == BASE_LINK:
        return get_pose(body, physics_client_id)

    link_state = get_link_state(body, link, physics_client_id)
    return link_state.pose()
