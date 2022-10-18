"""PyBullet helper class for geometry utilities."""
from __future__ import annotations

from typing import NamedTuple, Tuple

import numpy as np
import pybullet as p
from pybullet_utils.transformations import euler_from_quaternion, \
    quaternion_from_euler

from predicators.structs import Array

Pose3D = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]
RollPitchYaw = Tuple[float, float, float]


class Pose(NamedTuple):
    """Pose which is a position (translation) and rotation.

    We use a NamedTuple as it supports retrieving by integer indexing
    and most closely follows the PyBullet API.
    """
    # Cartesian (x, y, z) position
    position: Pose3D
    # Quaternion in (x, y, z, w) representation
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_rpy(cls, translation: Pose3D, rpy: RollPitchYaw) -> Pose:
        """Create a Pose from translation and Euler roll-pitch-yaw angles."""
        return cls(translation, quaternion_from_euler(*rpy))

    @property
    def rpy(self) -> RollPitchYaw:
        """Get the Euler roll-pitch-yaw representation."""
        return euler_from_quaternion(self.orientation)

    @classmethod
    def identity(cls) -> Pose:
        """Unit pose."""
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def multiply(self, *poses: Pose) -> Pose:
        """Multiplies poses (i.e., rigid transforms) together."""
        return multiply_poses(self, *poses)

    def invert(self) -> Pose:
        """Invert the pose (i.e., transform)."""
        pos, quat = p.invertTransform(self.position, self.orientation)
        return Pose(pos, quat)

    def allclose(self, other: Pose, atol: float = 1e-6) -> bool:
        """Return whether this pose is close enough to another pose."""
        return np.allclose(self.position, other.position, atol=atol) and \
            np.allclose(self.orientation, other.orientation, atol=atol)


def multiply_poses(*poses: Pose) -> Pose:
    """Multiplies poses (which are essentially transforms) together."""
    pose = poses[0]
    for next_pose in poses[1:]:
        pybullet_pose = p.multiplyTransforms(pose.position, pose.orientation,
                                             next_pose.position,
                                             next_pose.orientation)
        pose = Pose(pybullet_pose[0], pybullet_pose[1])
    return pose


def matrix_from_quat(quat: Quaternion) -> Array:
    """Get 3x3 rotation matrix from quaternion (xyzw)."""
    return np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)


def get_pose(body: int, physics_client_id: int) -> Pose:
    """Get the pose of a body."""
    pybullet_pose = p.getBasePositionAndOrientation(
        body, physicsClientId=physics_client_id)
    return Pose(pybullet_pose[0], pybullet_pose[1])
