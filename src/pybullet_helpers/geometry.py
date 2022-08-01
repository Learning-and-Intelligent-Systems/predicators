"""Pybullet helper class for geometry utilities."""
from typing import NamedTuple, Sequence

import numpy as np
import pybullet as p
from pybullet_utils.transformations import euler_from_quaternion, \
    quaternion_from_euler

from predicators.src.structs import Array, Pose3D, Quaternion, RollPitchYaw


class Pose(NamedTuple):
    """Pose which is a position (translation) and rotation.

    We use a NamedTuple as it supports retrieving by integer indexing
    and most closely follows the pybullet API.
    """

    position: Pose3D
    quat_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_rpy(cls, translation: Pose3D, rpy: RollPitchYaw) -> "Pose":
        """Create a Pose from translation and Euler roll-pitch-yaw angles."""
        return cls(translation, quaternion_from_euler(*rpy))

    @property
    def orientation(self) -> Quaternion:
        """The default quaternion representation is xyzw as followed by
        pybullet."""
        return self.quat_xyzw

    @property
    def quat_wxyz(self) -> Quaternion:
        """Get the wxyz quaternion representation."""
        return (
            self.quat_xyzw[3],
            self.quat_xyzw[0],
            self.quat_xyzw[1],
            self.quat_xyzw[2],
        )

    @property
    def rpy(self) -> RollPitchYaw:
        """Get the Euler roll-pitch-yaw representation."""
        return euler_from_quaternion(self.quat_xyzw)

    @classmethod
    def identity(cls) -> "Pose":
        """Unit pose."""
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def multiply(self, *poses: "Pose") -> "Pose":
        """Multiplies poses (i.e., rigid transforms) together."""
        return multiply_poses(self, *poses)

    def invert(self) -> "Pose":
        """Invert the pose (i.e., transform)."""
        pos, quat = p.invertTransform(self.position, self.quat_xyzw)
        return Pose(pos, quat)


def multiply_poses(*poses: Pose) -> Pose:
    """Multiplies poses (which are essentially transforms) together."""
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose.position, pose.quat_xyzw,
                                    next_pose.position, next_pose.quat_xyzw)
        pose = Pose(pose[0], pose[1])
    return pose


def matrix_from_quat(quat: Sequence[float]) -> Array:
    """Get 3x3 rotation matrix from quaternion (xyzw)."""
    return np.array(p.getMatrixFromQuaternion(quat, )).reshape(3, 3)


def get_pose(body: int, physics_client_id: int) -> Pose:
    """Get the pose of a body."""
    pybullet_pose = p.getBasePositionAndOrientation(
        body, physicsClientId=physics_client_id)
    return Pose(pybullet_pose[0], pybullet_pose[1])
