"""Pose / frame utilities for the real-bench domino env.

Ported (logic unchanged) from ``babyrobot.geometry`` so
``pybullet_domino_real`` is self-contained predicators code -- it imports
nothing from babyrobot. Only the pieces the env task-builder needs are here:
a quaternion-native ``Pose6D`` and the real-base-frame ->
predicators-domino-world transplant.

Conventions:
  * Poses are in the **robot base frame**; rotation stored as ``quat_xyzw``
    (scipy / PyBullet order) -- lossless and singularity-free (a STANDING domino
    has pitch near +-90 deg, which gimbal-locks an euler decomposition).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Pose6D:
    """A rigid pose in the robot base frame (translation + unit quaternion)."""
    xyz: Tuple[float, float, float]
    quat_xyzw: Tuple[float, float, float, float]

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> "Pose6D":
        """Build from a 4x4 homogeneous transform."""
        T = np.asarray(T, dtype=float)
        xyz = tuple(float(v) for v in T[:3, 3])
        quat = tuple(
            float(v) for v in Rotation.from_matrix(T[:3, :3]).as_quat())
        return cls(xyz, quat)  # type: ignore[arg-type]

    def to_matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous transform."""
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(self.quat_xyzw).as_matrix()
        T[:3, 3] = self.xyz
        return T


def domino_upright_yaw(pose: Pose6D) -> float:
    """Env yaw for a STANDING domino: heading of its width (body-y) axis.

    The domino body frame is x=L, y=W, z=H; a standing domino has body-x
    (L) vertical (pitch ~+-90 deg), exactly where a naive euler yaw read
    gimbal-locks. ``to_matrix()`` is exact regardless of orientation, so
    read the horizontal heading of the width axis directly off it.
    """
    w_axis = pose.to_matrix()[:3, 1]  # body-y (W) in world
    return float(np.arctan2(w_axis[1], w_axis[0]))


# The env's domino body is a box with half extents
# ``(domino_width/2, domino_depth/2, domino_height/2)`` and the real dims are
# handed to it as width=W, depth=thickness, height=L, so the two body frames
# are the same box under a permutation of axes:
#
#     env x (width W)      = real y
#     env y (thickness H)  = real z
#     env z (length L)     = real x
#
# Column j is env axis j written in real-body coordinates, so
# ``R_env = R_real @ _REAL_TO_ENV_BODY``. It is a proper rotation (det +1),
# i.e. a relabeling of the same physical box, not a reflection.
_REAL_TO_ENV_BODY = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def domino_env_euler(pose: Pose6D) -> Tuple[float, float, float]:
    """Env ``(roll, pitch, yaw)`` for a domino at ``pose``, standing or not.

    PyBullet's euler convention, so the result is exactly what
    ``_set_state`` writes back via ``getQuaternionFromEuler([roll, pitch,
    yaw])``. Unlike :func:`domino_upright_yaw` this makes no assumption
    that the domino is standing: a knocked-over domino comes back with
    the ``roll`` that put it there, which is the feature
    ``Toppled``/``Upright`` are read off.

    ``roll`` is a rotation about the domino's width axis, which is the
    way a domino physically falls -- so a clean topple onto either face
    is ``roll = +-pi/2`` with ``pitch = 0``, and the env's two angular
    features describe it exactly.

    ``pitch`` is returned only so callers can notice when it is *not*
    negligible. The domino type carries ``yaw`` and ``roll`` and no
    ``pitch``, so any pitch is dropped when the state is written into
    PyBullet; that happens for orientations no free domino reaches on a
    flat table (propped diagonally against a neighbour, say).
    """
    r_env = pose.to_matrix()[:3, :3] @ _REAL_TO_ENV_BODY
    roll, pitch, yaw = Rotation.from_matrix(r_env).as_euler("xyz")
    return float(roll), float(pitch), float(yaw)


# --- real base frame <-> predicators pybullet_domino WORLD frame -------------
# The domino options live in a Fetch-style world: robot base at (0.75, 0.72, z),
# yaw +pi/2, table top z=0.4. A bench Franka has the table BELOW its base (real
# domino z negative), so we transplant the whole real scene by a rigid
# yaw+translation, lifting by ``z_off`` so the real table plane lands on the env
# table top. Rigid => robot<->object geometry (hence planned joints) is
# preserved.
DOMINO_WORLD_ROBOT_XY = (0.75, 0.72)
DOMINO_WORLD_ROBOT_YAW = np.pi / 2
DOMINO_WORLD_TABLE_TOP_Z = 0.4


def domino_world_z_offset(table_z_base: float) -> float:
    """``z_off`` mapping the real table (base-frame height ``table_z_base``,
    typically negative) onto the env table top (0.4)."""
    return DOMINO_WORLD_TABLE_TOP_Z - float(table_z_base)


def base_to_world_transform(z_off: float) -> np.ndarray:
    """4x4 ``T_world_base``: real base at env robot base (xy=(0.75,0.72),
    z=``z_off``) with yaw +pi/2."""
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", DOMINO_WORLD_ROBOT_YAW).as_matrix()
    T[:3,
      3] = (DOMINO_WORLD_ROBOT_XY[0], DOMINO_WORLD_ROBOT_XY[1], float(z_off))
    return T


def pose_base_to_world(pose_base: Pose6D, z_off: float) -> Pose6D:
    """Map a real-base-frame pose into the predicators domino world frame."""
    return Pose6D.from_matrix(
        base_to_world_transform(z_off) @ pose_base.to_matrix())
