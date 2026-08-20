"""Fetch Robotics Mobile Manipulator (Fetch) with a kinematic mobile base."""

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots.fetch import FetchPyBulletRobot


class MobileFetchPyBulletRobot(FetchPyBulletRobot):
    """A Fetch robot with a kinematic (x, y, theta) mobile base."""

    # Base action corresponds to delta x, delta y, delta theta.
    base_action_dim: int = 3
    base_xy_delta_limit: float = 2.0
    base_yaw_delta_limit: float = np.pi

    # Default controller parameters for base motion.
    default_base_vel_norm: float = 0.2
    default_base_rot_vel: float = np.pi / 4
    default_arm_reach_radius: float = 0.8

    @classmethod
    def get_name(cls) -> str:
        return "mobile_fetch"

    @property
    def action_space(self) -> Box:
        """Action space includes arm joint targets + base deltas."""
        joint_low = np.array(self.joint_lower_limits, dtype=np.float32)
        joint_high = np.array(self.joint_upper_limits, dtype=np.float32)
        base_low = np.array([
            -self.base_xy_delta_limit, -self.base_xy_delta_limit,
            -self.base_yaw_delta_limit
        ],
                            dtype=np.float32)
        base_high = np.array([
            self.base_xy_delta_limit, self.base_xy_delta_limit,
            self.base_yaw_delta_limit
        ],
                             dtype=np.float32)
        low = np.concatenate([joint_low, base_low])
        high = np.concatenate([joint_high, base_high])
        return Box(low, high, dtype=np.float32)

    def get_base_pose(self) -> Pose:
        """Get the current base pose from PyBullet."""
        position, orientation = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client_id)
        return Pose(position, orientation)

    def set_base_pose(self, base_pose: Pose) -> None:
        """Set the base pose in PyBullet."""
        p.resetBasePositionAndOrientation(
            self.robot_id,
            base_pose.position,
            base_pose.orientation,
            physicsClientId=self.physics_client_id,
        )
