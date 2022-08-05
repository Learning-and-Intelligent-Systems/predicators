"""Handles the creation of robots."""
from typing import Sequence

from predicators.src.pybullet_helpers.robots.single_arm import \
    FetchPyBulletRobot, SingleArmPyBulletRobot
from predicators.src.structs import Pose3D


def create_single_arm_pybullet_robot(
        robot_name: str, ee_home_pose: Pose3D, ee_orientation: Sequence[float],
        physics_client_id: int) -> SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    if robot_name == "fetch":
        return FetchPyBulletRobot(ee_home_pose, ee_orientation,
                                  physics_client_id)
    raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")
