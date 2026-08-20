"""Handles the creation of robots."""
from typing import Dict, Optional, Type

from predicators.pybullet_helpers.geometry import Pose, Pose3D, multiply_poses
from predicators.pybullet_helpers.robots.fetch import FetchPyBulletRobot
from predicators.pybullet_helpers.robots.mobile_fetch import \
    MobileFetchPyBulletRobot
from predicators.pybullet_helpers.robots.panda import PandaPyBulletRobot
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.settings import CFG

# Note: these are static base poses which suffice for the current environments.
_ROBOT_TO_BASE_POSE: Dict[str, Pose] = {
    "fetch": Pose(position=(0.75, 0.7441, 0.0)),
    "mobile_fetch": Pose(position=(0.75, 0.7441, 0.0)),
    "panda": Pose(position=(0.8, 0.7441, 0.195)),
}

_ROBOT_TO_CLS: Dict[str, Type[SingleArmPyBulletRobot]] = {
    "fetch": FetchPyBulletRobot,
    "mobile_fetch": MobileFetchPyBulletRobot,
    "panda": PandaPyBulletRobot,
}

# Tuned for the Fetch robot, which does not specify a home position.
_DEFAULT_EE_HOME_POSITION: Pose3D = (1.35, 0.6, 0.7)


def get_robot_home_ee_position(robot_name: str,
                               base_pose: Optional[Pose] = None
                               ) -> Optional[Pose3D]:
    """The world-frame end-effector position of the robot's home configuration,
    or None if the robot has no home configuration.

    Envs use this to place the robot's home (and its initial state)
    where the robot actually rests, rather than at the Fetch-tuned
    position they specify.
    """
    if robot_name not in _ROBOT_TO_CLS:
        raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")
    home_ee_pose_in_base = _ROBOT_TO_CLS[robot_name].home_ee_pose_in_base()
    if home_ee_pose_in_base is None:
        return None
    if base_pose is None:
        base_pose = _ROBOT_TO_BASE_POSE[robot_name]
    return multiply_poses(base_pose, home_ee_pose_in_base).position


def create_single_arm_pybullet_robot(
    robot_name: str,
    physics_client_id: int,
    ee_home_pose: Optional[Pose] = None,
    base_pose: Optional[Pose] = None,
) -> SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    if robot_name not in _ROBOT_TO_CLS:
        raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")
    cls = _ROBOT_TO_CLS[robot_name]
    # Robots with a canonical home configuration (e.g. the Panda) derive their
    # home end-effector pose from it, so only fall back to the position above
    # for robots that have none.
    if ee_home_pose is None and cls.home_arm_joint_positions() is None:
        robot_to_ee_orn = CFG.pybullet_robot_ee_orns[CFG.env]
        assert robot_name in robot_to_ee_orn, \
            f"Default home orn not specified for robot {robot_name}."
        ee_orientation = robot_to_ee_orn[robot_name]
        ee_home_pose = Pose(_DEFAULT_EE_HOME_POSITION, ee_orientation)
    if base_pose is None:
        assert robot_name in _ROBOT_TO_BASE_POSE, \
            f"Base pose not specified for robot {robot_name}."
        base_pose = _ROBOT_TO_BASE_POSE[robot_name]
    return cls(physics_client_id, ee_home_pose, base_pose=base_pose)
