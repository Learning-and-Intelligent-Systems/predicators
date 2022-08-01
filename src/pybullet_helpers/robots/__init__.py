"""Handles the creation of robots."""
import importlib
import pkgutil
from typing import TYPE_CHECKING, Dict

from predicators.src.pybullet_helpers.geometry import Pose
from predicators.src.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.src.structs import Pose3D, Quaternion
from predicators.src.utils import get_all_concrete_subclasses

if not TYPE_CHECKING:
    # Load all modules so that utils.get_all_concrete_subclasses() works.
    for _, module_name, _ in pkgutil.walk_packages(__path__):
        if "__init__" not in module_name:
            # Important! We use an absolute import here to avoid issues
            # with isinstance checking when using relative imports.
            importlib.import_module(f"{__name__}.{module_name}")

# Note: these are static poses which suffice for the current environments.
_ROBOT_TO_BASE_POSE: Dict[str, Pose] = {
    "fetch": Pose(position=(0.75, 0.7441, 0.0)),
    "panda": Pose(position=(0.8, 0.7441, 0.25)),
}


def create_single_arm_pybullet_robot(
    robot_name: str,
    ee_home_pose: Pose3D,
    ee_orientation: Quaternion,
    physics_client_id: int,
) -> SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    for cls in get_all_concrete_subclasses(SingleArmPyBulletRobot):
        if cls.get_name() == robot_name:
            base_pose = _ROBOT_TO_BASE_POSE.get(cls.get_name(),
                                                Pose.identity())
            robot = cls(ee_home_pose,
                        ee_orientation,
                        physics_client_id,
                        base_pose=base_pose)
            break
    else:
        raise ValueError(f"Unrecognized robot name: {robot_name}.")

    return robot
