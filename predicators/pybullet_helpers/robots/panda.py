"""Franka Emika Panda robot."""
from typing import List, Optional

import numpy as np

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.ikfast import IKFastInfo
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.settings import CFG

# The Franka's canonical "ready" configuration.
PANDA_HOME_ARM_JOINTS = [
    0.0,
    -np.pi / 4,
    0.0,
    -3 * np.pi / 4,
    0.0,
    np.pi / 2,
    np.pi / 4,
]

# The tool_link pose induced by PANDA_HOME_ARM_JOINTS, in the base frame.
# Precomputed so that callers can ask where the Panda's home is before a URDF
# is loaded.
PANDA_HOME_EE_POSE_IN_BASE = Pose(position=(0.3069, 0.0, 0.4903),
                                  orientation=(0.0, 1.0, 0.0, 0.0))


class PandaPyBulletRobot(SingleArmPyBulletRobot):
    """Franka Emika Panda which we assume is fixed on some base."""

    @classmethod
    def get_name(cls) -> str:
        return "panda"

    @classmethod
    def home_arm_joint_positions(cls) -> Optional[List[float]]:
        return list(PANDA_HOME_ARM_JOINTS)

    @classmethod
    def home_ee_pose_in_base(cls) -> Optional[Pose]:
        return PANDA_HOME_EE_POSE_IN_BASE

    @classmethod
    def urdf_path(cls) -> str:
        return utils.get_env_asset_path(
            "urdf/franka_description/robots/panda_arm_hand.urdf")

    @property
    def end_effector_name(self) -> str:
        """The tool joint is offset from the final arm joint such that it
        represents the point in the center of the two fingertips of the gripper
        (fingertips, NOT the entire fingers).

        This differs from the "panda_hand" joint which represents the
        center of the gripper itself including parts of the gripper
        body.
        """
        return "tool_joint"

    @property
    def tool_link_name(self) -> str:
        return "tool_link"

    @property
    def left_finger_joint_name(self) -> str:
        return "panda_finger_joint1"

    @property
    def right_finger_joint_name(self) -> str:
        return "panda_finger_joint2"

    @property
    def open_fingers(self) -> float:
        return 0.04

    @property
    def closed_fingers(self) -> float:
        # Allow a config override so a thin object can be clamped;
        # None keeps the Panda default.
        if CFG.pybullet_closed_fingers is not None:
            return float(CFG.pybullet_closed_fingers)
        return 0.03

    @property
    def finger_motor_force(self) -> Optional[float]:
        # The URDF effort limit of the finger joints. Without a finite
        # cap, position-controlled fingers commanded past a grasped
        # object crush straight through it (the fingers only need to
        # come within grasp_tol of the object; the grasp itself is a
        # fixed constraint).
        return 20.0

    @property
    def push_ee_yaw_offset(self) -> float:
        # The Franka Hand leads with its knuckles spanning the
        # object's face.
        return np.pi / 2

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        return IKFastInfo(
            module_dir="panda_arm",
            module_name="ikfast_panda_arm",
            base_link="panda_link0",
            ee_link="panda_link8",
            free_joints=["panda_joint7"],
        )
