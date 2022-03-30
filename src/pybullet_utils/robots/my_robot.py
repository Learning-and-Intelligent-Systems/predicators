from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, List, Dict, Any, ClassVar

import pybullet as p

from predicators.src import utils
from predicators.src.pybullet_utils.robots import _SingleArmPyBulletRobot
from predicators.src.structs import Pose3D


Quaternion = Tuple[float, float, float, float]
RollPitchYaw = Tuple[float, float, float]


@dataclass(frozen=True)
class Pose:
    pos: Pose3D
    quat_xyzw: Quaternion

    @cached_property
    def rpy(self) -> RollPitchYaw:
        return p.getEulerFromQuaternion(self.quat_xyzw)


class MyPandaPyBulletRobot(_SingleArmPyBulletRobot):
    """Franka Emika Panda which we assume is fixed on some base."""

    # Parameters that aren't important enough to need to clog up settings.py
    _base_pose: Pose = Pose(pos=(0.75, 0.7441, 0.2), quat_xyzw=(0.0, 0.0, 0.0, 1.0))

    _ee_orientation: Quaternion[float] = [-1.0, 0.0, 0.0, 0.0]
    _finger_action_nudge_magnitude: float = 0.001

    _robot_urdf: ClassVar[str] = utils.get_env_asset_path(
        "urdf/franka_description/robots/panda_arm_hand.urdf"
    )

    def _initialize(self) -> None:
        self._panda_id = p.loadURDF(
            self._robot_urdf,
            basePosition=self._base_pose.pos,
            baseOrientation=self._base_pose.quat_xyzw,
            useFixedBase=True,
            physicsClientId=self._physics_client_id,
        )

    @property
    def end_effector_id(self) -> int:
        return self.get_joint_index("tool_joint")

    def end_effector_pose(self) -> Pose:
        """Compute end-effector pose"""
        link_state = p.getLinkState(
            self.robot_id, self.end_effector_id, physicsClientId=self._physics_client_id
        )
        pos = link_state[4]
        quat = link_state[5]
        return Pose(pos, quat)

    @cached_property
    def num_joints(self) -> int:
        return p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)

    @cached_property
    def joint_infos(self) -> List[List[Any]]:
        return [
            p.getJointInfo(
                self.robot_id, joint_idx, physicsClientId=self._physics_client_id
            )
            for joint_idx in range(self.num_joints)
        ]

    @cached_property
    def joint_names(self) -> List[str]:
        return [info[1].decode("utf-8") for info in self.joint_infos]

    @cached_property
    def joint_to_index(self) -> Dict[str, int]:
        return {info[1].decode("utf-8"): info[0] for info in self.joint_infos}

    def get_joint_index(self, joint_name: str) -> int:
        if joint_name not in self.joint_to_index:
            raise ValueError(f"Joint {joint_name} not found in robot")

        return self.joint_to_index[joint_name]
