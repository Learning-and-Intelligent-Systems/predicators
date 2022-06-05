from typing import List

import numpy as np
import pybullet as p

from predicators.src import utils
from predicators.src.pybullet_helpers.ikfast import ikfast_inverse_kinematics
from predicators.src.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from predicators.src.pybullet_helpers.utils import (
    get_link_from_name,
    get_relative_link_pose,
)
from predicators.src.settings import CFG
from predicators.src.structs import JointsState, Pose3D
from pybullet_tools.utils import Pose, draw_pose, wait_for_user


class PandaPyBulletRobot(SingleArmPyBulletRobot):
    """Franka Emika Panda which we assume is fixed on some base."""

    def _initialize(self) -> None:
        # Base pose and orientation (robot is fixed)
        self._world_from_base = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self._physics_client_id
        )

        # TODO!!! fix this. Hardcoded constant at the moment, we just want it facing down
        # self._ee_orientation = [-1.0, 0.0, 0.0, 0.0]
        # in wxyz i think
        self._ee_orientation = p.getQuaternionFromEuler([-np.pi, 0, np.pi / 2])
        self._ee_orientation_euler = p.getEulerFromQuaternion(self._ee_orientation)
        # Extract IDs for individual robot links and joints.

        # TODO: change this, because it's highly confusing that this is not
        # the tool tip, since end_effector_id is the tool tip.
        self._end_effector_link = get_link_from_name(
            self.robot_id, "panda_link8", self._physics_client_id
        )
        self._tool_link = get_link_from_name(
            self.robot_id, "tool_link", self._physics_client_id
        )

    @classmethod
    def get_name(cls) -> str:
        return "panda"

    @classmethod
    def urdf_path(cls) -> str:
        return utils.get_env_asset_path(
            "urdf/franka_description/robots/panda_arm_hand.urdf"
        )

    @property
    def end_effector_name(self) -> str:
        # TODO explain or change this
        return "tool_joint"

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
        return 0.03

    def inverse_kinematics(
        self, end_effector_pose: Pose3D, validate: bool
    ) -> List[float]:

        # TODO explain what IKFast is doing
        # X_TE
        tool_from_ee = get_relative_link_pose(
            self.robot_id,
            self._end_effector_link,
            self._tool_link,
            self._physics_client_id,
        )
        # print("X_TE", tool_from_ee)

        # X_BE = (X_WB)^-1 * X_WT * X_TE
        base_from_ee = p.multiplyTransforms(
            *p.multiplyTransforms(
                *p.invertTransform(*self._world_from_base),
                # End effector means tool tip here
                end_effector_pose,
                self._ee_orientation,
            ),
            *tool_from_ee,
        )

        try:
            joints_state = ikfast_inverse_kinematics(
                self,
                base_from_ee[0],
                base_from_ee[1],
                physics_client_id=self._physics_client_id,
            )
        except Exception as e:
            pose = Pose(point=end_effector_pose)
            handles = draw_pose(pose)
            wait_for_user("ik fast failed")
            raise e

        # Add fingers to state
        final_joint_state = list(joints_state)
        first_finger_idx, second_finger_idx = sorted(
            [self.left_finger_joint_idx, self.right_finger_joint_idx]
        )
        final_joint_state.insert(first_finger_idx, self.open_fingers)
        final_joint_state.insert(second_finger_idx, self.open_fingers)

        if validate:
            self._validate_joints_state(
                final_joint_state, target_pose=end_effector_pose
            )
        return final_joint_state
