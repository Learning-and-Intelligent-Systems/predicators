"""Franka Emika Panda robot."""
from typing import Optional

from predicators import utils
from predicators.pybullet_helpers.ikfast import IKFastInfo
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.ikfast.utils import \
    ikfast_closest_inverse_kinematics
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import JointPositions
import pybullet as p


class PandaPyBulletRobot(SingleArmPyBulletRobot):
    """Franka Emika Panda which we assume is fixed on some base."""

    @classmethod
    def get_name(cls) -> str:
        return "panda"

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
        return 0.035

    @classmethod
    def ikfast_info(cls) -> Optional[IKFastInfo]:
        return IKFastInfo(
            module_dir="panda_arm",
            module_name="ikfast_panda_arm",
            base_link="panda_link0",
            ee_link="panda_link8",
            free_joints=["panda_joint7"],
        )

    def _ikfast_inverse_kinematics(self,
                                   end_effector_pose: Pose) -> JointPositions:
        """IK using IKFast.

        Returns the joint positions.
        """
        ik_solutions = ikfast_closest_inverse_kinematics(
            self,
            world_from_target=end_effector_pose,
        )

        # Store current joint positions so we can reset
        initial_joint_states = self.get_joints()
        
        filtered_joint_positions = filter(lambda j: not self._check_self_collision(j), map(self._convert_ikfast_to_joint_positions, ik_solutions))
        joint_positions = next(filtered_joint_positions, None)

        # Reset joint positions
        self.set_joints(initial_joint_states)

        if joint_positions is None:
            raise InverseKinematicsError(
                f"No IK solution found for target pose {end_effector_pose} "
                "using IKFast")

        # Use first solution as it is closest to current joint state
        return joint_positions
    
    def _convert_ikfast_to_joint_positions(self, ik_solution: JointPositions) -> JointPositions:
        joint_positions = list(ik_solution)
        first_finger_idx, second_finger_idx = sorted(
            [self.left_finger_joint_idx, self.right_finger_joint_idx])
        joint_positions.insert(first_finger_idx, self.open_fingers)
        joint_positions.insert(second_finger_idx, self.open_fingers)
        return joint_positions

    def _check_self_collision(self, joint_positions: JointPositions) -> bool:
        self.set_joints(joint_positions)
        a = self.link_from_name("panda_link1")
        b = self.link_from_name("panda_link2")
        c = self.link_from_name("panda_link5")
        p.setCollisionFilterPair(self.robot_id, self.robot_id, a, c, True, physicsClientId=self.physics_client_id)
        p.setCollisionFilterPair(self.robot_id, self.robot_id, b, c, True, physicsClientId=self.physics_client_id)
        p.performCollisionDetection(physicsClientId=self.physics_client_id)
        if p.getClosestPoints(self.robot_id, self.robot_id, 0.08, a, c):
            return True
        if p.getClosestPoints(self.robot_id, self.robot_id, 0.08, b, c):
            return True
        return False
