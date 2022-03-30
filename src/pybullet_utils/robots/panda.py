from collections import namedtuple
from functools import cached_property
from typing import Sequence, List

import numpy as np
import pybullet as p

from predicators.src import utils
from predicators.src.pybullet_utils.robots import _SingleArmPyBulletRobot
from predicators.src.pybullet_utils.utils import get_kinematic_chain, inverse_kinematics
from predicators.src.structs import Array, Pose3D

from scipy.spatial.transform import Rotation as R


def wait_for_user(message, client):
    return threaded_input(message, client)


def quat2rot(quat):
    """
    Convert quaternion to rotation matrix.

    Args:
        quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).

    Returns:
        np.ndarray: rotation matrix (shape: :math:`[3, 3]`).

    """
    r = R.from_quat(quat)
    if hasattr(r, "as_matrix"):
        return r.as_matrix()
    else:
        return r.as_dcm()


def quat2euler(quat, axes="xyz"):
    """
    Convert quaternion to euler angles.

    Args:
        quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).
        axes (str): Specifies sequence of axes for rotations.
            3 characters belonging to the set {'X', 'Y', 'Z'}
            for intrinsic rotations (rotation about the axes of a
            coordinate system XYZ attached to a moving body),
            or {'x', 'y', 'z'} for extrinsic rotations (rotation about
            the axes of the fixed coordinate system).

    Returns:
        np.ndarray: euler angles (shape: :math:`[3,]`).
    """
    r = R.from_quat(quat)
    return r.as_euler(axes)


def threaded_input(message, client):
    import threading

    data = []
    thread = threading.Thread(target=lambda: data.append(input(message)), args=[])
    thread.start()
    try:
        while thread.is_alive():
            update_viewer(client)
    finally:
        thread.join()
    return data[-1]


MouseEvent = namedtuple(
    "MouseEvent", ["eventType", "mousePosX", "mousePosY", "buttonIndex", "buttonState"]
)


def update_viewer(client):
    return list(
        MouseEvent(*event) for event in p.getMouseEvents(physicsClientId=client)
    )


HOME_POSITION = [
    -0.017792060227770554,
    -0.7601235411041661,
    0.019782607023391807,
    -2.342050140544315,
    0.029840531355804868,
    1.5411935298621688,
    0.7534486589746342,
]
# HOME_POSITION = [-0.19, 0.08, 0.23, -2.43, 0.03, 2.52, 0.86]


class PandaPyBulletRobot(_SingleArmPyBulletRobot):
    """Franka Emika Panda which we assume is fixed on some base."""

    # Parameters that aren't important enough to need to clog up settings.py
    _base_pose: Pose3D = (0.75, 0.7441, 0.2)

    _base_orientation: Sequence[float] = [0.0, 0.0, 0.0, 1.0]
    _ee_orientation: Sequence[float] = [-1.0, 0.0, 0.0, 0.0]
    _finger_action_nudge_magnitude: float = 0.001

    def _initialize(self) -> None:
        self._panda_id = p.loadURDF(
            utils.get_env_asset_path(
                "urdf/franka_description/robots/panda_arm_hand.urdf"
            ),
            basePosition=self._base_pose,
            baseOrientation=self._base_orientation,
            useFixedBase=True,
            physicsClientId=self._physics_client_id,
        )

        # Extract IDs for individual robot links and joints.
        print(self.joint_names)
        self._ee_id = self.joint_names.index("tool_joint")
        self._arm_joints = get_kinematic_chain(
            self._panda_id, self._ee_id, physics_client_id=self._physics_client_id
        )
        self._left_finger_id = self.joint_names.index("panda_finger_joint1")
        self._right_finger_id = self.joint_names.index("panda_finger_joint2")
        self._arm_joints.append(self._left_finger_id)
        self._arm_joints.append(self._right_finger_id)

        self._initial_joint_values = inverse_kinematics(
            self._panda_id,
            self._ee_id,
            self._ee_home_pose,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id,
        )

        self.jnt_to_id = {}
        self.non_fixed_jnt_names = []
        for i in range(
            p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        ):
            info = p.getJointInfo(
                self.robot_id, i, physicsClientId=self._physics_client_id
            )
            jnt_name = info[1].decode("UTF-8")
            self.jnt_to_id[jnt_name] = info[0]

        for _ in range(200):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # Set arm joint motors.
        for joint_name, joint_val in zip(self.joint_names, HOME_POSITION):
            joint_idx = self.jnt_to_id[joint_name]
            print("setting", joint_name, joint_idx, "to", joint_val)
            p.setJointMotorControl2(
                bodyIndex=self._panda_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_val,
                physicsClientId=self._physics_client_id,
            )

        p.setJointMotorControl2(
            bodyIndex=self._panda_id,
            jointIndex=self.jnt_to_id["panda_finger_joint1"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.04,
            physicsClientId=self._physics_client_id,
        )
        p.setJointMotorControl2(
            bodyIndex=self._panda_id,
            jointIndex=self.jnt_to_id["panda_finger_joint2"],
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.04,
            physicsClientId=self._physics_client_id,
        )

        for _ in range(200):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        print("after setting arm joint motors")
        self.print_current_state()

    def print_current_state(self):
        for joint_name, joint in self.jnt_to_id.items():
            state = p.getJointState(
                bodyUniqueId=self._panda_id,
                jointIndex=joint,
                physicsClientId=self._physics_client_id,
            )
            print(joint_name, state)

        pose, quat, rpy = self.get_ee_pose()
        print("ee pose:", pose, rpy)

    @cached_property
    def joint_names(self) -> List[str]:
        return [
            p.getJointInfo(
                self.robot_id, joint_idx, physicsClientId=self._physics_client_id
            )[1].decode("utf-8")
            for joint_idx in range(
                p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
            )
        ]

    def joint_index(self, joint_name: str) -> int:
        """Return the index of the joint with the given name."""
        if joint_name not in self.joint_names:
            raise ValueError(f"Joint {joint_name} not found.")

        return self.joint_names.index(joint_name)

    def get_ee_pose(self):
        """
        Return the end effector pose.

        Returns:
            4-element tuple containing

            - np.ndarray: x, y, z position of the EE (shape: :math:`[3,]`).
            - np.ndarray: quaternion representation of the
              EE orientation (shape: :math:`[4,]`).
            - np.ndarray: rotation matrix representation of the
              EE orientation (shape: :math:`[3, 3]`).
            - np.ndarray: euler angle representation of the
              EE orientation (roll, pitch, yaw with
              static reference frame) (shape: :math:`[3,]`).
        """
        info = p.getLinkState(
            self._panda_id, self._ee_id, physicsClientId=self._physics_client_id
        )
        pos = info[4]
        quat = info[5]

        rot_mat = quat2rot(quat)
        euler = quat2euler(quat, axes="xyz")  # [roll, pitch, yaw]
        return np.array(pos), np.array(quat), euler

    @property
    def robot_id(self) -> int:
        return self._panda_id

    @property
    def end_effector_id(self) -> int:
        return self._ee_id

    @property
    def left_finger_id(self) -> int:
        return self._left_finger_id

    @property
    def right_finger_id(self) -> int:
        return self._right_finger_id

    def reset_state(self, robot_state: Array) -> None:
        rx, ry, rz, rf = robot_state
        p.resetBasePositionAndOrientation(
            self._panda_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id,
        )
        assert np.allclose((rx, ry, rz), self._ee_home_pose)
        joint_values = self._initial_joint_values
        for joint_id, joint_val in zip(self._arm_joints, joint_values):
            p.resetJointState(
                self._panda_id,
                joint_id,
                joint_val,
                physicsClientId=self._physics_client_id,
            )
        for finger_id in [self._left_finger_id, self._right_finger_id]:
            p.resetJointState(
                self._panda_id, finger_id, rf, physicsClientId=self._physics_client_id
            )

    def get_state(self) -> Array:
        ee_link_state = p.getLinkState(
            self._panda_id, self._ee_id, physicsClientId=self._physics_client_id
        )
        rx, ry, rz = ee_link_state[4]
        rf = p.getJointState(
            self._panda_id,
            self._left_finger_id,
            physicsClientId=self._physics_client_id,
        )[0]
        # pose_x, pose_y, pose_z, fingers
        return np.array([rx, ry, rz, rf], dtype=np.float32)

    def set_motors(self, ee_delta: Pose3D, f_delta: float) -> None:
        ee_link_state = p.getLinkState(
            self._panda_id, self._ee_id, physicsClientId=self._physics_client_id
        )
        current_position = ee_link_state[4]
        target_position = np.add(current_position, ee_delta).tolist()

        # We assume that the robot is already close enough to the target
        # position that IK will succeed with one call, so validate is False.
        # Furthermore, updating the state of the robot during simulation, which
        # validate=True would do, is risky and discouraged by PyBullet.
        joint_values = inverse_kinematics(
            self._panda_id,
            self._ee_id,
            target_position,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id,
            validate=False,
        )

        # Set arm joint motors.
        for joint_idx, joint_val in zip(self._arm_joints, joint_values):
            p.setJointMotorControl2(
                bodyIndex=self._panda_id,
                jointIndex=joint_idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_val,
                physicsClientId=self._physics_client_id,
            )

        # Set finger joint motors.
        for finger_id in [self._left_finger_id, self._right_finger_id]:
            current_val = p.getJointState(
                self._panda_id, finger_id, physicsClientId=self._physics_client_id
            )[0]
            # Fingers drift if left alone. If the finger action is near zero,
            # nudge the fingers toward being open or closed, based on which end
            # of the spectrum they are currently closer to.
            if abs(f_delta) < self._finger_action_tol:
                assert self._open_fingers > self._closed_fingers
                if abs(current_val - self._open_fingers) < abs(
                    current_val - self._closed_fingers
                ):
                    nudge = self._finger_action_nudge_magnitude
                else:
                    nudge = -self._finger_action_nudge_magnitude
                target_val = current_val + nudge
            else:
                target_val = current_val + f_delta
            p.setJointMotorControl2(
                bodyIndex=self._panda_id,
                jointIndex=finger_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_val,
                physicsClientId=self._physics_client_id,
            )
