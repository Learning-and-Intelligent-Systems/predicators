from typing import List, \
    NamedTuple, Sequence

import numpy as np
import pybullet as p
from pybullet_utils.transformations import euler_from_quaternion, \
    quaternion_from_euler

from predicators.src.settings import CFG
from predicators.src.structs import Array, JointsState, Pose3D, Quaternion, \
    RollPitchYaw

_BASE_LINK = -1


class Pose(NamedTuple):
    """Pose which is a position (translation) and rotation.

    We use a NamedTuple as it supports retrieving by 0-indexing and most
    closely follows the pybullet API.
    """

    position: Pose3D
    quat_xyzw: Quaternion = (0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_rpy(cls, translation: Pose3D, rpy: RollPitchYaw) -> "Pose":
        return cls(translation, quaternion_from_euler(*rpy))

    @property
    def quat(self) -> Quaternion:
        """The default quaternion representation is xyzw as followed by
        pybullet."""
        return self.quat_xyzw

    @property
    def quat_wxyz(self) -> Quaternion:
        return (
            self.quat_xyzw[3],
            self.quat_xyzw[0],
            self.quat_xyzw[1],
            self.quat_xyzw[2],
        )

    @property
    def rpy(self) -> RollPitchYaw:
        return euler_from_quaternion(self.quat_xyzw)

    @classmethod
    def identity(cls) -> "Pose":
        return cls((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def multiply(self, *poses: "Pose") -> "Pose":
        """Multiplies poses (i.e., rigid transforms) together."""
        return multiply_poses(self, *poses)

    def invert(self) -> "Pose":
        pos, quat = p.invertTransform(self.position, self.quat_xyzw)
        return Pose(pos, quat)


def multiply_poses(*poses: Pose) -> Pose:
    """Multiplies poses together."""
    pose = poses[0]
    for next_pose in poses[1:]:
        pose = p.multiplyTransforms(pose.position, pose.quat_xyzw,
                                    next_pose.position, next_pose.quat_xyzw)
        pose = Pose(pose[0], pose[1])
    return pose


def matrix_from_quat(quat: Sequence[float], physics_client_id: int) -> Array:
    return np.array(
        p.getMatrixFromQuaternion(quat,
                                  physicsClientId=physics_client_id)).reshape(
                                      3, 3)


def get_pose(body: int, physics_client_id: int) -> Pose:
    pybullet_pose = p.getBasePositionAndOrientation(
        body, physicsClientId=physics_client_id)
    return Pose(pybullet_pose[0], pybullet_pose[1])


def get_link_from_name(body: int, name: str, physics_client_id: int) -> int:
    """Get the link ID from the name of the link."""
    base_info = p.getBodyInfo(body, physicsClientId=physics_client_id)
    base_name = base_info[0].decode(encoding="UTF-8")
    if name == base_name:
        return -1  # base link
    for link in range(p.getNumJoints(body, physicsClientId=physics_client_id)):
        joint_info = p.getJointInfo(body,
                                    link,
                                    physicsClientId=physics_client_id)
        joint_name = joint_info[12].decode("UTF-8")
        if joint_name == name:
            return link
    raise ValueError(f"Body {body} has no link with name {name}.")


def get_link_pose(body: int, link: int, physics_client_id: int) -> Pose:
    """Get the position and orientation for a link."""
    if link == _BASE_LINK:
        return get_pose(body, physics_client_id)
    link_state = p.getLinkState(body, link, physicsClientId=physics_client_id)
    return Pose(link_state[0], link_state[1])


def get_relative_link_pose(
        body: int, link1: int, link2: int,
        physics_client_id: int) -> Pose:
    """Get the pose of one link relative to another link on the same body."""
    world_from_link1 = get_link_pose(body, link1, physics_client_id)
    world_from_link2 = get_link_pose(body, link2, physics_client_id)
    link2_from_link1 = multiply_poses(world_from_link2.invert(),
                                      world_from_link1)
    return link2_from_link1


def get_kinematic_chain(robot: int, end_effector: int,
                        physics_client_id: int) -> List[int]:
    """Get all of the free joints from robot base to end effector.

    Includes the end effector.
    """
    kinematic_chain = []
    while end_effector > -1:
        joint_info = p.getJointInfo(robot,
                                    end_effector,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            kinematic_chain.append(end_effector)
        end_effector = joint_info[-1]
    return kinematic_chain


def pybullet_inverse_kinematics(
    robot: int,
    end_effector: int,
    target_position: Pose3D,
    target_orientation: Sequence[float],
    joints: Sequence[int],
    physics_client_id: int,
    validate: bool = True,
) -> JointsState:
    """Runs IK and returns a joints state for the given (free) joints.

    If validate is True, the PyBullet IK solver is called multiple
    times, resetting the robot state each time, until the target
    position is reached. If the target position is not reached after a
    maximum number of iters, an exception is raised.
    """
    # Figure out which joint each dimension of the return of IK corresponds to.
    free_joints = []
    num_joints = p.getNumJoints(robot, physicsClientId=physics_client_id)
    for idx in range(num_joints):
        joint_info = p.getJointInfo(robot,
                                    idx,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            free_joints.append(idx)
    assert set(joints).issubset(set(free_joints))

    # Record the initial state of the joints so that we can reset them after.
    if validate:
        initial_joints_states = p.getJointStates(
            robot, free_joints, physicsClientId=physics_client_id)
        assert len(initial_joints_states) == len(free_joints)

    # Running IK once is often insufficient, so we run it multiple times until
    # convergence. If it does not converge, an error is raised.
    convergence_tol = CFG.pybullet_ik_tol
    for _ in range(CFG.pybullet_max_ik_iters):
        free_joint_vals = p.calculateInverseKinematics(
            robot,
            end_effector,
            target_position,
            targetOrientation=target_orientation,
            physicsClientId=physics_client_id,
        )
        assert len(free_joints) == len(free_joint_vals)
        if not validate:
            break
        # Update the robot state and check if the desired position and
        # orientation are reached.
        for joint, joint_val in zip(free_joints, free_joint_vals):
            p.resetJointState(robot,
                              joint,
                              targetValue=joint_val,
                              physicsClientId=physics_client_id)
        # TODO can this be replaced with get_link_pose?
        ee_link_state = p.getLinkState(
            robot,
            end_effector,
            computeForwardKinematics=True,
            physicsClientId=physics_client_id,
        )
        position = ee_link_state[4]
        # Note: we are checking positions only for convergence.
        if np.allclose(position, target_position, atol=convergence_tol):
            break
    else:
        raise Exception("Inverse kinematics failed to converge.")

    # Reset the joint states to their initial values to avoid modifying the
    # PyBullet internal state.
    if validate:
        for joint, (pos, vel, _, _) in zip(free_joints, initial_joints_states):
            p.resetJointState(
                robot,
                joint,
                targetValue=pos,
                targetVelocity=vel,
                physicsClientId=physics_client_id,
            )
    # Order the found free_joint_vals based on the requested joints.
    joint_vals = []
    for joint in joints:
        free_joint_idx = free_joints.index(joint)
        joint_val = free_joint_vals[free_joint_idx]
        joint_vals.append(joint_val)

    return joint_vals


