"""Vanilla PyBullet Inverse Kinematics.

The IKFast solver is preferred over PyBullet IK, if available for the
given robot.
"""

from typing import Sequence

import numpy as np
import pybullet as p

from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.joint import JointPositions, \
    get_joint_infos, get_joints
from predicators.pybullet_helpers.link import get_link_pose
from predicators.settings import CFG


class InverseKinematicsError(ValueError):
    """Thrown when inverse kinematics fails to find a solution."""


def pybullet_inverse_kinematics(
    robot: int,
    end_effector: int,
    target_position: Pose3D,
    target_orientation: Quaternion,
    joints: Sequence[int],
    physics_client_id: int,
    validate: bool = True,
) -> JointPositions:
    """Runs IK and returns joint positions for the given (free) joints.

    If validate is True, the PyBullet IK solver is called multiple
    times, resetting the robot state each time, until the target
    position is reached. If the target position is not reached after a
    maximum number of iters, an exception is raised.
    """
    # Figure out which joint each dimension of the return of IK corresponds to.
    all_joints = get_joints(robot, physics_client_id=physics_client_id)
    joint_infos = get_joint_infos(robot,
                                  all_joints,
                                  physics_client_id=physics_client_id)
    free_joints = [
        joint_info.jointIndex for joint_info in joint_infos
        if joint_info.qIndex > -1
    ]
    assert set(joints).issubset(set(free_joints))

    # Record the initial state of the joints (positions and velocities) so
    # that we can reset them after.
    initial_joints_states = p.getJointStates(robot,
                                             free_joints,
                                             physicsClientId=physics_client_id)
    if validate:
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

        # Note: we are checking end-effector positions only for convergence.
        ee_link_pose = get_link_pose(robot, end_effector, physics_client_id)
        if np.allclose(ee_link_pose.position,
                       target_position,
                       atol=convergence_tol):
            break
    else:
        raise InverseKinematicsError("Inverse kinematics failed to converge.")

    # Reset the joint state (positions and velocities) to their initial values
    # to avoid modifying the PyBullet internal state.
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
