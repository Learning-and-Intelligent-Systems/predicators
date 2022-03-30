"""Test cases for pybullet_utils."""

import numpy as np
import pybullet as p
import pytest

from predicators.src import utils
from predicators.src.envs.utils import get_kinematic_chain, inverse_kinematics
from predicators.src.settings import CFG


@pytest.fixture(scope="module", name="scene_attributes")
def _setup_pybullet_test_scene():
    """Creates a PyBullet scene with a fetch robot.

    Initialized only once for efficiency.
    """
    scene = {}

    physics_client_id = p.connect(p.DIRECT)
    scene["physics_client_id"] = physics_client_id

    p.resetSimulation(physicsClientId=physics_client_id)

    fetch_id = p.loadURDF(utils.get_env_asset_path("urdf/robots/fetch.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    scene["fetch_id"] = fetch_id

    base_pose = [0.75, 0.7441, 0.0]
    base_orientation = [0., 0., 0., 1.]
    p.resetBasePositionAndOrientation(fetch_id,
                                      base_pose,
                                      base_orientation,
                                      physicsClientId=physics_client_id)

    joint_names = [
        p.getJointInfo(fetch_id, i,
                       physicsClientId=physics_client_id)[1].decode("utf-8")
        for i in range(
            p.getNumJoints(fetch_id, physicsClientId=physics_client_id))
    ]
    ee_id = joint_names.index('gripper_axis')
    scene["ee_id"] = ee_id
    scene["ee_orientation"] = [1., 0., -1., 0.]

    scene["robot_home"] = [1.35, 0.75, 0.75]

    arm_joints = get_kinematic_chain(fetch_id,
                                     ee_id,
                                     physics_client_id=physics_client_id)
    scene["initial_joint_states"] = p.getJointStates(
        fetch_id, arm_joints, physicsClientId=physics_client_id)

    return scene


def test_get_kinematic_chain(scene_attributes):
    """Tests for get_kinematic_chain()."""
    arm_joints = get_kinematic_chain(
        scene_attributes["fetch_id"],
        scene_attributes["ee_id"],
        physics_client_id=scene_attributes["physics_client_id"])
    # Fetch arm has 7 DOF.
    assert len(arm_joints) == 7


def test_inverse_kinematics(scene_attributes):
    """Tests for inverse_kinematics()."""
    arm_joints = get_kinematic_chain(
        scene_attributes["fetch_id"],
        scene_attributes["ee_id"],
        physics_client_id=scene_attributes["physics_client_id"])

    # Reset the joint states to their initial values.
    def _reset_joints():
        for joint, joint_state in zip(
                arm_joints, scene_attributes["initial_joint_states"]):
            position, velocity, _, _ = joint_state
            p.resetJointState(
                scene_attributes["fetch_id"],
                joint,
                targetValue=position,
                targetVelocity=velocity,
                physicsClientId=scene_attributes["physics_client_id"])

    target_position = scene_attributes["robot_home"]
    # With validate = False, one call to IK is not good enough.
    _reset_joints()
    joint_values = inverse_kinematics(
        scene_attributes["fetch_id"],
        scene_attributes["ee_id"],
        target_position,
        scene_attributes["ee_orientation"],
        arm_joints,
        physics_client_id=scene_attributes["physics_client_id"],
        validate=False)
    for joint, joint_value in zip(arm_joints, joint_values):
        p.resetJointState(
            scene_attributes["fetch_id"],
            joint,
            targetValue=joint_value,
            physicsClientId=scene_attributes["physics_client_id"])
    ee_link_state = p.getLinkState(
        scene_attributes["fetch_id"],
        scene_attributes["ee_id"],
        computeForwardKinematics=True,
        physicsClientId=scene_attributes["physics_client_id"])
    assert not np.allclose(
        ee_link_state[4], target_position, atol=CFG.pybullet_ik_tol)
    # With validate = True, IK does work.
    _reset_joints()
    joint_values = inverse_kinematics(
        scene_attributes["fetch_id"],
        scene_attributes["ee_id"],
        target_position,
        scene_attributes["ee_orientation"],
        arm_joints,
        physics_client_id=scene_attributes["physics_client_id"],
        validate=True)
    for joint, joint_value in zip(arm_joints, joint_values):
        p.resetJointState(
            scene_attributes["fetch_id"],
            joint,
            targetValue=joint_value,
            physicsClientId=scene_attributes["physics_client_id"])
    ee_link_state = p.getLinkState(
        scene_attributes["fetch_id"],
        scene_attributes["ee_id"],
        computeForwardKinematics=True,
        physicsClientId=scene_attributes["physics_client_id"])
    assert np.allclose(ee_link_state[4],
                       target_position,
                       atol=CFG.pybullet_ik_tol)
    # With validate = True, if the position is impossible to reach, an error
    # is raised.
    target_position = [
        target_position[0], target_position[1], target_position[2] + 100.0
    ]
    with pytest.raises(Exception) as e:
        inverse_kinematics(
            scene_attributes["fetch_id"],
            scene_attributes["ee_id"],
            target_position,
            scene_attributes["ee_orientation"],
            arm_joints,
            physics_client_id=scene_attributes["physics_client_id"],
            validate=True)
    assert "Inverse kinematics failed to converge." in str(e)
