"""Tests for PandaPyBullet Robot."""
from unittest.mock import patch

import numpy as np
import pybullet as p
import pytest
from pybullet_utils.transformations import quaternion_from_euler

from predicators import utils
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.joint import get_joint_infos, get_joints
from predicators.pybullet_helpers.robots import PandaPyBulletRobot
from predicators.pybullet_helpers.robots.panda import PANDA_HOME_ARM_JOINTS, \
    PANDA_HOME_EE_POSE_IN_BASE


@pytest.fixture(scope="function", name="panda")
def _panda_fixture(physics_client_id) -> PandaPyBulletRobot:
    """Get a PandaPyBulletRobot instance."""
    # Use reset control, so we can see effects of actions without stepping.
    utils.reset_config({"pybullet_control_mode": "reset"})

    home_pose = Pose((0.5, 0.0, 0.5),
                     quaternion_from_euler(np.pi, 0, np.pi / 2))
    panda = PandaPyBulletRobot(physics_client_id, home_pose)
    assert panda.get_name() == "panda"
    assert panda.physics_client_id == physics_client_id
    # Panda must have IKFast
    assert panda.ikfast_info() is not None

    return panda


def test_panda_pybullet_robot_initial_configuration(panda):
    """Check initial configuration matches expected position."""
    # Check get_state
    state = panda.get_state()
    assert len(state) == 8
    xyz = state[:3]
    finger_pos = state[-1]
    assert np.allclose(xyz, (0.5, 0.0, 0.5), atol=1e-3)
    assert np.isclose(finger_pos, panda.open_fingers)


def test_panda_pybullet_robot_links(panda):
    """Test link utilities on PandaPyBulletRobot."""
    # Panda 7 DOF and the left and right fingers are appended last.
    assert panda.left_finger_joint_idx == 7
    assert panda.right_finger_joint_idx == 8

    # Tool link is last link in Panda URDF
    num_links = len(panda.joint_infos)
    assert panda.tool_link_id == num_links - 1
    assert panda.tool_link_name == "tool_link"

    # Check base link
    assert panda.base_link_name == "panda_link0"

    with pytest.raises(ValueError):
        # Non-existent link
        panda.link_from_name("non_existent_link")


def test_panda_pybullet_robot_joints(panda):
    """Test joint utilities on PandaPyBulletRobot."""
    # Check joint limits match action space
    assert np.allclose(panda.action_space.low, panda.joint_lower_limits)
    assert np.allclose(panda.action_space.high, panda.joint_upper_limits)

    # Check joint infos match expected
    panda_joints = get_joints(panda.robot_id, panda.physics_client_id)
    assert panda.joint_infos == get_joint_infos(panda.robot_id, panda_joints,
                                                panda.physics_client_id)

    # Check getting joints
    assert panda.joint_info_from_name(
        "panda_joint5").jointName == "panda_joint5"
    assert (panda.joint_from_name("panda_joint5") ==
            panda.joint_info_from_name("panda_joint5").jointIndex)

    # Check Panda joints - 7 joints for arm + 2 fingers
    assert panda.arm_joints == [0, 1, 2, 3, 4, 5, 6, 9, 10]

    with pytest.raises(ValueError):
        panda.joint_from_name("non_existent_joint")
    with pytest.raises(ValueError):
        panda.joint_info_from_name("non_existent_joint")


def test_panda_pybullet_robot_inverse_kinematics_no_solutions(panda):
    """Test when IKFast returns no solutions."""
    # Impossible target pose with no solutions
    pose = Pose((999.0, 99.0, 999.0), (0.7071, 0.7071, 0.0, 0.0))
    with pytest.raises(ValueError):
        panda.inverse_kinematics(end_effector_pose=pose, validate=True)


def test_panda_pybullet_robot_inverse_kinematics_incorrect_solution(panda):
    """Test when IKFast returns an incorrect solution.

    Note that this doesn't happen in reality, but we need to check we
    validate correctly).
    """
    pose = Pose((0.25, 0.25, 0.25), (0.7071, 0.7071, 0.0, 0.0))
    # Note: the ikfast_closest_inverse_kinematics import happens
    # in the single_arm.py module, not the panda.py module.
    with patch("predicators.pybullet_helpers.robots.single_arm."
               "ikfast_closest_inverse_kinematics") as ikfast_mock:
        # Patch return value of IKFast to be an incorrect solution
        ikfast_mock.return_value = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

        # If validate=False, error shouldn't be raised
        panda.inverse_kinematics(end_effector_pose=pose, validate=False)

        # If validate=True, error should be raised as solution doesn't match
        # desired end effector pose
        with pytest.raises(ValueError):
            panda.inverse_kinematics(end_effector_pose=pose, validate=True)


def test_panda_pybullet_robot_inverse_kinematics(panda):
    """Test IKFast normal functionality on PandaPyBulletRobot."""
    pose = Pose((0.25, 0.25, 0.25), (0.7071, 0.7071, 0.0, 0.0))
    joint_positions = panda.inverse_kinematics(end_effector_pose=pose,
                                               validate=True)
    recovered_pose = panda.forward_kinematics(joint_positions)
    assert np.allclose(recovered_pose.position, pose.position)


def test_panda_home_ee_pose_matches_forward_kinematics(panda):
    """PANDA_HOME_EE_POSE_IN_BASE must stay in sync with the home joints.

    It is precomputed so that callers can locate the home pose without a
    URDF, so nothing else would catch it drifting.
    """
    home_joints = list(PANDA_HOME_ARM_JOINTS) + [
        panda.open_fingers, panda.open_fingers
    ]
    home_pose = panda.forward_kinematics(home_joints)
    assert np.allclose(home_pose.position,
                       PANDA_HOME_EE_POSE_IN_BASE.position,
                       atol=1e-3)
    assert np.allclose(np.abs(home_pose.orientation),
                       np.abs(PANDA_HOME_EE_POSE_IN_BASE.orientation),
                       atol=1e-3)


def test_panda_homes_to_canonical_configuration(physics_client_id):
    """With no home pose requested, the Panda homes to the Franka's canonical
    configuration."""
    utils.reset_config({"pybullet_control_mode": "reset"})
    panda = PandaPyBulletRobot(physics_client_id=physics_client_id)
    assert np.allclose(panda.initial_joint_positions[:7],
                       PANDA_HOME_ARM_JOINTS,
                       atol=1e-3)
    assert np.allclose(panda.get_state()[:3],
                       PANDA_HOME_EE_POSE_IN_BASE.position,
                       atol=1e-3)


def test_panda_home_keeps_canonical_arm_under_rolled_orientation(
        physics_client_id):
    """The home configuration keeps its canonical arm shape when the home
    orientation is rolled about the gripper axis, as every env's is: the free
    wrist joint absorbs the roll.

    Plain IK would instead swing the shoulder, since a wrist roll costs
    more than a shoulder swing under its closest-solution metric.
    """
    utils.reset_config({"pybullet_control_mode": "reset"})
    # The canonical home pose, rolled 90 degrees about the (downward) gripper
    # axis -- i.e. the top-down grasp orientation the envs use.
    rolled_home_pose = Pose(PANDA_HOME_EE_POSE_IN_BASE.position,
                            (0.7071, 0.7071, 0.0, 0.0))
    panda = PandaPyBulletRobot(physics_client_id, rolled_home_pose)
    home_joints = panda.initial_joint_positions
    # The arm joints are canonical...
    assert np.allclose(home_joints[:6], PANDA_HOME_ARM_JOINTS[:6], atol=0.2)
    # ...and the wrist took the roll.
    assert np.isclose(home_joints[6],
                      PANDA_HOME_ARM_JOINTS[6] + np.pi / 2,
                      atol=0.2)
    assert np.allclose(panda.get_state()[:3],
                       rolled_home_pose.position,
                       atol=1e-3)


def test_panda_pushes_with_its_front_face(panda):
    """The Franka Hand pushes front-on, unlike the base class's default."""
    assert panda.push_ee_yaw_offset == pytest.approx(np.pi / 2)


def test_panda_finger_dynamics(panda):
    """The gripper must be able to stall on a grasped object instead of
    crushing through it: finite finger motor force, real finger inertials, and.

    joint damping stable under that force cap (damping < 2 * mass / dt).
    """
    # The URDF effort limit, applied as the motor force cap in set_motors.
    assert panda.finger_motor_force == pytest.approx(20.0)
    for finger_id in (panda.left_finger_id, panda.right_finger_id):
        mass = p.getDynamicsInfo(panda.robot_id, finger_id,
                                 panda.physics_client_id)[0]
        damping = p.getJointInfo(panda.robot_id, finger_id,
                                 panda.physics_client_id)[6]
        # Real Franka finger mass, not PyBullet's mass=1 default for links
        # that lack an <inertial> tag.
        assert mass == pytest.approx(0.015)
        # PyBullet's joint damping is only stable when the motor can pin
        # the joint velocity; with a finite force cap that requires
        # damping < 2 * mass / dt (dt = 1/240).
        assert 0 < damping < 2 * mass * 240


def test_panda_set_motors_position_mode(physics_client_id):
    """set_motors in position control mode issues the capped finger motor
    command without error and holds the commanded finger position."""
    utils.reset_config({"pybullet_control_mode": "position"})
    panda = PandaPyBulletRobot(physics_client_id=physics_client_id)
    target = list(panda.initial_joint_positions)
    target[panda.left_finger_joint_idx] = 0.02
    target[panda.right_finger_joint_idx] = 0.02
    for _ in range(50):
        panda.set_motors(target)
        p.stepSimulation(physicsClientId=physics_client_id)
    joints = panda.get_joints()
    assert np.isclose(joints[panda.left_finger_joint_idx], 0.02, atol=1e-3)
    assert np.isclose(joints[panda.right_finger_joint_idx], 0.02, atol=1e-3)
