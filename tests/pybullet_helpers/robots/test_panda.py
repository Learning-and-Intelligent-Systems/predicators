"""Tests for PandaPyBullet Robot."""
from unittest.mock import patch

import numpy as np
import pytest
from pybullet_utils.transformations import quaternion_from_euler

from predicators import utils
from predicators.pybullet_helpers.joint import get_joint_infos, get_joints
from predicators.pybullet_helpers.robots import PandaPyBulletRobot


@pytest.fixture(scope="function", name="panda")
def _panda_fixture(physics_client_id) -> PandaPyBulletRobot:
    """Get a PandaPyBulletRobot instance."""
    # Use reset control, so we can see effects of actions without stepping.
    utils.reset_config({"pybullet_control_mode": "reset"})

    panda = PandaPyBulletRobot((0.5, 0.0, 0.5),
                               quaternion_from_euler(np.pi, 0, np.pi / 2),
                               physics_client_id)
    assert panda.get_name() == "panda"
    assert panda.physics_client_id == physics_client_id
    # Panda must have IKFast
    assert panda.ikfast_info() is not None

    return panda


def test_panda_pybullet_robot_initial_configuration(panda):
    """Check initial configuration matches expected position."""
    # Check get_state
    state = panda.get_state()
    assert len(state) == 4
    xyz = state[:3]
    finger_pos = state[3]
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
    with pytest.raises(ValueError):
        panda.set_joints_with_ik(end_effector_pose=(999.0, 99.0, 999.0),
                                 validate=True)


def test_panda_pybullet_robot_inverse_kinematics_incorrect_solution(panda):
    """Test when IKFast returns an incorrect solution.

    Note that this doesn't happen in reality, but we need to check we
    validate correctly).
    """
    # Note: the ikfast_closest_inverse_kinematics import happens
    # in the single_arm.py module, not the panda.py module.
    with patch("predicators.pybullet_helpers.robots.single_arm."
               "ikfast_closest_inverse_kinematics") as ikfast_mock:
        # Patch return value of IKFast to be an incorrect solution
        ikfast_mock.return_value = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

        # If validate=False, error shouldn't be raised
        panda.set_joints_with_ik(end_effector_pose=(0.25, 0.25, 0.25),
                                 validate=False)

        # If validate=True, error should be raised as solution doesn't match
        # desired end effector pose
        with pytest.raises(ValueError):
            panda.set_joints_with_ik(end_effector_pose=(0.25, 0.25, 0.25),
                                     validate=True)


def test_panda_pybullet_robot_inverse_kinematics(panda):
    """Test IKFast normal functionality on PandaPyBulletRobot."""
    joint_positions = panda.set_joints_with_ik(end_effector_pose=(0.25, 0.25,
                                                                  0.25),
                                               validate=True)
    assert np.allclose(panda.forward_kinematics(joint_positions),
                       (0.25, 0.25, 0.25))
