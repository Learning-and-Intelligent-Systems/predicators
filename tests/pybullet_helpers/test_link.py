"""Tests for link PyBullet helper utilities."""
from unittest.mock import call, patch

import pytest

import predicators.pybullet_helpers.link
from predicators.pybullet_helpers.geometry import Pose, multiply_poses
from predicators.pybullet_helpers.link import LinkState, get_relative_link_pose

_MODULE_PATH = predicators.pybullet_helpers.link.__name__


def test_link_state():
    """Tests for LinkState()."""
    link_state = LinkState(
        linkWorldPosition=(0.0, 1.0, 0.0),
        linkWorldOrientation=(0.0, 1.0, 0.0, 1.0),
        localInertialFramePosition=(0.0, 0.0, 0.0),
        localInertialFrameOrientation=(0.0, 0.0, 0.0, 1.0),
        worldLinkFramePosition=(0.0, 0.0, 0.0),
        worldLinkFrameOrientation=(0.0, 0.0, 0.0, 1.0),
    )
    com_pose = link_state.com_pose
    assert com_pose.position == (0.0, 1.0, 0.0)
    assert com_pose.orientation == (0.0, 1.0, 0.0, 1.0)
    pose = link_state.pose
    assert pose.position == (0.0, 0.0, 0.0)
    assert pose.orientation == (0.0, 0.0, 0.0, 1.0)


@pytest.mark.parametrize("body, link1, link2, physics_client_id",
                         [(0, 1, 2, 0), (1, 2, 8, 2)])
def test_get_relative_link_pose(body, link1, link2, physics_client_id):
    """Tests for get_relative_link_pose()."""
    world_from_link1 = Pose(position=(0.5, 0.5, 0.5),
                            orientation=(0.0, 0.0, 0.0, 1.0))
    world_from_link2 = Pose(position=(1.0, 1.0, 1.0),
                            orientation=(0.0, 1.0, 0.0, 1.0))
    link2_from_link1 = multiply_poses(world_from_link2.invert(),
                                      world_from_link1)

    with patch(f"{_MODULE_PATH}.get_link_pose") as mock_get_link_pose:
        mock_get_link_pose.side_effect = [world_from_link1, world_from_link2]
        relative_link_pose = get_relative_link_pose(body, link1, link2,
                                                    physics_client_id)
        assert relative_link_pose == link2_from_link1

        assert mock_get_link_pose.call_count == 2
        mock_get_link_pose.assert_has_calls([
            call(body, link1, physics_client_id),
            call(body, link2, physics_client_id)
        ])
