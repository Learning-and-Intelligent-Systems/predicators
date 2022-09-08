"""Tests for link PyBullet helper utilities."""

from predicators.pybullet_helpers.link import LinkState


def test_link_state():
    """Tests for LinkState()."""
    link_state = LinkState(linkWorldPosition=(0.0, 1.0, 0.0),
                           linkWorldOrientation=(0.0, 1.0, 0.0, 1.0),
                           localInertialFramePosition=(0.0, 0.0, 0.0),
                           localInertialFrameOrientation=(0.0, 0.0, 0.0, 1.0),
                           worldLinkFramePosition=(0.0, 0.0, 0.0),
                           worldLinkFrameOrientation=(0.0, 0.0, 0.0, 1.0))
    com_pose = link_state.com_pose
    assert com_pose.position == (0.0, 1.0, 0.0)
    assert com_pose.orientation == (0.0, 1.0, 0.0, 1.0)
    pose = link_state.pose
    assert pose.position == (0.0, 0.0, 0.0)
    assert pose.orientation == (0.0, 0.0, 0.0, 1.0)
