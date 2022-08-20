"""Tests for geometry PyBullet helper utilities."""

import numpy as np
import pybullet as p

from predicators.src.pybullet_helpers.geometry import Pose, get_pose, \
    matrix_from_quat


def test_pose():
    """Tests for Pose()."""
    position = (5.0, 0.5, 1.0)
    orientation = (0.0, 0.0, 0.0, 1.0)
    pose = Pose(position, orientation)
    rpy = pose.rpy
    reconstructed_pose = Pose.from_rpy(position, rpy)
    assert pose.allclose(reconstructed_pose)
    unit_pose = Pose.identity()
    assert not pose.allclose(unit_pose)
    multiplied_pose = pose.multiply(unit_pose, unit_pose, unit_pose)
    assert pose.allclose(multiplied_pose)
    inverted_pose = pose.invert()
    assert not pose.allclose(inverted_pose)
    assert pose.allclose(inverted_pose.invert())


def test_matrix_from_quat():
    """Tests for matrix_from_quat()."""
    mat = matrix_from_quat((0.0, 0.0, 0.0, 1.0))
    assert np.allclose(mat, np.eye(3))
    mat = matrix_from_quat((0.0, 0.0, 0.0, -1.0))
    assert np.allclose(mat, np.eye(3))
    mat = matrix_from_quat((1.0, 0.0, 0.0, 1.0))
    expected_mat = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ])
    assert np.allclose(mat, expected_mat)


def test_get_pose(physics_client_id):
    """Tests for get_pose()."""
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=[1, 1, 1],
                                          physicsClientId=physics_client_id)
    mass = 0
    position = (1.0, 0.0, 3.0)
    orientation = (0.0, 1.0, 0.0, 0.0)
    expected_pose = Pose(position, orientation)
    body = p.createMultiBody(mass,
                             collision_id,
                             basePosition=position,
                             baseOrientation=orientation,
                             physicsClientId=physics_client_id)
    pose = get_pose(body, physics_client_id)
    assert pose.allclose(expected_pose)
