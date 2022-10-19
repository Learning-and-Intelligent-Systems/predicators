"""Tests for behavior_utils."""

import numpy as np
import pytest

from predicators.behavior_utils import behavior_utils as utils


def test_aabb_volume():
    """Tests for get_aabb_volume()."""
    lo = np.array([1.0, 1.5, -1.0])
    hi = np.array([2.0, 2.5, 0.0])
    # Test zero volume calculation
    assert utils.get_aabb_volume(lo, lo) == 0.0
    # Test ordinary calculation
    assert utils.get_aabb_volume(lo, hi) == 1.0
    with pytest.raises(AssertionError):
        # Test assertion error when lower bound is
        # greater than upper bound
        lo1 = np.array([10.0, 12.5, 10.0])
        hi1 = np.array([-10.0, -12.5, -10.0])
        assert utils.get_aabb_volume(lo1, hi1)


def test_aabb_closest_point():
    """Tests for get_closest_point_on_aabb()."""
    # Test ordinary usage
    xyz = [1.5, 3.0, -2.5]
    lo = np.array([1.0, 1.5, -1.0])
    hi = np.array([2.0, 2.5, 0.0])
    assert utils.get_closest_point_on_aabb(xyz, lo, hi) == [1.5, 2.5, -1.0]
    with pytest.raises(AssertionError):
        # Test error where lower bound is greater than upper bound.
        lo1 = np.array([10.0, 12.5, 10.0])
        hi1 = np.array([-10.0, -12.5, -10.0])
        utils.get_closest_point_on_aabb(xyz, lo1, hi1)
