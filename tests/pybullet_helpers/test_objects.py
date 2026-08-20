"""Unit tests for predicators.pybullet_helpers.objects."""
import numpy as np
import pytest

from predicators.pybullet_helpers.objects import \
    sample_collision_free_2d_positions
from predicators.utils import Circle, Rectangle


def test_sample_collision_free_2d_positions_circles_no_overlap():
    """Sampled circles never overlap with each other."""
    rng = np.random.default_rng(0)
    radius = 0.05
    positions = sample_collision_free_2d_positions(
        num_samples=8,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        shape_type="circle",
        shape_params=[radius],
        rng=rng,
    )
    assert len(positions) == 8
    circles = [Circle(x, y, radius) for x, y in positions]
    for i, c1 in enumerate(circles):
        for c2 in circles[i + 1:]:
            assert not c1.intersects(c2)


def test_sample_collision_free_2d_positions_within_bounds():
    """Sampled positions stay inside the requested x/y range."""
    rng = np.random.default_rng(0)
    positions = sample_collision_free_2d_positions(
        num_samples=5,
        x_range=(-0.5, 0.5),
        y_range=(2.0, 3.0),
        shape_type="circle",
        shape_params=[0.05],
        rng=rng,
    )
    for x, y in positions:
        assert -0.5 <= x <= 0.5
        assert 2.0 <= y <= 3.0


def test_sample_collision_free_2d_positions_rectangles_no_overlap():
    """Sampled rectangles never overlap with each other."""
    rng = np.random.default_rng(1)
    w, h, theta = 0.05, 0.05, 0.0
    positions = sample_collision_free_2d_positions(
        num_samples=4,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        shape_type="rectangle",
        shape_params=[w, h, theta],
        rng=rng,
    )
    assert len(positions) == 4
    rects = [Rectangle(x, y, w, h, theta) for x, y in positions]
    for i, r1 in enumerate(rects):
        for r2 in rects[i + 1:]:
            assert not r1.intersects(r2)


def test_sample_collision_free_2d_positions_reproducible():
    """Same seed produces the same positions."""
    pos_a = sample_collision_free_2d_positions(
        num_samples=4,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        shape_type="circle",
        shape_params=[0.05],
        rng=np.random.default_rng(123),
    )
    pos_b = sample_collision_free_2d_positions(
        num_samples=4,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        shape_type="circle",
        shape_params=[0.05],
        rng=np.random.default_rng(123),
    )
    assert pos_a == pos_b


def test_sample_collision_free_2d_positions_impossible_raises():
    """Asking for more shapes than fit raises RuntimeError."""
    # 4 disks of radius 0.5 cannot fit non-overlapping in [0,1]^2.
    rng = np.random.default_rng(0)
    with pytest.raises(RuntimeError, match="Max tries exceeded"):
        sample_collision_free_2d_positions(
            num_samples=4,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            shape_type="circle",
            shape_params=[0.5],
            rng=rng,
            max_tries_total=200,
        )


def test_sample_collision_free_2d_positions_invalid_shape_raises():
    """An unknown shape_type raises ValueError."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Unsupported shape_type"):
        sample_collision_free_2d_positions(
            num_samples=1,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            shape_type="triangle",
            shape_params=[0.05],
            rng=rng,
        )
