"""Tests for ``PyBulletEnv._reconstruction_diff`` angle-modulo handling.

Regression coverage for commit 222680da9 ("Compare angle features
modulo 2π in reconstruction diff"). Before the fix, a wrist of 4.68
(legal, but outside the canonical (-π, π] range that PyBullet reports
back from ``_get_state``) would diff against a reconstructed -1.60 and
trip the reconstruction warning even though the two represent the
same physical orientation.

These tests don't spin up PyBullet — they just exercise the
classmethod on hand-built ``State`` instances.
"""
# pylint: disable=protected-access,unused-import
from __future__ import annotations

import math

import numpy as np
import pytest

# Bootstrap circular imports.
import predicators.utils  # noqa: F401
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.structs import Object, State, Type


@pytest.fixture(name="robot_type")
def _robot_type():
    """Type with one angle feature and one position feature."""
    return Type("robot", ["wrist", "x"])


def _state(robot_type: Type, wrist: float, x: float) -> State:
    obj = Object("robot0", robot_type)
    return State({obj: np.array([wrist, x], dtype=np.float64)})


@pytest.fixture(name="ee_type")
def _ee_type():
    """Type carrying the full robot EE orientation triple plus a position.

    ``(roll, tilt, wrist)`` is a free SO(3) orientation, so the diff
    must compare it as a rotation rather than axis-by-axis.
    """
    return Type("robot", ["roll", "tilt", "wrist", "x"])


def _ee_state(ee_type: Type,
              roll: float,
              tilt: float,
              wrist: float,
              x: float = 0.5) -> State:
    obj = Object("robot0", ee_type)
    return State({obj: np.array([roll, tilt, wrist, x], dtype=np.float64)})


def test_reconstruction_diff_angle_wraps_modulo_2pi(robot_type):
    """Values that differ by an exact multiple of 2π represent the same
    physical orientation and must not appear in the diff."""
    requested = _state(robot_type, wrist=0.0, x=0.5)
    reconstructed = _state(robot_type, wrist=2 * math.pi, x=0.5)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert diff == "", diff
    # Also: a near-2π offset under atol should round-trip cleanly.
    requested = _state(robot_type, wrist=4.68, x=0.5)
    reconstructed = _state(robot_type, wrist=4.68 - 2 * math.pi, x=0.5)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert diff == "", diff


def test_reconstruction_diff_angle_pi_vs_negative_pi(robot_type):
    """+π and -π are the same orientation — shortest-arc delta is 0."""
    requested = _state(robot_type, wrist=math.pi, x=0.0)
    reconstructed = _state(robot_type, wrist=-math.pi, x=0.0)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert diff == ""


def test_reconstruction_diff_angle_real_mismatch_is_reported(robot_type):
    """π/2 vs -π/2 are opposite orientations — the shortest-arc delta is π,
    which exceeds atol and must surface in the diff."""
    requested = _state(robot_type, wrist=math.pi / 2, x=0.0)
    reconstructed = _state(robot_type, wrist=-math.pi / 2, x=0.0)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "robot0.wrist" in diff


def test_reconstruction_diff_non_angle_feature_uses_raw_delta(robot_type):
    """Non-angle features (``x`` here) compare with raw subtraction, no modulo
    wrap-around — a 1.0-unit delta is reported as 1.0."""
    requested = _state(robot_type, wrist=0.0, x=0.0)
    reconstructed = _state(robot_type, wrist=0.0, x=1.0)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "robot0.x" in diff
    assert "robot0.wrist" not in diff


def test_reconstruction_diff_object_set_mismatch(robot_type):
    """Objects present in only one state surface as a top-level diff line —
    unrelated to the angle-modulo logic but the same helper handles it."""
    o0 = Object("robot0", robot_type)
    o1 = Object("robot1", robot_type)
    requested = State({o0: np.array([0.0, 0.0])})
    reconstructed = State({o1: np.array([0.0, 0.0])})
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "only in requested" in diff
    assert "only in reconstructed" in diff


# ---------------------------------------------------------------------------
# Gimbal-lock orientation handling.
#
# Regression coverage for the boil run that crashed in _set_state with a
# ~2.42 rad per-axis roll/wrist "mismatch" while the EE pointed straight
# down (tilt=π/2). At gimbal lock the roll/wrist split is degenerate — only
# the rotation is meaningful — so the triple must be compared as a rotation.
# The two euler triples below encode the SAME physical orientation (geodesic
# angle ~0.004 rad), yet differ by ~2.42 rad on each of roll and wrist.
# ---------------------------------------------------------------------------

_GIMBAL_REQ = (2.419305, math.pi / 2, -0.709600)
_GIMBAL_REC = (0.0, math.pi / 2, -3.132968)


def test_reconstruction_diff_gimbal_lock_does_not_raise(ee_type):
    """The crash values must clear the raise threshold: same orientation, so
    the rotation angle is ~0 and the diff is empty at raise_atol."""
    requested = _ee_state(ee_type, *_GIMBAL_REQ)
    reconstructed = _ee_state(ee_type, *_GIMBAL_REC)
    diff = PyBulletEnv._reconstruction_diff(
        requested, reconstructed, atol=PyBulletEnv._reconstruction_raise_atol)
    assert diff == "", diff


def test_reconstruction_diff_gimbal_lock_reports_rotation_not_per_axis(
        ee_type):
    """Below atol the residual surfaces as one small <orientation> angle, never
    as the misleading ~2.42 rad per-axis roll/wrist rows."""
    requested = _ee_state(ee_type, *_GIMBAL_REQ)
    reconstructed = _ee_state(ee_type, *_GIMBAL_REC)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "robot0.<orientation>" in diff
    # The per-axis rows (format "robot0.roll: requested=...") must be gone —
    # they are what tripped the spurious raise.
    assert "robot0.roll:" not in diff
    assert "robot0.wrist:" not in diff
    # The reported rotation angle is the true tiny residual, not ~2.42.
    assert "Δangle=0.00" in diff


def test_reconstruction_diff_orientation_genuine_mismatch_reported(ee_type):
    """A real rotation difference (here 1.0 rad about Z at tilt=0, away from
    gimbal lock) is reported accurately as the rotation angle."""
    requested = _ee_state(ee_type, 0.0, 0.0, 0.0)
    reconstructed = _ee_state(ee_type, 0.0, 0.0, 1.0)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "robot0.<orientation>" in diff
    assert "Δangle=1.00" in diff


def test_reconstruction_diff_orientation_large_mismatch_would_raise(ee_type):
    """A genuinely corrupt orientation (2.5 rad) still exceeds raise_atol so
    the guard keeps catching real reconstruction failures."""
    requested = _ee_state(ee_type, 0.0, 0.0, 0.0)
    reconstructed = _ee_state(ee_type, 0.0, 0.0, 2.5)
    diff = PyBulletEnv._reconstruction_diff(
        requested, reconstructed, atol=PyBulletEnv._reconstruction_raise_atol)
    assert "robot0.<orientation>" in diff


def test_reconstruction_diff_orientation_position_still_per_feature(ee_type):
    """Non-orientation features on an EE-typed object (here ``x``) keep the
    plain per-feature path even though roll/tilt/wrist are grouped."""
    requested = _ee_state(ee_type, 0.0, 0.0, 0.0, x=0.0)
    reconstructed = _ee_state(ee_type, 0.0, 0.0, 0.0, x=1.0)
    diff = PyBulletEnv._reconstruction_diff(requested, reconstructed)
    assert "robot0.x" in diff
    assert "<orientation>" not in diff
