"""Tests for the home-configuration sync in PyBulletEnv.

The dummy env classes below stay abstract (they implement none of the
abstract methods), so they are never instantiated and never registered
as real envs; only the classmethods under test are exercised.
"""
# pylint: disable=protected-access

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.structs import Type


def test_sync_robot_init_pos_with_home():
    """robot_init_{x,y,z} follow the robot's home position, and restore to the
    env-declared values for robots without a home configuration."""

    class _ParentEnv(PyBulletEnv):
        robot_init_x = 1.0
        robot_init_y = 2.0
        robot_init_z = 3.0

    home = (0.9, 0.8, 0.7)
    _ParentEnv._sync_robot_init_pos_with_home(home)
    assert (_ParentEnv.robot_init_x, _ParentEnv.robot_init_y,
            _ParentEnv.robot_init_z) == home
    # A robot without a home configuration restores the declared values.
    _ParentEnv._sync_robot_init_pos_with_home(None)
    assert (_ParentEnv.robot_init_x, _ParentEnv.robot_init_y,
            _ParentEnv.robot_init_z) == (1.0, 2.0, 3.0)
    # Syncing again after a restore still works.
    _ParentEnv._sync_robot_init_pos_with_home(home)
    assert (_ParentEnv.robot_init_x, _ParentEnv.robot_init_y,
            _ParentEnv.robot_init_z) == home


def test_sync_robot_init_pos_with_home_inherited():
    """A subclass that inherits robot_init_{x,y,z} from an already-synced
    parent must restore the parent's DECLARED values, not the synced values it
    happens to inherit."""

    class _ParentEnv(PyBulletEnv):
        robot_init_x = 1.0
        robot_init_y = 2.0
        robot_init_z = 3.0

    class _ChildEnv(_ParentEnv):
        pass

    # The parent syncs to a Panda-style home first...
    _ParentEnv._sync_robot_init_pos_with_home((0.9, 0.8, 0.7))
    # ...then the child runs with a Fetch-style robot (no home). Its first
    # sync must see through the parent's synced values.
    _ChildEnv._sync_robot_init_pos_with_home(None)
    assert (_ChildEnv.robot_init_x, _ChildEnv.robot_init_y,
            _ChildEnv.robot_init_z) == (1.0, 2.0, 3.0)


def test_get_robot_ee_init_orn():
    """Robots with a home configuration home to the orientation the initial
    state encodes; robots without one keep the env default."""
    default_orn = p.getQuaternionFromEuler([0.3, 0.2, 0.1])

    class _OrnEnv(PyBulletEnv):
        _robot_type = Type("robot", ["x", "y", "z", "tilt", "wrist"])
        robot_init_tilt = np.pi
        robot_init_wrist = 0.0

        @classmethod
        def get_robot_ee_home_orn(cls):
            """A fixed default, bypassing the CFG lookup."""
            return default_orn

    # Without a home configuration, the env default is kept.
    assert _OrnEnv.get_robot_ee_init_orn(False) == default_orn
    # With one, tilt/wrist come from the initial state's features; roll is
    # not a robot feature here, so it falls back to the default's roll.
    orn = _OrnEnv.get_robot_ee_init_orn(True)
    assert np.allclose(orn, p.getQuaternionFromEuler([0.3, np.pi, 0.0]))

    class _NoTypeEnv(PyBulletEnv):

        @classmethod
        def get_robot_ee_home_orn(cls):
            """A fixed default, bypassing the CFG lookup."""
            return default_orn

    # An env with no class-level _robot_type keeps the default too.
    assert _NoTypeEnv.get_robot_ee_init_orn(True) == default_orn
