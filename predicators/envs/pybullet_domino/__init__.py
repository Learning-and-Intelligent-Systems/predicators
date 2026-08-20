"""PyBullet Domino Environment Package.

This package provides a modular, component-based domino
environment for PyBullet.

Example usage:

    from predicators.envs.pybullet_domino import (
        PyBulletDominoEnv, PyBulletDominoFanEnv)

    env = PyBulletDominoEnv(use_gui=True)
    # or
    env = PyBulletDominoFanEnv(use_gui=True)
"""

from predicators.envs.pybullet_domino.env import PyBulletDominoEnv, \
    PyBulletDominoFanEnv

__all__ = [
    "PyBulletDominoEnv",
    "PyBulletDominoFanEnv",
]
