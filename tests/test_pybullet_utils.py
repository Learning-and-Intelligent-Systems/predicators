"""Test cases for pybullet_utils."""

import pybullet as p
from predicators.src import utils

_PYBULLET_SCENE_IS_SETUP = False
_PYBULLET_SCENE_ATTRIBUTES = {}


def _setup_pybullet_test_scene():
    """Creates a PyBullet scene with a fetch robot.

    Initialized only once and cached globally for efficiency.
    """
    global _PYBULLET_SCENE_IS_SETUP
    if _PYBULLET_SCENE_IS_SETUP:
        return _PYBULLET_SCENE_ATTRIBUTES
    _PYBULLET_SCENE_IS_SETUP = True
    
    physics_client_id = p.connect(p.DIRECT)
    _PYBULLET_SCENE_ATTRIBUTES["physics_client_id"] = physics_client_id

    p.resetSimulation(physicsClientId=physics_client_id)
    
    fetch_id = p.loadURDF(
        utils.get_env_asset_path("urdf/robots/fetch.urdf"),
        useFixedBase=True,
        physicsClientId=self._physics_client_id)
    _PYBULLET_SCENE_ATTRIBUTES["fetch_id"] = fetch_id

    base_pose = [0.75, 0.7441, 0.0]
    base_orientation = [0., 0., 0., 1.]
    p.resetBasePositionAndOrientation(
        fetch_id,
        base_pose,
        base_orientation,
        physicsClientId=physics_client_id)

    import ipdb; ipdb.set_trace()
