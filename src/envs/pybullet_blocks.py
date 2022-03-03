"""A PyBullet version of Blocks."""

from typing import Sequence
import pybullet as p
from predicators.src.envs.blocks import BlocksEnv
from predicators.src import utils
from predicators.src.settings import CFG


class PyBulletBlocksEnv(BlocksEnv):
    """PyBullet Blocks domain."""

    # Fetch robot parameters.
    _base_position: Sequence[float] = [0.75, 0.7441, 0.0]
    _base_orientation: Sequence[float] = [0., 0., 0., 1.]

    # Table parameters.
    _table_position: Sequence[float] = [1.65, 0.75, 0.0]
    _table_orientation: Sequence[float] = [0., 0., 0., 1.]

    # Camera parameters.
    _camera_distance: float = 1.5
    _yaw: float = 90.0
    _pitch: float = -24
    _camera_target: Sequence[float] = [1.65, 0.75, 0.42]

    def __init__(self) -> None:
        super().__init__()

        # One-time initialization of pybullet assets. Note that this happens
        # in __init__ because many class attributes are created.
        if CFG.pybullet_use_gui:
            self._physics_client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                self._camera_distance,
                self._yaw,
                self._pitch,
                self._camera_target,
                physicsClientId=self._physics_client_id)
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._physics_client_id)

        # Load plane.
        p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"), [0, 0, -1],
                   useFixedBase=True,
                   physicsClientId=self._physics_client_id)

        # Load Fetch robot.
        self._fetch_id = p.loadURDF(
            utils.get_env_asset_path("urdf/robots/fetch.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_position,
            self._base_orientation,
            physicsClientId=self._physics_client_id)

        # Load table.
        self._table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_position,
            self._table_orientation,
            physicsClientId=self._physics_client_id)

        import ipdb
        ipdb.set_trace()
