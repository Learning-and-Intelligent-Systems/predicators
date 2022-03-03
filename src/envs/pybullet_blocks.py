"""A PyBullet version of Blocks."""

from typing import Sequence, Dict
from gym.spaces import Box
import pybullet as p
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.structs import Object
from predicators.src import utils
from predicators.src.pybullet_utils import get_kinematic_chain, \
    inverse_kinematics
from predicators.src.settings import CFG


class PyBulletBlocksEnv(BlocksEnv):
    """PyBullet Blocks domain."""

    # Fetch robot parameters.
    _base_position: Sequence[float] = [0.75, 0.7441, 0.0]
    _base_orientation: Sequence[float] = [0., 0., 0., 1.]
    _ee_orientation: Sequence[float] = [1., 0., -1., 0.]
    _ee_initial_position: Sequence[float] = [1., 0.7, 0.5]

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

        # Extract IDs for individual robot links and joints.
        joint_names = [
            p.getJointInfo(
                self._fetch_id, i,
                physicsClientId=self._physics_client_id)[1].decode("utf-8")
            for i in range(
                p.getNumJoints(self._fetch_id,
                               physicsClientId=self._physics_client_id))
        ]
        self._ee_id = joint_names.index('gripper_axis')
        self._arm_joints = get_kinematic_chain(
            self._fetch_id,
            self._ee_id,
            physics_client_id=self._physics_client_id)
        self._left_finger_id = joint_names.index("l_gripper_finger_joint")
        self._right_finger_id = joint_names.index("r_gripper_finger_joint")
        self._arm_joints.append(self._left_finger_id)
        self._arm_joints.append(self._right_finger_id)

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

        # Set gravity.
        p.setGravity(0., 0., -10., physicsClientId=self._physics_client_id)

        # Determine good initial joint values.
        self._initial_joint_values = inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            self._ee_initial_position,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id)

        # Blocks are created at reset.
        self._block_ids: Dict[Object, int] = {}

        # When a block is held, a constraint is created to prevent slippage.
        self._held_constraint_id: Optional[int] = None

    @property
    def action_space(self) -> Box:
        # dimensions: [dx, dy, dz, fingers]
        return Box(low=-1, high=1, shape=(4,), dtype=np.float32)
