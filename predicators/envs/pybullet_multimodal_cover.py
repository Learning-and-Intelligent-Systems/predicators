"""A PyBullet version of Blocks."""

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.multimodal_cover import MultiModalCoverEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Array, EnvironmentTask, Object, State
import pickle as pkl


def get_asset_path(filename: str) -> str:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', filename)
    assert os.path.exists(path), f"Asset {path} does not exist"
    return path


def get_tagged_block_sizes() -> List[Tuple[float, float, float]]:
    tags_path = get_asset_path('tags')
    return [
        dimensions
        for block_info_fname in os.listdir(tags_path)
        for d, w, h in [sorted(pkl.load(open(os.path.join(tags_path, block_info_fname), 'rb'))['dimensions'])]
        for dimensions in [(d, w, h), (w, d, h)]
    ]


def get_zone_data() -> any:
    f_name = get_asset_path('zone_data.pkl')
    with (f_name, 'rb') as f:
        return pkl.load(f)


class PyBulletMultiModalCoverEnv(PyBulletEnv, MultiModalCoverEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    if CFG.multi_modal_cover_real_robot:
        block_poses: ClassVar[Pose] = pkl.load(open(get_asset_path("block-poses.pkl")))

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # We track the correspondence between PyBullet object IDs and Object
        # instances for blocks. This correspondence changes with the task.
        self._block_id_to_block: Dict[int, Object] = {}
        self._zone_id_to_zone: Dict[int, Object] = {}

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle blocks-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)
        logging.info(f'CLIENT ID: {physics_client_id}')
        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if True:  # pragma: no cover

            # Draw the workspace on the table for clarity.
            p.addUserDebugLine([cls.x_lb, cls.y_lb, cls.table_height],
                               [cls.x_ub, cls.y_lb, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_ub, cls.table_height],
                               [cls.x_ub, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_lb, cls.y_lb, cls.table_height],
                               [cls.x_lb, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            p.addUserDebugLine([cls.x_ub, cls.y_lb, cls.table_height],
                               [cls.x_ub, cls.y_ub, cls.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=physics_client_id)
            # Draw coordinate frame labels for reference.
            p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)
            # Draw the pick z location at the x/y midpoint.
            mid_x = (cls.x_ub + cls.x_lb) / 2
            mid_y = (cls.y_ub + cls.y_lb) / 2
            p.addUserDebugText("*", [mid_x, mid_y, cls.pick_z],
                               [1.0, 0.0, 0.0],
                               physicsClientId=physics_client_id)

        # Create blocks. Note that we create the maximum number once, and then
        # later on, in reset_state(), we will remove blocks from the workspace
        # (teleporting them far away) based on which ones are in the state.
        num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test)) if not CFG.multi_modal_cover_real_robot else \
            len(get_tagged_block_sizes())

        logging.info(num_blocks)
        block_ids = []

        if CFG.multi_modal_cover_real_robot:
            block_sizes = get_tagged_block_sizes()
        else:
            block_sizes = [[CFG.blocks_block_size, CFG.blocks_block_size, CFG.blocks_block_size] for _ in
                           range(num_blocks)]

        for i in range(num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            half_extents = (block_sizes[i][0] / 2.0, block_sizes[i][1] / 2.0,
                            block_sizes[i][2] / 2.0)

            block_ids.append(
                create_pybullet_block(color, half_extents, cls._obj_mass,
                                      cls._obj_friction, cls._default_orn,
                                      physics_client_id))
        logging.info(block_ids)

        zone_ids = []

        if CFG.multi_modal_cover_real_robot:
            cls.zone_extents = get_zone_data()

        for i, zone in enumerate(cls.zone_extents):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            zone_width = zone[1][0] - zone[0][0]
            zone_height = zone[1][1] - zone[0][1]

            half_extents = (zone_width / 2.0, zone_height / 2.0,
                            cls._target_height / 2.0)
            zone_ids.append(
                create_pybullet_block(color, half_extents, cls._obj_mass,
                                      cls._obj_friction, cls._default_orn,
                                      physics_client_id))

        bodies["block_ids"] = block_ids
        bodies["zone_ids"] = zone_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._block_ids = pybullet_bodies["block_ids"]
        self._zone_ids = pybullet_bodies["zone_ids"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       robot_ee_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        # The orientation is fixed in this environment.
        qx, qy, qz, qw = self.get_robot_ee_home_orn()
        f = self.fingers_state_to_joint(self._pybullet_robot,
                                        state.get(self._robot, "fingers"))
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"), qx, qy, qz, qw, f
        ],
            dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_multimodal_cover"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle blocks-specific resetting."""
        super()._reset_state(state)
        logging.info("resetting pybullet state")
        logging.info(state)
        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        zone_objs = state.get_objects(self._zone_type)

        logging.info(block_objs)
        self._block_id_to_block = {}
        self._zone_id_to_zone = {}
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]

            self._block_id_to_block[block_id] = block_obj
            bx = state.get(block_obj, "pose_x")
            by = state.get(block_obj, "pose_y")
            bz = state.get(block_obj, "pose_z")
            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        for i, zone_obj in enumerate(zone_objs):
            zone_id = self._zone_ids[i]

            self._zone_id_to_zone[zone_id] = zone_obj

            lower_extent_x = state.get(zone_obj, "lower_extent_x")
            lower_extent_y = state.get(zone_obj, "lower_extent_y")
            upper_extent_x = state.get(zone_obj, "upper_extent_x")
            upper_extent_y = state.get(zone_obj, "upper_extent_y")

            p.resetBasePositionAndOrientation(
                zone_id, [(upper_extent_x + lower_extent_x) / 2.0, (upper_extent_y + lower_extent_y) / 2.0,
                          self.table_height + self._target_height / 2.0],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Check if we're holding some block.
        held_block = self._get_held_block(state)
        if held_block is not None:
            self._force_grasp_object(held_block)

        # For any blocks not involved, put them out of view.
        h = 0.5
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]

            assert block_id not in self._block_id_to_block
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * h],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_id_to_block and self._held_obj_id. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        state_dict = {}

        # Get robot state.
        rx, ry, rz, ox, oy, oz, ow, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(rf)
        state_dict[self._robot] = np.array([rx, ry, rz, ox, oy, oz, ow, fingers],
                                           dtype=np.float32)
        joint_positions = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_obj_id)
            # pose_x, pose_y, pose_z, held
            state_dict[block] = np.array([bx, by, bz, held],
                                         dtype=np.float32)

        zone_width = self.zone_extents[0][1][0] - self.zone_extents[0][0][0]
        zone_height = self.zone_extents[0][1][1] - self.zone_extents[0][0][1]

        for zone_id, zone in self._zone_id_to_zone.items():
            (x, y, z), _ = p.getBasePositionAndOrientation(
                zone_id, physicsClientId=self._physics_client_id)

            lower_extent_x = x - zone_width / 2.0
            lower_extent_y = y - zone_height / 2.0
            upper_extent_x = x + zone_width / 2.0
            upper_extent_y = y + zone_height / 2.0

            # pose_x, pose_y, pose_z, held
            state_dict[zone] = np.array([lower_extent_x, lower_extent_y, upper_extent_x, upper_extent_y],
                                        dtype=np.float32)

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)



        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num_tasks, possible_num_blocks, rng)
        return self._add_pybullet_state_to_tasks(tasks)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        if CFG.pybullet_robot == "panda":
            # gripper rotated 90deg so parallel to x-axis
            normal = np.array([1., 0., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }

    def _force_grasp_object(self, block: Object) -> None:
        block_to_block_id = {b: i for i, b in self._block_id_to_block.items()}
        block_id = block_to_block_id[block]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        assert block_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = block_id
        self._create_grasp_constraint()

    @classmethod
    def fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                               fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either pybullet_robot.closed_fingers or
        pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
        return closed_f if fingers_state == 0.0 else open_f

    def _fingers_joint_to_state(self, fingers_joint: float) -> float:
        """Convert the finger joint values in PyBullet to values for the State.

        The joint values given as input are the ones coming out of
        self._pybullet_robot.get_state().
        """
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        # Fingers in the State should be either 0 or 1.
        return int(fingers_joint > (open_f + closed_f) / 2)
