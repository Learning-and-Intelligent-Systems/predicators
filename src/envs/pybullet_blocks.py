"""A PyBullet version of Blocks."""

import logging
from typing import Callable, ClassVar, Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.envs.pybullet_env import PyBulletEnv, _PyBulletState, \
    create_pybullet_block
from predicators.src.envs.pybullet_robots import _SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.src.settings import CFG
from predicators.src.structs import Array, Object, ParameterizedOption, \
    Pose3D, State


class PyBulletBlocksEnv(PyBulletEnv, BlocksEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Option parameters.
    _offset_z: ClassVar[float] = 0.01

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Sequence[float]] = [0., 0., 0., 1.]

    def __init__(self) -> None:
        super().__init__()

        # Override options, keeping the types and parameter spaces the same.
        open_fingers_func = lambda s, _1, _2: (s.get(self._robot, "fingers"),
                                               self.open_fingers)
        close_fingers_func = lambda s, _1, _2: (s.get(self._robot, "fingers"),
                                                self.closed_fingers)

        ## Pick option
        types = self._Pick.types
        params_space = self._Pick.params_space
        self._Pick: ParameterizedOption = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the block which we will grasp.
                self._create_move_to_above_block_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: self.pick_z,
                    finger_status="open"),
                # Open fingers.
                self._create_change_fingers_option(
                    "OpenFingers", types, params_space, open_fingers_func),
                # Move down to grasp.
                self._create_move_to_above_block_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda block_z: (block_z + self._offset_z),
                    finger_status="open"),
                # Close fingers.
                self._create_change_fingers_option(
                    "CloseFingers", types, params_space, close_fingers_func),
                # Move back up.
                self._create_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: self.pick_z,
                    finger_status="closed"),
            ])

        ## Stack option
        types = self._Stack.types
        params_space = self._Stack.params_space
        self._Stack: ParameterizedOption = \
            utils.LinearChainParameterizedOption("Stack",
            [
                # Move to above the block on which we will stack.
                self._create_move_to_above_block_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda _: self.pick_z,
                    finger_status="closed"),
                # Move down to place.
                self._create_move_to_above_block_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda block_z: (
                        block_z + self.block_size + self._offset_z),
                    finger_status="closed"),
                # Open fingers.
                self._create_change_fingers_option(
                    "OpenFingers", types, params_space, open_fingers_func),
                # Move back up.
                self._create_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: self.pick_z,
                    finger_status="open"),
            ])

        ## PutOnTable option
        types = self._PutOnTable.types
        params_space = self._PutOnTable.params_space
        place_z = self.table_height + self.block_size + self._offset_z
        self._PutOnTable: ParameterizedOption = \
            utils.LinearChainParameterizedOption("PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                self._create_move_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnTable",
                    z=self.pick_z,
                    finger_status="closed"),
                # Move down to place.
                self._create_move_to_above_table_option(
                    name="MoveEndEffectorToPutOnTable",
                    z=place_z,
                    finger_status="closed"),
                # Open fingers.
                self._create_change_fingers_option(
                    "OpenFingers", types, params_space, open_fingers_func),
                # Move back up.
                self._create_move_to_above_table_option(
                    name="MoveEndEffectorBackUp", z=self.pick_z,
                    finger_status="open"),
            ])

        # We track the correspondence between PyBullet object IDs and Object
        # instances for blocks. This correspondence changes with the task.
        self._block_id_to_block: Dict[int, Object] = {}

    def _initialize_pybullet(self) -> None:
        """Run super(), then handle blocks-specific initialization."""
        super()._initialize_pybullet()

        # Load table.
        self._table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id)

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert CFG.pybullet_use_gui, \
                "pybullet_use_gui must be True to use pybullet_draw_debug."
            # Draw the workspace on the table for clarity.
            p.addUserDebugLine([self.x_lb, self.y_lb, self.table_height],
                               [self.x_ub, self.y_lb, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0)
            p.addUserDebugLine([self.x_lb, self.y_ub, self.table_height],
                               [self.x_ub, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0)
            p.addUserDebugLine([self.x_lb, self.y_lb, self.table_height],
                               [self.x_lb, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0)
            p.addUserDebugLine([self.x_ub, self.y_lb, self.table_height],
                               [self.x_ub, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0)
            # Draw coordinate frame labels for reference.
            p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0])
            p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0])
            p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0])
            # Draw the pick z location at the x/y midpoint.
            mid_x = (self.x_ub + self.x_lb) / 2
            mid_y = (self.y_ub + self.y_lb) / 2
            p.addUserDebugText("*", [mid_x, mid_y, self.pick_z],
                               [1.0, 0.0, 0.0])

        # Create blocks. Note that we create the maximum number once, and then
        # later on, in reset_state(), we will remove blocks from the workspace
        # (teleporting them far away) based on which ones are in the state.
        num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
        self._block_ids = []
        for i in range(num_blocks):
            color = self._obj_colors[i % len(self._obj_colors)]
            orientation = [0.0, 0.0, 0.0, 1.0]  # default
            self._block_ids.append(
                create_pybullet_block(color, self.block_size, self._obj_mass,
                                      self._obj_friction, orientation,
                                      self._physics_client_id))

    def _create_pybullet_robot(self) -> _SingleArmPyBulletRobot:
        ee_home = (self.robot_init_x, self.robot_init_y, self.robot_init_z)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot, ee_home,
                                                self.open_fingers,
                                                self.closed_fingers,
                                                self._max_vel_norm,
                                                self._physics_client_id)

    def _extract_robot_state(self, state: State) -> Array:
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"),
            state.get(self._robot, "fingers")
        ],
                        dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_blocks"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle blocks-specific resetting."""
        super()._reset_state(state)

        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block = {}
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            self._block_id_to_block[block_id] = block_obj
            bx = state.get(block_obj, "pose_x")
            by = state.get(block_obj, "pose_y")
            bz = state.get(block_obj, "pose_z")
            # Assume not holding in the initial state
            assert self._get_held_block(state) is None
            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz], [0.0, 0.0, 0.0, 1.0],
                physicsClientId=self._physics_client_id)

        # For any blocks not involved, put them out of view.
        h = self.block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            assert block_id not in self._block_id_to_block
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * h], [0.0, 0.0, 0.0, 1.0],
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
        state_dict[self._robot] = self._pybullet_robot.get_state()
        joint_state = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_obj_id)
            # pose_x, pose_y, pose_z, held
            state_dict[block] = np.array([bx, by, bz, held], dtype=np.float32)

        state = _PyBulletState(state_dict, simulator_state=joint_state)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _create_move_to_above_block_option(
            self, name: str, z_func: Callable[[float], float],
            finger_status: str) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        block argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """
        types = [self._robot_type, self._block_type]
        params_space = Box(0, 1, (0, ))

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D, str]:
            assert not params
            robot, block = objects
            current_pose = (state.get(robot,
                                      "pose_x"), state.get(robot, "pose_y"),
                            state.get(robot, "pose_z"))
            target_pose = (state.get(block,
                                     "pose_x"), state.get(block, "pose_y"),
                           z_func(state.get(block, "pose_z")))
            return current_pose, target_pose, finger_status

        return self._create_move_end_effector_to_pose_option(
            name, types, params_space,
            _get_current_and_target_pose_and_finger_status)

    def _create_move_to_above_table_option(
            self, name: str, z: float,
            finger_status: str) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        table.

        The z position of the target pose must be provided.
        """
        types = [self._robot_type]
        params_space = Box(0, 1, (2, ))

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D, str]:
            robot, = objects
            current_pose = (state.get(robot,
                                      "pose_x"), state.get(robot, "pose_y"),
                            state.get(robot, "pose_z"))
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            target_pose = (self.x_lb + (self.x_ub - self.x_lb) * x_norm,
                           self.y_lb + (self.y_ub - self.y_lb) * y_norm, z)
            return current_pose, target_pose, finger_status

        return self._create_move_end_effector_to_pose_option(
            name, types, params_space,
            _get_current_and_target_pose_and_finger_status)
