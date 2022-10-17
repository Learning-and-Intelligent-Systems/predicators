"""A PyBullet version of Blocks."""

import logging
from typing import Callable, ClassVar, Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box
from pybullet_utils.transformations import quaternion_from_euler

from predicators import utils
from predicators.envs.blocks import BlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, State


class PyBulletBlocksEnv(PyBulletEnv, BlocksEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Option parameters.
    _offset_z: ClassVar[float] = 0.01

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    # Robot parameters.
    _ee_orn: ClassVar[Dict[str, Quaternion]] = {
        # Fetch gripper down since it's thin we don't need to rotate 90 degrees.
        "fetch": quaternion_from_euler(0.0, np.pi / 2, -np.pi),
        # Panda gripper down and rotated 90 degrees as it's big and can cause
        # collisions.
        "panda": quaternion_from_euler(np.pi, 0, np.pi / 2)
    }
    _move_to_pose_tol: ClassVar[float] = 1e-4

    def __init__(self) -> None:
        super().__init__()

        # Override options, keeping the types and parameter spaces the same.
        open_fingers_func = lambda s, _1, _2: (self._fingers_state_to_joint(
            s.get(self._robot, "fingers")), self._pybullet_robot.open_fingers)
        close_fingers_func = lambda s, _1, _2: (self._fingers_state_to_joint(
            s.get(self._robot, "fingers")), self._pybullet_robot.closed_fingers
                                                )

        ## Pick option
        types = self._Pick.types
        params_space = self._Pick.params_space
        self._Pick: ParameterizedOption = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Move to far above the block which we will grasp.
                self._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreGrasp",
                    z_func=lambda _: self.pick_z,
                    finger_status="open"),
                # Open fingers.
                create_change_fingers_option(
                    self._pybullet_robot, "OpenFingers", types, params_space,
                    open_fingers_func, self._max_vel_norm, self._grasp_tol),
                # Move down to grasp.
                self._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToGrasp",
                    z_func=lambda block_z: (block_z + self._offset_z),
                    finger_status="open"),
                # Close fingers.
                create_change_fingers_option(
                    self._pybullet_robot, "CloseFingers", types, params_space,
                    close_fingers_func, self._max_vel_norm, self._grasp_tol),
                # Move back up.
                self._create_blocks_move_to_above_block_option(
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
                self._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToPreStack",
                    z_func=lambda _: self.pick_z,
                    finger_status="closed"),
                # Move down to place.
                self._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorToStack",
                    z_func=lambda block_z: (
                        block_z + self.block_size + self._offset_z),
                    finger_status="closed"),
                # Open fingers.
                create_change_fingers_option(self._pybullet_robot,
                    "OpenFingers", types, params_space, open_fingers_func,
                    self._max_vel_norm, self._grasp_tol),
                # Move back up.
                self._create_blocks_move_to_above_block_option(
                    name="MoveEndEffectorBackUp",
                    z_func=lambda _: self.pick_z,
                    finger_status="open"),
            ])

        ## PutOnTable option
        types = self._PutOnTable.types
        params_space = self._PutOnTable.params_space
        place_z = self.table_height + self.block_size / 2 + self._offset_z
        self._PutOnTable: ParameterizedOption = \
            utils.LinearChainParameterizedOption("PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                self._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPrePutOnTable",
                    z=self.pick_z,
                    finger_status="closed"),
                # Move down to place.
                self._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorToPutOnTable",
                    z=place_z,
                    finger_status="closed"),
                # Open fingers.
                create_change_fingers_option(self._pybullet_robot,
                    "OpenFingers", types, params_space, open_fingers_func,
                    self._max_vel_norm, self._grasp_tol),
                # Move back up.
                self._create_blocks_move_to_above_table_option(
                    name="MoveEndEffectorBackUp", z=self.pick_z,
                    finger_status="open"),
            ])

        # We track the correspondence between PyBullet object IDs and Object
        # instances for blocks. This correspondence changes with the task.
        self._block_id_to_block: Dict[int, Object] = {}

    def _initialize_pybullet(self) -> None:
        """Run super(), then handle blocks-specific initialization."""
        super()._initialize_pybullet()

        # Load table in both the main client and the copy.
        self._table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id)
        p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                   useFixedBase=True,
                   physicsClientId=self._physics_client_id2)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id2)

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert CFG.pybullet_use_gui, \
                "pybullet_use_gui must be True to use pybullet_draw_debug."
            # Draw the workspace on the table for clarity.
            p.addUserDebugLine([self.x_lb, self.y_lb, self.table_height],
                               [self.x_ub, self.y_lb, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=self._physics_client_id)
            p.addUserDebugLine([self.x_lb, self.y_ub, self.table_height],
                               [self.x_ub, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=self._physics_client_id)
            p.addUserDebugLine([self.x_lb, self.y_lb, self.table_height],
                               [self.x_lb, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=self._physics_client_id)
            p.addUserDebugLine([self.x_ub, self.y_lb, self.table_height],
                               [self.x_ub, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0,
                               physicsClientId=self._physics_client_id)
            # Draw coordinate frame labels for reference.
            p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                               physicsClientId=self._physics_client_id)
            p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                               physicsClientId=self._physics_client_id)
            p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0],
                               physicsClientId=self._physics_client_id)
            # Draw the pick z location at the x/y midpoint.
            mid_x = (self.x_ub + self.x_lb) / 2
            mid_y = (self.y_ub + self.y_lb) / 2
            p.addUserDebugText("*", [mid_x, mid_y, self.pick_z],
                               [1.0, 0.0, 0.0],
                               physicsClientId=self._physics_client_id)

        # Create blocks. Note that we create the maximum number once, and then
        # later on, in reset_state(), we will remove blocks from the workspace
        # (teleporting them far away) based on which ones are in the state.
        num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
        self._block_ids = []
        for i in range(num_blocks):
            color = self._obj_colors[i % len(self._obj_colors)]
            half_extents = (self.block_size / 2.0, self.block_size / 2.0,
                            self.block_size / 2.0)
            self._block_ids.append(
                create_pybullet_block(color, half_extents, self._obj_mass,
                                      self._obj_friction, self._default_orn,
                                      self._physics_client_id))

    def _create_pybullet_robot(
            self, physics_client_id: int) -> SingleArmPyBulletRobot:
        ee_home = (self.robot_init_x, self.robot_init_y, self.robot_init_z)
        ee_orn = self._ee_orn[CFG.pybullet_robot]
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home,
                                                ee_orn)

    def _extract_robot_state(self, state: State) -> Array:
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"),
            self._fingers_state_to_joint(state.get(self._robot, "fingers")),
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
                block_id, [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id)
            # Update the block color. RGB values are between 0 and 1.
            r = state.get(block_obj, "color_r")
            g = state.get(block_obj, "color_g")
            b = state.get(block_obj, "color_b")
            color = (r, g, b, 1.0)  # alpha = 1.0
            p.changeVisualShape(block_id,
                                linkIndex=-1,
                                rgbaColor=color,
                                physicsClientId=self._physics_client_id)

        # For any blocks not involved, put them out of view.
        h = self.block_size
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
        rx, ry, rz, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(rf)
        state_dict[self._robot] = np.array([rx, ry, rz, fingers],
                                           dtype=np.float32)
        joint_positions = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_obj_id)
            visual_data = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0]
            r, g, b, _ = visual_data[7]
            # pose_x, pose_y, pose_z, held
            state_dict[block] = np.array([bx, by, bz, held, r, g, b],
                                         dtype=np.float32)

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

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

    def _create_blocks_move_to_above_block_option(
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

        return create_move_end_effector_to_pose_option(
            self._pybullet_robot, name, types, params_space,
            _get_current_and_target_pose_and_finger_status,
            self._move_to_pose_tol, self._max_vel_norm,
            self._finger_action_nudge_magnitude)

    def _create_blocks_move_to_above_table_option(
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

        return create_move_end_effector_to_pose_option(
            self._pybullet_robot, name, types, params_space,
            _get_current_and_target_pose_and_finger_status,
            self._move_to_pose_tol, self._max_vel_norm,
            self._finger_action_nudge_magnitude)

    def _fingers_state_to_joint(self, fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either self._pybullet_robot.closed_fingers or
        self._pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
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
