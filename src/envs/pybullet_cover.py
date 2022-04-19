"""A PyBullet version of Cover."""

from typing import ClassVar, Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs.cover import CoverEnv
from predicators.src.envs.pybullet_env import PyBulletEnv, \
    create_pybullet_block
from predicators.src.envs.pybullet_robots import _SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, Object, \
    ParameterizedOption, Pose3D, State, Type


class PyBulletCoverEnv(PyBulletEnv, CoverEnv):
    """PyBullet Cover domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Option parameters.
    _open_fingers: ClassVar[float] = 0.04
    _closed_fingers: ClassVar[float] = 0.01

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Sequence[float]] = [0., 0., 0., 1.]

    # Robot parameters.
    _ee_orn: ClassVar[Sequence[float]] = p.getQuaternionFromEuler(
        [np.pi / 2, np.pi / 2, -np.pi])
    _move_to_pose_tol: ClassVar[float] = 1e-7

    # Object parameters.
    _obj_len_hgt: ClassVar[float] = 0.045
    _max_obj_width: ClassVar[float] = 0.07  # highest width normalized to this

    # Dimension and workspace parameters.
    _table_height: ClassVar[float] = 0.2
    _y_lb: ClassVar[float] = 0.4
    _y_ub: ClassVar[float] = 1.1
    _robot_init_y: ClassVar[float] = (_y_lb + _y_ub) / 2
    _offset: ClassVar[float] = 0.01
    _pickplace_z: ClassVar[
        float] = _table_height + _obj_len_hgt * 0.5 + _offset
    _target_height: ClassVar[float] = 0.0001

    def __init__(self) -> None:
        super().__init__()

        # Override PickPlace option
        types = self._PickPlace.types
        params_space = self._PickPlace.params_space
        # Note: this isn't exactly correct because the first argument should be
        # the current finger joint value, which we don't have in the State `s`.
        # This could lead to slippage or bad grasps, but we haven't seen this
        # in practice, so we'll leave it as is instead of changing the State.
        toggle_fingers_func = lambda s, _1, _2: (
            (self._open_fingers, self._closed_fingers)
            if self._HandEmpty_holds(s, []) else
            (self._closed_fingers, self._open_fingers))
        self._PickPlace: ParameterizedOption = \
            utils.LinearChainParameterizedOption(
                "PickPlace",
                [
                    # Move to far above the location we will pick/place at.
                    self._create_cover_move_option(
                        name="MoveEndEffectorToPrePose",
                        target_z=self._workspace_z),
                    # Move down to pick/place.
                    self._create_cover_move_option(
                        name="MoveEndEffectorToPose",
                        target_z=self._pickplace_z),
                    # Toggle fingers.
                    self._pybullet_robot.create_change_fingers_option(
                        "ToggleFingers", types, params_space,
                        toggle_fingers_func),
                    # Move back up.
                    self._create_cover_move_option(
                        name="MoveEndEffectorBackUp",
                        target_z=self._workspace_z)
                ])
        self._block_id_to_block: Dict[int, Object] = {}
        self._target_id_to_target: Dict[int, Object] = {}

    def _initialize_pybullet(self) -> None:
        """Run super(), then handle cover-specific initialization."""
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
        self._table_id2 = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id2)
        p.resetBasePositionAndOrientation(
            self._table_id2,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id2)

        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))
        self._block_ids = []
        for i in range(CFG.cover_num_blocks):
            color = self._obj_colors[i % len(self._obj_colors)]
            width = CFG.cover_block_widths[i] / max_width * self._max_obj_width
            half_extents = (self._obj_len_hgt / 2.0, width / 2.0,
                            self._obj_len_hgt / 2.0)
            self._block_ids.append(
                create_pybullet_block(color, half_extents, self._obj_mass,
                                      self._obj_friction, self._default_orn,
                                      self._physics_client_id))
        self._target_ids = []
        for i in range(CFG.cover_num_targets):
            color = self._obj_colors[i % len(self._obj_colors)]
            color = (color[0], color[1], color[2], 0.5)  # slightly transparent
            width = CFG.cover_target_widths[i] / max_width * self._max_obj_width
            half_extents = (self._obj_len_hgt / 2.0, width / 2.0,
                            self._target_height / 2.0)
            self._target_ids.append(
                create_pybullet_block(color, half_extents, self._obj_mass,
                                      self._obj_friction, self._default_orn,
                                      self._physics_client_id))

    def _create_pybullet_robot(
            self, physics_client_id: int) -> _SingleArmPyBulletRobot:
        ee_home = (self._workspace_x, self._robot_init_y, self._workspace_z)
        return create_single_arm_pybullet_robot(
            CFG.pybullet_robot, ee_home, self._ee_orn, self._open_fingers,
            self._closed_fingers, self._move_to_pose_tol, self._max_vel_norm,
            self._grasp_tol, physics_client_id)

    def _extract_robot_state(self, state: State) -> Array:
        if self._HandEmpty_holds(state, []):
            fingers = self._open_fingers
        else:
            fingers = self._closed_fingers
        y_norm = state.get(self._robot, "hand")
        # De-normalize robot y to actual coordinates.
        ry = self._y_lb + (self._y_ub - self._y_lb) * y_norm
        rx = state.get(self._robot, "pose_x")
        rz = state.get(self._robot, "pose_z")
        return np.array([rx, ry, rz, fingers], dtype=np.float32)

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle cover-specific resetting."""
        super()._reset_state(state)
        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))

        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block = {}
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            width_unnorm = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            assert width == state.get(block_obj, "width")
            self._block_id_to_block[block_id] = block_obj
            bx = self._workspace_x
            # De-normalize block y to actual coordinates.
            y_norm = state.get(block_obj, "pose")
            by = self._y_lb + (self._y_ub - self._y_lb) * y_norm
            if state.get(block_obj, "grasp") != -1:
                # If an object starts out held, it has a different z.
                bz = self._workspace_z - self._offset
            else:
                bz = self._table_height + self._obj_len_hgt * 0.5
            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id)
            if state.get(block_obj, "grasp") != -1:
                # If an object starts out held, set up the grasp constraint.
                self._held_obj_id = self._detect_held_object()
                assert self._held_obj_id == block_id
                self._create_grasp_constraint()

        # Reset targets based on the state.
        target_objs = state.get_objects(self._target_type)
        self._target_id_to_target = {}
        for i, target_obj in enumerate(target_objs):
            target_id = self._target_ids[i]
            width_unnorm = p.getVisualShapeData(
                target_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            assert width == state.get(target_obj, "width")
            self._target_id_to_target[target_id] = target_obj
            tx = self._workspace_x
            # De-normalize target y to actual coordinates.
            y_norm = state.get(target_obj, "pose")
            ty = self._y_lb + (self._y_ub - self._y_lb) * y_norm
            tz = self._table_height + self._obj_len_hgt * 0.5
            p.resetBasePositionAndOrientation(
                target_id, [tx, ty, tz],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Draw hand regions as debug lines.
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert CFG.pybullet_use_gui, \
                "pybullet_use_gui must be True to use pybullet_draw_debug."
            p.removeAllUserDebugItems(physicsClientId=self._physics_client_id)
            for hand_lb, hand_rb in self._get_hand_regions(state):
                # De-normalize hand bounds to actual coordinates.
                y_lb = self._y_lb + (self._y_ub - self._y_lb) * hand_lb
                y_rb = self._y_lb + (self._y_ub - self._y_lb) * hand_rb
                p.addUserDebugLine(
                    [self._workspace_x, y_lb, self._table_height + 1e-4],
                    [self._workspace_x, y_rb, self._table_height + 1e-4],
                    [0.0, 0.0, 1.0],
                    lineWidth=5.0,
                    physicsClientId=self._physics_client_id)

    def step(self, action: Action) -> State:
        # In the cover environment, we need to first check the hand region
        # constraint before we can call PyBullet.
        # Use self._pybullet_robot2 to run forward kinematics, since that
        # method shouldn't be run on the client that is doing simulation.
        _, ry, rz = self._pybullet_robot2.forward_kinematics(action.arr)
        hand = (ry - self._y_lb) / (self._y_ub - self._y_lb)
        hand_regions = self._get_hand_regions(self._current_state)
        # If we're going down to grasp, we need to be in a hand region.
        # Otherwise, we don't care if we're between hand regions.
        # To decide whether we should care about hand regions, we use a
        # value z_thresh that is the average between the resting z
        # and the z used for picking/placing a block.
        z_thresh = (self._pickplace_z + self._workspace_z) / 2
        if rz < z_thresh and not any(hand_lb <= hand <= hand_rb
                                     for hand_lb, hand_rb in hand_regions):
            # The constraint is violated, so no-op.
            return self._current_state.copy()
        return super().step(action)

    def _get_state(self) -> State:
        state_dict = {}
        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))

        # Get robot state.
        rx, ry, rz, _ = self._pybullet_robot.get_state()
        hand = (ry - self._y_lb) / (self._y_ub - self._y_lb)
        state_dict[self._robot] = np.array([hand, rx, rz], dtype=np.float32)
        joint_state = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            width_unnorm = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            (_, by, _), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            pose = (by - self._y_lb) / (self._y_ub - self._y_lb)
            held = (block_id == self._held_obj_id)
            if held:
                grasp_unnorm = p.getConstraintInfo(
                    self._held_constraint_id, self._physics_client_id)[7][1]
                # Normalize grasp.
                grasp = grasp_unnorm / (self._y_ub - self._y_lb)
            else:
                grasp = -1
            state_dict[block] = np.array([1.0, 0.0, width, pose, grasp],
                                         dtype=np.float32)

        # Get target states.
        for target_id, target in self._target_id_to_target.items():
            width_unnorm = p.getVisualShapeData(
                target_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            (_, ty, _), _ = p.getBasePositionAndOrientation(
                target_id, physicsClientId=self._physics_client_id)
            pose = (ty - self._y_lb) / (self._y_ub - self._y_lb)
            state_dict[target] = np.array([0.0, 1.0, width, pose],
                                          dtype=np.float32)

        state = utils.PyBulletState(state_dict, simulator_state=joint_state)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        return {
            self._pybullet_robot.left_finger_id: np.array([1., 0., 0.]),
            self._pybullet_robot.right_finger_id: np.array([-1., 0., 0.]),
        }

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_cover"

    def _create_cover_move_option(self, name: str,
                                  target_z: float) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose in Cover."""
        types: Sequence[Type] = []
        params_space = Box(0, 1, (1, ))

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D, str]:
            assert not objects
            hand = state.get(self._robot, "hand")
            # De-normalize hand feature to actual table coordinates.
            current_y = self._y_lb + (self._y_ub - self._y_lb) * hand
            current_pose = (state.get(self._robot, "pose_x"), current_y,
                            state.get(self._robot, "pose_z"))
            y_norm, = params
            # De-normalize parameter to actual table coordinates.
            target_y = self._y_lb + (self._y_ub - self._y_lb) * y_norm
            target_pose = (self._workspace_x, target_y, target_z)
            if self._HandEmpty_holds(state, []):
                finger_status = "open"
            else:
                finger_status = "closed"
            return current_pose, target_pose, finger_status

        return self._pybullet_robot.create_move_end_effector_to_pose_option(
            name, types, params_space,
            _get_current_and_target_pose_and_finger_status)
