"""A PyBullet version of Cover."""

from typing import Any, ClassVar, Dict, List, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.cover import CoverEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, Object, \
    State, Type
from predicators.utils import BoundingBox, NSPredicate, RawState, VLMQuery


class PyBulletCoverEnv(PyBulletEnv, CoverEnv):
    """PyBullet Cover domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    # Object parameters.
    _obj_len_hgt: ClassVar[float] = 0.045
    _max_obj_width: ClassVar[float] = 0.07  # highest width normalized to this

    # Dimension and workspace parameters.
    _table_height: ClassVar[float] = 0.2
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    _offset: ClassVar[float] = 0.01
    pickplace_z: ClassVar[float] = _table_height + _obj_len_hgt * 0.5 + _offset
    _target_height: ClassVar[float] = 0.0001

    # Types
    bbox_features = ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"]
    _block_type = Type("block",
                       ["is_block", "is_target", "width", "pose", "grasp"]+
                       bbox_features)
    _target_type = Type("target", ["is_block", "is_target", "width", "pose"]+
                        bbox_features)
    _robot_type = Type("robot", ["hand", "pose_x", "pose_z"]+bbox_features)
    _table_type = Type("table", bbox_features)

    _obj_id_to_obj: Dict[int, Object] = {}

    # _Covers_NSP = NSPredicate("Covers", [_block_type, _target_type],
    #                             _Covers_NSP_holds)

    def _Covers_NSP_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        """Determine if the block is covering (directly on top of) the target
        region."""
        block, target = objects
        # Necessary but not sufficient condition for covering: no part of the
        # target region is outside the block.
        if state.get(target, "bbox_left") < state.get(block, "bbox_left") or\
            state.get(target, "bbox_right") > state.get(block, "bbox_right"):
            return False

        return _OnTable_NSP_holds(state, [block])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._block_id_to_block: Dict[int, Object] = {}
        self._target_id_to_target: Dict[int, Object] = {}

        # Create a copy of the pybullet robot for checking forward kinematics
        # in step() without changing the "real" robot state.
        fk_physics_id = p.connect(p.DIRECT)
        self._pybullet_robot_fk = self._create_pybullet_robot(fk_physics_id)

    def simulate(self, state: State, action: Action) -> State:
        # To implement this, need to handle resetting to states where the
        # block is held, and need to take into account the offset between
        # the hand and the held block, which reset_state() doesn't yet.
        raise NotImplementedError("Simulate not implemented for PyBulletCover")

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle cover-specific initialization."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        bodies["table_id"] = table_id
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)

        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))
        block_ids = []
        for i in range(CFG.cover_num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            width = CFG.cover_block_widths[i] / max_width * cls._max_obj_width
            half_extents = (cls._obj_len_hgt / 2.0, width / 2.0,
                            cls._obj_len_hgt / 2.0)
            block_ids.append(
                create_pybullet_block(color, half_extents, cls._obj_mass,
                                      cls._obj_friction, cls._default_orn,
                                      physics_client_id))
        bodies["block_ids"] = block_ids

        target_ids = []
        for i in range(CFG.cover_num_targets):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            color = (color[0], color[1], color[2], 0.5)  # slightly transparent
            width = CFG.cover_target_widths[i] / max_width * cls._max_obj_width
            half_extents = (cls._obj_len_hgt / 2.0, width / 2.0,
                            cls._target_height / 2.0)
            target_ids.append(
                create_pybullet_block(color, half_extents, cls._obj_mass,
                                      cls._obj_friction, cls._default_orn,
                                      physics_client_id))

        bodies["target_ids"] = target_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_id = pybullet_bodies["table_id"]
        self._block_ids = pybullet_bodies["block_ids"]
        self._target_ids = pybullet_bodies["target_ids"]

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        robot_ee_orn = cls.get_robot_ee_home_orn()
        ee_home = Pose((cls.workspace_x, cls.robot_init_y, cls.workspace_z),
                       robot_ee_orn)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        if self._HandEmpty_holds(state, []):
            fingers = self._pybullet_robot.open_fingers
        else:
            fingers = self._pybullet_robot.closed_fingers
        y_norm = state.get(self._robot, "hand")
        # De-normalize robot y to actual coordinates.
        ry = self.y_lb + (self.y_ub - self.y_lb) * y_norm
        rx = state.get(self._robot, "pose_x")
        rz = state.get(self._robot, "pose_z")
        # The orientation is fixed in this environment.
        qx, qy, qz, qw = self.get_robot_ee_home_orn()
        return np.array([rx, ry, rz, qx, qy, qz, qw, fingers],
                        dtype=np.float32)

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle cover-specific resetting."""
        super()._reset_state(state)
        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))

        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        self._obj_id_to_obj = {}
        self._obj_id_to_obj[self._pybullet_robot.robot_id] = self._robot
        self._obj_id_to_obj[self._table_id] = self._table
        self._block_id_to_block = {}
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            width_unnorm = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            assert width == state.get(block_obj, "width")
            self._block_id_to_block[block_id] = block_obj
            self._obj_id_to_obj[block_id] = block_obj
            bx = self.workspace_x
            # De-normalize block y to actual coordinates.
            y_norm = state.get(block_obj, "pose")
            by = self.y_lb + (self.y_ub - self.y_lb) * y_norm
            if state.get(block_obj, "grasp") != -1:
                # If an object starts out held, it has a different z.
                bz = self.workspace_z - self._offset
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

        # For any blocks not involved, put them out of view.
        h = self._obj_len_hgt
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            assert block_id not in self._block_id_to_block
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * h],
                self._default_orn,
                physicsClientId=self._physics_client_id)

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
            self._obj_id_to_obj[target_id] = target_obj
            tx = self.workspace_x
            # De-normalize target y to actual coordinates.
            y_norm = state.get(target_obj, "pose")
            ty = self.y_lb + (self.y_ub - self.y_lb) * y_norm
            tz = self._table_height + self._obj_len_hgt * 0.5
            p.resetBasePositionAndOrientation(
                target_id, [tx, ty, tz],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # Draw hand regions as debug lines.
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert self.using_gui, \
                "use_gui must be True to use pybullet_draw_debug."
            p.removeAllUserDebugItems(physicsClientId=self._physics_client_id)
            for hand_lb, hand_rb in self._get_hand_regions(state):
                # De-normalize hand bounds to actual coordinates.
                y_lb = self.y_lb + (self.y_ub - self.y_lb) * hand_lb
                y_rb = self.y_lb + (self.y_ub - self.y_lb) * hand_rb
                p.addUserDebugLine(
                    [self.workspace_x, y_lb, self._table_height + 1e-4],
                    [self.workspace_x, y_rb, self._table_height + 1e-4],
                    [0.0, 0.0, 1.0],
                    lineWidth=5.0,
                    physicsClientId=self._physics_client_id)

    def step(self, action: Action) -> State:
        # In the cover environment, we need to first check the hand region
        # constraint before we can call PyBullet.
        # Use self._pybullet_robot_fk to run forward kinematics, since that
        # method shouldn't be run on the client that is doing simulation.
        joint_positions = action.arr.tolist()
        _, ry, rz = self._pybullet_robot_fk.forward_kinematics(
            joint_positions).position
        hand = (ry - self.y_lb) / (self.y_ub - self.y_lb)
        hand_regions = self._get_hand_regions(self._current_state)
        # If we're going down to grasp, we need to be in a hand region.
        # Otherwise, we don't care if we're between hand regions.
        # To decide whether we should care about hand regions, we use a
        # value z_thresh that is the average between the resting z
        # and the z used for picking/placing a block.
        z_thresh = (self.pickplace_z + self.workspace_z) / 2
        if rz < z_thresh and not any(hand_lb <= hand <= hand_rb
                                     for hand_lb, hand_rb in hand_regions):
            # The constraint is violated, so noop.
            state_copy = self._current_state.copy()
            return state_copy
        return super().step(action)

    def _get_state(self) -> State:
        state_dict = {}
        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))

        # Get robot state.
        rx, ry, rz, _, _, _, _, _ = self._pybullet_robot.get_state()
        hand = (ry - self.y_lb) / (self.y_ub - self.y_lb)
        state_dict[self._robot] = np.array([hand, rx, rz], dtype=np.float32)
        joint_positions = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            width_unnorm = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            (_, by, _), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            pose = (by - self.y_lb) / (self.y_ub - self.y_lb)
            held = (block_id == self._held_obj_id)
            if held:
                grasp_unnorm = p.getConstraintInfo(
                    self._held_constraint_id, self._physics_client_id)[7][1]
                # Normalize grasp.
                grasp = grasp_unnorm / (self.y_ub - self.y_lb)
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
            pose = (ty - self.y_lb) / (self.y_ub - self.y_lb)
            state_dict[target] = np.array([0.0, 1.0, width, pose],
                                          dtype=np.float32)

        # Get table state.
        state_dict[self._table] = np.array([], dtype=np.float32)

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        # Both fetch and panda have grippers parallel to x-axis
        return {
            self._pybullet_robot.left_finger_id: np.array([1., 0., 0.]),
            self._pybullet_robot.right_finger_id: np.array([-1., 0., 0.]),
        }

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_cover"

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num, rng)
        return self._add_pybullet_state_to_tasks(tasks)
