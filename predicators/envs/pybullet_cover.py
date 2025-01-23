"""A PyBullet version of Cover.

python predicators/main.py --approach oracle --env pybullet_cover --seed 0 \
--num_train_tasks 0 --num_test_tasks 1 --use_gui --debug  \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --make_test_videos \
# --sesame_check_expected_atoms False
"""
from typing import Any, ClassVar, Dict, List

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.cover import CoverEnv
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Object, State, Array


class PyBulletCoverEnv(PyBulletEnv, CoverEnv):
    """PyBullet Cover domain, refactored to utilize the updated PyBulletEnv.
    x: robot -> table
    y: table left -> right

    """

    # ------------------------
    # Class-level constants
    # ------------------------
    # Table parameters
    _table_height: ClassVar[float] = 0.4
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, _table_height / 2)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.62)
    
    # Object parameters
    _obj_len_hgt: ClassVar[float] = 0.045
    _max_obj_width: ClassVar[float] = 0.07  # highest width normalized to this

    # Dimension and workspace parameters
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    robot_init_x: ClassVar[float] = CoverEnv.workspace_x
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = CoverEnv.workspace_z
    _offset: ClassVar[float] = 0.01
    pickplace_z: ClassVar[float] = _table_height + _obj_len_hgt * 0.5 + _offset
    _target_height: ClassVar[float] = 0.0001

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Store block/target IDs (from initialize_pybullet) so that we can
        # reset their positions in _reset_custom_env_state().
        self._table_id: int = -1
        # self._block_ids: list[int] = []
        # self._target_ids: list[int] = []

        # Optional "forward-kinematics" client for advanced logic in step()
        fk_physics_id = p.connect(p.DIRECT)
        self._pybullet_robot_fk = self._create_pybullet_robot(fk_physics_id)

    # -----------------------------------------------------------------------
    # Required Hooks
    # -----------------------------------------------------------------------
    @classmethod
    def get_name(cls) -> str:
        return "pybullet_cover"

    @classmethod
    def initialize_pybullet(
        cls, using_gui: bool
    ) -> tuple[int, SingleArmPyBulletRobot, dict[str, any]]:
        """Create the world: plane, table, block IDs, etc."""
        # Call parent method first
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui
        )

        # Load table
        table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=physics_client_id
        )
        bodies["table_id"] = table_id
        p.resetBasePositionAndOrientation(
            table_id,
            cls._table_pose,
            cls._table_orientation,
            physicsClientId=physics_client_id
        )

        # Create blocks
        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))
        block_ids = []
        for i in range(CFG.cover_num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            width = CFG.cover_block_widths[i] / max_width * cls._max_obj_width
            half_extents = (
                cls._obj_len_hgt / 2.0,
                width / 2.0,
                cls._obj_len_hgt / 2.0
            )
            block_id = create_pybullet_block(
                color=color,
                half_extents=half_extents,
                mass=cls._obj_mass,
                friction=cls._obj_friction,
                physics_client_id=physics_client_id
            )
            block_ids.append(block_id)
        bodies["block_ids"] = block_ids

        # Create targets
        target_ids = []
        for i in range(CFG.cover_num_targets):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            color = (color[0], color[1], color[2], 0.5)  # semi-transparent
            width = (CFG.cover_target_widths[i]
                     / max_width * cls._max_obj_width)
            half_extents = (
                cls._obj_len_hgt / 2.0,
                width / 2.0,
                cls._target_height / 2.0
            )
            target_id = create_pybullet_block(
                color=color,
                half_extents=half_extents,
                mass=cls._obj_mass,
                friction=cls._obj_friction,
                physics_client_id=physics_client_id
            )
            target_ids.append(target_id)
        bodies["target_ids"] = target_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: dict[str, any]) -> None:
        """Store references to PyBullet IDs for environment assets."""
        self._table_id = pybullet_bodies["table_id"]
        for blk, id in zip(self._blocks, pybullet_bodies["block_ids"]):
            blk.id = id
        # self._block_ids = pybullet_bodies["block_ids"]
        # self._target_ids = pybullet_bodies["target_ids"]
        for tgt, id in zip(self._targets, pybullet_bodies["target_ids"]):
            tgt.id = id

    def _create_task_specific_objects(self, state: State) -> None:
        """No domain-specific extra creation needed here."""
        pass

    def _reset_custom_env_state(self, state: State) -> None:
        """After the parent class has reset the robot, handle the block/target
        positions. Because our block objects do not have standard 'x','y','z'
        features, we do the custom placement here.
        """
        max_width = max(max(CFG.cover_block_widths),
                        max(CFG.cover_target_widths))

        # 1) Reset blocks
        block_objs = state.get_objects(self._block_type)
        used_block_ids = []
        for i, block_obj in enumerate(block_objs):
            block_id = block_obj.id # self._block_ids[i]
            used_block_ids.append(block_id)

            # Double-check shape correctness
            width_unnorm = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id
            )[0][3][1]
            # Re-derive the original (cover) width
            width = width_unnorm / self._max_obj_width * max_width
            assert np.isclose(width, state.get(block_obj, "width"), atol=1e-5), \
                "Mismatch in block width!"

            # De-normalize the 'pose' feature => y coordinate
            y_norm = state.get(block_obj, "pose")
            by = self.y_lb + (self.y_ub - self.y_lb) * y_norm
            bx = self.workspace_x

            # If the block is initially held (grasp != -1), place near hand
            # otherwise place on table
            grasp_val = state.get(block_obj, "grasp")
            if grasp_val != -1:
                # If an object starts out held, it sits slightly below the EE
                bz = self.workspace_z - self._offset
            else:
                bz = self._table_height + self._obj_len_hgt * 0.5

            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz], self._default_orn,
                physicsClientId=self._physics_client_id
            )

            # If initially held, set up constraint
            if grasp_val != -1:
                # self._held_obj_id = self._detect_held_object()
                # try:
                #     assert self._held_obj_id == block_id, \
                #     "Expected to detect the block as held but did not."
                # except:
                #     breakpoint()
                self._held_obj_id = block_id
                self._create_grasp_constraint()

        # Put any leftover blocks out of view
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._blocks)):
            # block_id = self._block_ids[i]
            block_id = self._blocks[i].id
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, 2.0],
                self._default_orn,
                physicsClientId=self._physics_client_id
            )

        # 2) Reset targets
        target_objs = state.get_objects(self._target_type)
        for i, target_obj in enumerate(target_objs):
            # target_id = self._target_ids[i]
            target_id = self._targets[i].id
            width_unnorm = p.getVisualShapeData(
                target_id, physicsClientId=self._physics_client_id
            )[0][3][1]
            width = width_unnorm / self._max_obj_width * max_width
            assert np.isclose(width, state.get(target_obj, "width"), atol=1e-5)

            # De-normalize the 'pose' feature => y coordinate
            y_norm = state.get(target_obj, "pose")
            ty = self.y_lb + (self.y_ub - self.y_lb) * y_norm
            tx = self.workspace_x
            tz = self._table_height + self._obj_len_hgt * 0.5

            p.resetBasePositionAndOrientation(
                target_id,
                [tx, ty, tz],
                self._default_orn,
                physicsClientId=self._physics_client_id
            )

        # 3) Optionally draw hand regions as debug lines
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert self.using_gui, \
                "use_gui must be True to use pybullet_draw_debug."
            p.removeAllUserDebugItems(physicsClientId=self._physics_client_id)
            for hand_lb, hand_rb in self._get_hand_regions(state):
                y_lb_val = self.y_lb + (self.y_ub - self.y_lb) * hand_lb
                y_rb_val = self.y_lb + (self.y_ub - self.y_lb) * hand_rb
                p.addUserDebugLine(
                    [self.workspace_x, y_lb_val, self._table_height + 1e-4],
                    [self.workspace_x, y_rb_val, self._table_height + 1e-4],
                    [0.0, 0.0, 1.0],
                    lineWidth=5.0,
                    physicsClientId=self._physics_client_id
                )

    def _get_object_ids_for_held_check(self) -> list[int]:
        """We only consider blocks for 'held' detection here."""
        return [blk.id for blk in self._blocks]

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        # Both fetch and panda have grippers parallel to x-axis
        return {
            self._pybullet_robot.left_finger_id: np.array([1., 0., 0.]),
            self._pybullet_robot.right_finger_id: np.array([-1., 0., 0.]),
        }

    def _extract_robot_state(self, state: State) -> np.ndarray:
        """Convert from our domain's features (hand, pose_x, pose_z, etc.) into
        the [x,y,z, qx,qy,qz,qw, fingers] array expected by the PyBullet robot.
        """
        # 1) Determine fingers (closed if any block is being held)
        #    "Held" if any block has 'grasp' != -1
        is_holding_something = False
        for obj in state.get_objects(self._block_type):
            if state.get(obj, "grasp") != -1:
                is_holding_something = True
                break
        if is_holding_something:
            fingers = self._pybullet_robot.closed_fingers
        else:
            fingers = self._pybullet_robot.open_fingers

        # 2) The robot object
        #    By default, we have exactly one robot object in the state
        robot_obj = state.get_objects(self._robot_type)[0]
        # Domain features
        hand_norm = state.get(robot_obj, "hand")
        rx = state.get(robot_obj, "pose_x")
        rz = state.get(robot_obj, "pose_z")

        # De-normalize the hand => actual y coordinate
        ry = self.y_lb + (self.y_ub - self.y_lb) * hand_norm

        # 3) The orientation is fixed; e.g. pointing downward
        #    (If your domain never changes orientation, use default.)
        qx, qy, qz, qw = self.get_robot_ee_home_orn()

        return np.array([rx, ry, rz, qx, qy, qz, qw, fingers], dtype=np.float32)

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Domain-specific feature extraction for blocks, targets, and the
        (robot).
        """
        # # 1) If it's the robot
        # if obj.type == self._robot_type:
        #     # The parent's _get_robot_state_dict() will set x,y,z,fingers
        #     # We can handle additional features here:
        #     rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        #     if feature == "hand":
        #         # Re-normalize the y coordinate
        #         return (ry - self.y_lb) / (self.y_ub - self.y_lb)
        #     elif feature == "pose_x":
        #         return rx
        #     elif feature == "pose_z":
        #         return rz
        #     raise ValueError(f"Unknown robot feature: {feature}")

        # 2) If it's a block
        if obj.type == self._block_type:
            block_id = obj.id
            if feature == "is_block":
                return 1.0
            if feature == "is_target":
                return 0.0
            if feature == "width":
                # Re-compute from shape data
                shape_data = p.getVisualShapeData(
                    block_id, physicsClientId=self._physics_client_id
                )[0]
                # shape_data[3] is halfExtents => (x_half, y_half, z_half)
                # shape_data[3][1] is the block's half-width in Y
                y_half = shape_data[3][1]
                # Convert it to domain-level width
                max_width = max(max(CFG.cover_block_widths),
                                max(CFG.cover_target_widths))
                width = (y_half * 2.0) / self._max_obj_width * max_width
                return width
            if feature == "pose":
                # Recompute from the block's actual y => normalized
                (bx, by, bz), _ = p.getBasePositionAndOrientation(
                    block_id, physicsClientId=self._physics_client_id
                )
                return (by - self.y_lb) / (self.y_ub - self.y_lb)
            if feature == "grasp":
                # If it's the currently-held block, read the pivot offset
                if block_id == self._held_obj_id and \
                   self._held_constraint_id is not None:
                    # Example: read pivot in child's local frame
                    pivot_in_B = p.getConstraintInfo(
                        self._held_constraint_id,
                        physicsClientId=self._physics_client_id
                    )[7]
                    # pivot_in_B is a 3D offset => we only care about y,
                    # then normalize
                    grasp_unnorm = pivot_in_B[1]
                    return grasp_unnorm / (self.y_ub - self.y_lb)
                else:
                    return -1.0
            raise ValueError(f"Unknown block feature: {feature}")

        # 3) If it's a target
        if obj.type == self._target_type:
            target_id = obj.id
            if feature == "is_block":
                return 0.0
            if feature == "is_target":
                return 1.0
            if feature == "width":
                shape_data = p.getVisualShapeData(
                    target_id, physicsClientId=self._physics_client_id
                )[0]
                y_half = shape_data[3][1]
                max_width = max(max(CFG.cover_block_widths),
                                max(CFG.cover_target_widths))
                width = (y_half * 2.0) / self._max_obj_width * max_width
                return width
            if feature == "pose":
                (tx, ty, tz), _ = p.getBasePositionAndOrientation(
                    target_id, physicsClientId=self._physics_client_id
                )
                return (ty - self.y_lb) / (self.y_ub - self.y_lb)
            raise ValueError(f"Unknown target feature: {feature}")

        # If we somehow get here, no type matched
        raise ValueError(f"Unknown object type or feature: {obj}, {feature}")

    # -----------------------------------------------------------------------
    # Step logic (unchanged except for removing direct calls to _get_state())
    # -----------------------------------------------------------------------
    def step(self, action: Action, render_obs: bool = False) -> State:
        """Override to handle the Cover domain's 'hand region' constraint
        before calling the parent's step().
        """
        # Check if the pick/place position satisfies the hand constraints
        if not self._satisfies_hand_contraints(action):
            # Constraint violated => no-op
            return self._current_state.copy()

        # Otherwise, proceed with normal PyBullet step
        next_state = super().step(action, render_obs=render_obs)
        return next_state
    
    def _satisfies_hand_contraints(self, action: Action) -> bool:
        joint_positions = action.arr.tolist()
        _, ry, rz = self._pybullet_robot_fk.forward_kinematics(joint_positions
                                                               ).position

        if self._is_below_z_threshold(rz):
              return self._is_in_valid_hand_region(ry)
        return True

    def _is_below_z_threshold(self, rz: float) -> bool:
        """Check if the z position is below the threshold."""
        z_thresh = (self.pickplace_z + self.workspace_z) / 2
        return rz < z_thresh

    def _is_in_valid_hand_region(self, ry: float) -> bool:
        """Check if the hand position is within any valid hand region."""
        hand = (ry - self.y_lb) / (self.y_ub - self.y_lb)
        hand_regions = self._get_hand_regions(self._current_state)
        return any(lb <= hand <= rb for lb, rb in hand_regions)