"""A PyBullet version of Blocks, refactored to use the new PyBulletEnv
hooks."""

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.blocks import BlocksEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_pybullet_block
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, State


class PyBulletBlocksEnv(PyBulletEnv, BlocksEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py
    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.62)

    # Table parameters
    table_height: ClassVar[float] = 0.4
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, table_height / 2)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        super().__init__(use_gui, **kwargs)
        # Store references
        self._table_id: int = -1
        # self._block_ids: List[int] = []
        # Maps PyBullet IDs -> BlocksEnv block objects
        self._block_id_to_block: Dict[int, Object] = {}
        self._prev_held_obj_id: Optional[int] = None

    # -----------------------------------------------------------------------
    # Required Hooks
    # -----------------------------------------------------------------------
    @classmethod
    def get_name(cls) -> str:
        return "pybullet_blocks"

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Create the plane, table, debug lines, and maximum number of
        blocks."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Load the table
        table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        p.resetBasePositionAndOrientation(table_id,
                                          cls._table_pose,
                                          cls._table_orientation,
                                          physicsClientId=physics_client_id)
        bodies["table_id"] = table_id

        # Optional debug lines
        if CFG.pybullet_draw_debug and using_gui:  # pragma: no cover
            cls._draw_table_workspace_debug_lines(physics_client_id)

        # Create the maximum number of blocks
        num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
        block_ids = []
        block_size = CFG.blocks_block_size
        for i in range(num_blocks):
            color = cls._obj_colors[i % len(cls._obj_colors)]
            half_extents = (block_size / 2.0, block_size / 2.0,
                            block_size / 2.0)
            block_id = create_pybullet_block(
                color=color,
                half_extents=half_extents,
                mass=cls._obj_mass,
                friction=cls._obj_friction,
                physics_client_id=physics_client_id,
            )
            block_ids.append(block_id)
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to table and block IDs."""
        self._table_id = pybullet_bodies["table_id"]
        self._block_ids = pybullet_bodies["block_ids"]
        for blk, blk_id in zip(self._blocks, self._block_ids):
            blk.id = blk_id

    def _set_domain_specific_state(self, state: State) -> None:
        """Set block positions, grasp constraints, out-of-view placement, ID
        mapping, and block colors."""
        block_objs = state.get_objects(self._block_type)

        # Place the relevant blocks
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            bx = state.get(block_obj, "pose_x")
            by = state.get(block_obj, "pose_y")
            bz = state.get(block_obj, "pose_z")
            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        # If there is a held block, create the constraint
        held_block = self._get_held_block(state)
        if held_block is not None:
            self._force_grasp_object(held_block)

        # Teleport any leftover blocks out of view
        block_size = CFG.blocks_block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * block_size],
                self._default_orn,
                physicsClientId=self._physics_client_id)

        self._block_id_to_block.clear()

        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            self._block_id_to_block[block_id] = block_obj
            r = state.get(block_obj, "color_r")
            g = state.get(block_obj, "color_g")
            b = state.get(block_obj, "color_b")
            p.changeVisualShape(block_id,
                                linkIndex=-1,
                                rgbaColor=(r, g, b, 1.0),
                                physicsClientId=self._physics_client_id)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Called by the parent class when constructing the `PyBulletState`.

        We read off the relevant block or robot features from PyBullet.
        """

        if obj.type == self._block_type:
            # Find the PyBullet ID for this block
            # (One approach: invert the dictionary)
            block_id = None
            for bid, block_obj in self._block_id_to_block.items():
                if block_obj == obj:
                    block_id = bid
                    break
            if block_id is None:
                raise ValueError(f"Object {obj} not found in "
                                 f"_block_id_to_block")

            # Pose from PyBullet
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)

            if feature == "pose_x":
                return bx
            if feature == "pose_y":
                return by
            if feature == "pose_z":
                return bz
            if feature == "held":
                # Compare block_id with self._held_obj_id
                return 1.0 if block_id == self._held_obj_id else 0.0
            if feature == "color_r":
                # read from PyBullet
                visual_data = p.getVisualShapeData(
                    block_id, physicsClientId=self._physics_client_id)[0]
                (r, g, b, _a) = visual_data[7]
                return r
            if feature == "color_g":
                visual_data = p.getVisualShapeData(
                    block_id, physicsClientId=self._physics_client_id)[0]
                (r, g, b, _a) = visual_data[7]
                return g
            if feature == "color_b":
                visual_data = p.getVisualShapeData(
                    block_id, physicsClientId=self._physics_client_id)[0]
                (r, g, b, _a) = visual_data[7]
                return b
            # If you have an extra "clear" feature (BlocksEnvClear),
            # you can either compute it purely from the state, or store it in
            # the underlying simulator in some way. Typically we'd do it purely
            # from the state in the abstract BlocksEnv logic, so you might just:
            if feature == "clear":
                # Let the base BlocksEnv handle it or do something here
                # (Often you'd do something like recomputing from the scene, or
                # just trust the parent's logic at the symbolic level.)
                pass

            raise ValueError(f"Unknown block feature: {feature}")

        raise ValueError(f"Unknown object type {obj.type} or feature "
                         f"{feature}")

    def step(self, action: Action, render_obs: bool = False) -> State:
        self._prev_held_obj_id = self._held_obj_id
        return super().step(action, render_obs=render_obs)

    def _domain_specific_step(self) -> None:
        if CFG.blocks_high_towers_are_unstable:
            state = self._get_state()
            self._apply_force_to_high_towers(state)

    def _extract_robot_state(self, state: State) -> np.ndarray:
        """As needed, parse from the robot's `pose_x`, `pose_y`, `pose_z`,
        `fingers` in the `State` to the 8D array [rx,ry,rz, qx,qy,qz,qw,
        finger]."""
        # Usually these features exist in your robot object:
        robot_obj = state.get_objects(self._robot_type)[0]
        rx = state.get(robot_obj, "pose_x")
        ry = state.get(robot_obj, "pose_y")
        rz = state.get(robot_obj, "pose_z")
        f = state.get(robot_obj, "fingers")
        # Convert from 0 or 1 to the actual PyBullet joint positions
        f = self._fingers_state_to_joint(self._pybullet_robot, f)
        # If your environment uses a constant orientation (like pointing down)
        # you can just do:
        qx, qy, qz, qw = self.get_robot_ee_home_orn()
        return np.array([rx, ry, rz, qx, qy, qz, qw, f], dtype=np.float32)

    def _get_robot_state_dict(self) -> Dict[str, float]:
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(self._pybullet_robot, rf)
        return {
            "pose_x": rx,
            "pose_y": ry,
            "pose_z": rz,
            "fingers": fingers,
        }

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return the IDs of blocks for which we might be checking 'held'
        contact."""
        return list(self._block_id_to_block.keys())

    def _get_state(self, _render_obs: bool = False) -> State:
        """Create a State based on the current PyBullet state.

        Uses self._block_id_to_block mapping instead of obj.id.
        """
        state_dict = {}

        # Get robot state.
        rx, ry, rz, _, _, _, _, rf = self._pybullet_robot.get_state()
        fingers = self._fingers_joint_to_state(self._pybullet_robot, rf)
        state_dict[self._robot] = np.array([rx, ry, rz, fingers])
        joint_positions = self._pybullet_robot.get_joints()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_obj_id)
            visual_data = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id)[0]
            r, g, b, _ = visual_data[7]
            state_dict[block] = np.array([bx, by, bz, held, r, g, b])

        state = utils.PyBulletState(state_dict,
                                    simulator_state=joint_positions)
        return state

    # -----------------------------------------------------------------------
    # Domain-Specific Logic
    # -----------------------------------------------------------------------
    def _force_grasp_object(self, block: Object) -> None:
        """Manually create a fixed constraint for a block that is marked 'held'
        in the State.

        Called from _set_domain_specific_state().
        """
        # Find block's pybullet ID
        block_id = None
        for bid, block_obj in self._block_id_to_block.items():
            if block_obj == block:
                block_id = bid
                break
        if block_id is None:
            return
        # Set the held object id and create the grasp constraint.
        self._held_obj_id = block_id
        self._create_grasp_constraint()

    # If you want a custom step() for PyBullet blocks, you can override it here.
    # However, if there's no domain-specific constraint, you might not need to.
    def _apply_force_to_high_towers(self, state: State) -> None:
        """Apply downward force to blocks that form towers of height 3."""
        # Only apply force if we just released a block
        just_released_obj = self._just_released_object(state)
        # logging.debug(f"just_released_obj: {just_released_obj}")
        if just_released_obj is None:
            return
        if self._count_block_height(state, just_released_obj) >= 2:
            # Apply downward force
            force = [0, -100, 0]
            pos = p.getBasePositionAndOrientation(
                just_released_obj.id,
                physicsClientId=self._physics_client_id)[0]
            p.applyExternalForce(
                just_released_obj.id,
                -1,  # -1 for base link
                force,
                pos,
                p.WORLD_FRAME,
                physicsClientId=self._physics_client_id)

    def _just_released_object(self, state: State) -> Optional[Object]:
        """Check if we just released an object in this step."""
        # return the block Object that just released
        if self._held_obj_id is None and self._prev_held_obj_id is not None:
            for block_obj in state.get_objects(self._block_type):
                if block_obj.id == self._prev_held_obj_id:
                    return block_obj
        return None

    # -----------------------------------------------------------------------
    # Task Generation
    # -----------------------------------------------------------------------
    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = super()._get_tasks(num_tasks, possible_num_blocks, rng)
        return self._add_pybullet_state_to_tasks(tasks)

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    @staticmethod
    def _draw_table_workspace_debug_lines(physics_client_id: int) -> None:
        """Optionally draws red lines marking the workspace on the table."""
        # Draw the bounding lines at x_lb, x_ub, y_lb, y_ub
        x_lb = BlocksEnv.x_lb
        x_ub = BlocksEnv.x_ub
        y_lb = BlocksEnv.y_lb
        y_ub = BlocksEnv.y_ub
        z = BlocksEnv.table_height

        p.addUserDebugLine([x_lb, y_lb, z], [x_ub, y_lb, z], [1.0, 0.0, 0.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([x_lb, y_ub, z], [x_ub, y_ub, z], [1.0, 0.0, 0.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([x_lb, y_lb, z], [x_lb, y_ub, z], [1.0, 0.0, 0.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        p.addUserDebugLine([x_ub, y_lb, z], [x_ub, y_ub, z], [1.0, 0.0, 0.0],
                           lineWidth=5.0,
                           physicsClientId=physics_client_id)
        # Possibly more debug text...


if __name__ == "__main__":
    # Run a simple simulation to test the environment.
    import time

    CFG.seed = 0
    CFG.env = "pybullet_blocks"
    CFG.num_train_tasks = 1
    env = PyBulletBlocksEnv(use_gui=True)
    _task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
    env._set_state(_task.init)  # pylint: disable=protected-access

    while True:
        # Hold the robot's current joint positions so the arm doesn't swing
        # toward URDF home and disturb the blocks.
        _act = Action(np.array(env._pybullet_robot.get_joints()))  # pylint: disable=protected-access
        env.step(_act)
        time.sleep(0.01)
