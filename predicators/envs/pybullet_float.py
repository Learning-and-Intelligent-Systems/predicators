"""Optimized single-object communicating vessel example.

python predicators/main.py --approach oracle --env pybullet_float \
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --debug \
--sesame_check_expected_atoms False
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import (create_object, update_object,
                                            sample_collision_free_2d_positions)
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


def create_water_body(size_z,
                      size_x=0.2,
                      size_y=0.2,
                      base_position=(0, 0, 0),
                      color=[0, 0, 1, 0.8],
                      physics_client_id=None):
    """Create a semi-transparent 'water' box in PyBullet."""
    water_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size_x / 2, size_y / 2, size_z / 2],
        rgbaColor=color,
        physicsClientId=physics_client_id)
    base_position = list(base_position)
    base_position[2] += size_z / 2  # shift up so box sits on base_position
    water_body_id = p.createMultiBody(baseMass=0,
                                      baseVisualShapeIndex=water_visual,
                                      basePosition=base_position,
                                      physicsClientId=physics_client_id)
    return water_body_id


class PyBulletFloatEnv(PyBulletEnv):
    """Communicating vessel environment with a single 'vessel' object (plus
    blocks). The vessel has x, y, z, water_height. Internally, we treat two
    compartments but share a single water_height because the fluid is
    connected.

    Optimizations:
    - Only update water geometry if water level changes.
    - Water level changes only when a block "enters" or "exits" water.
    """

    # Vessel geometry / URDF config
    COMM_VESSEL_URDF: ClassVar[str] = "urdf/comm_vessel2.urdf"
    CONTAINER_OPENING_LEN: ClassVar[float] = 0.1
    CONTAINER_GAP: ClassVar[float] = 0.3
    TUBE_OPENING_LEN: ClassVar[float] = 0.05
    VESSEL_WALL_THICKNESS: ClassVar[float] = 0.01

    # Cross-sectional area for each compartment => total area is 2 *
    # CONTAINER_AREA
    CONTAINER_AREA: ClassVar[float] = CONTAINER_OPENING_LEN**2

    # Table / workspace config
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0., 0., np.pi / 2])

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2

    # Robot config
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2
    max_angular_vel: ClassVar[float] = np.pi / 4

    # Camera parameters
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 60
    _camera_pitch: ClassVar[float] = -30
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Vessel placement
    VESSEL_BASE_X: ClassVar[float] = 0.55
    VESSEL_BASE_Y: ClassVar[float] = 1.35

    # Water
    initial_water_height: ClassVar[float] = 0.13
    z_ub_water: ClassVar[float] = 0.5

    # Blocks
    block_size: ClassVar[float] = 0.06
    block_mass: ClassVar[float] = 0.05
    block_friction: ClassVar[float] = 1.2
    block_color_light: ClassVar[Tuple[float, float, float,
                                      float]] = (0.0, 1.0, 0.0, 1.0)
    block_color_heavy: ClassVar[Tuple[float, float, float,
                                      float]] = (1.0, 0.6, 0.0, 1.0)

    # Types
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _vessel_type = Type("vessel", ["x", "y", "z", "water_height"])
    _block_type = Type("block",
                       ["x", "y", "z", "in_water", "is_held"],
                       sim_features=["id", "is_light"])

    def __init__(self, use_gui: bool = True) -> None:
        self._robot = Object("robot", self._robot_type)
        self._vessel = Object("vessel", self._vessel_type)
        self._block0 = Object("block0", self._block_type)
        self._block1 = Object("block1", self._block_type)
        self._block2 = Object("block2", self._block_type)
        self._blocks = [self._block0, self._block1, self._block2]

        super().__init__(use_gui)

        self._InWater = Predicate("InWater", [self._block_type],
                                  self._InWater_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._block_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)

        # Track water geometry in PyBullet
        self._water_ids: Dict[str, Optional[int]] = {
            "left": None,
            "right": None
        }

        # Keep track of which blocks are currently displacing water
        # i.e., which blocks have "fully entered" the water
        self._block_is_displacing: Dict[Object, bool] = {
            self._block0: False,
            self._block1: False,
        }

        self._held_obj_id = None

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_float"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InWater, self._HandEmpty, self._Holding}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return set()

    @property
    def types(self) -> Set[Type]:
        return {self._vessel_type, self._block_type, self._robot_type}

    # -------------------------------------------------------------------------
    # PyBullet Setup

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool) -> Tuple[int, Any, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Table
        table_id = create_object(asset_path="urdf/table.urdf",
                                 position=cls.table_pos,
                                 orientation=cls.table_orn,
                                 scale=1.0,
                                 use_fixed_base=True,
                                 physics_client_id=physics_client_id)
        bodies["table_id"] = table_id

        # Vessel
        vessel_id = create_object(asset_path=cls.COMM_VESSEL_URDF,
                                  position=(cls.VESSEL_BASE_X,
                                            cls.VESSEL_BASE_Y, cls.z_lb),
                                  scale=1.0,
                                  use_fixed_base=True,
                                  physics_client_id=physics_client_id)
        bodies["vessel_id"] = vessel_id

        # Three blocks
        block_ids = []
        for _ in range(3):
            body_id = create_pybullet_block(
                color=[1, 1, 1, 1],
                half_extents=[cls.block_size / 2] * 3,
                mass=cls.block_mass,
                friction=cls.block_friction,
                orientation=[0, 0, 0, 1],
                physics_client_id=physics_client_id)
            block_ids.append(body_id)
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._vessel.id = pybullet_bodies["vessel_id"]
        num_blocks = len(pybullet_bodies["block_ids"])
        for i, (blk, id) in enumerate(zip(self._blocks, 
                                          pybullet_bodies["block_ids"])):
            if i == num_blocks - 1:
                blk.is_light = 1.0
            else:
                blk.is_light = 0.0
            blk.id = id
        

    # -------------------------------------------------------------------------
    # State Management

    def _get_object_ids_for_held_check(self) -> List[int]:
        return [block_obj.id for block_obj in self._blocks]
    
    def _create_task_specific_objects(self, state: State) -> None:
        pass

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._block_type:
            # if feature == "is_light":
            #     return self._is_block_light(obj.id)
            if feature == "in_water":
                (bx, by, bz), _ = p.getBasePositionAndOrientation(
                    obj.id, physicsClientId=self._physics_client_id)
                in_water_val = 0.0
                # If block is within bounding region and top is below water surface
                if self._is_in_left_compartment(bx, by):
                    if bz < self._current_water_height:
                        in_water_val = 1.0
                elif self._is_in_right_compartment(bx, by):
                    if bz < self._current_water_height:
                        in_water_val = 1.0
                return in_water_val
        elif obj.type == self._vessel_type:
            if feature == "water_height":
                return self._current_water_height
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _reset_custom_env_state(self, state: State) -> None:

        # Initialize water level
        self._current_water_height = state.get(self._vessel, "water_height")
        # Clear old water
        for wid in self._water_ids.values():
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
        self._water_ids = {"left": None, "right": None}

        # Reset blocks
        for blk in self._blocks:
            # Set block's color based on is_light
            # update_object(blk.id,
            #               color=PyBulletFloatEnv.block_color_light \
            #                 if state.get(blk, "is_light") > 0.5
            #                 else PyBulletFloatEnv.block_color_heavy,
            #               physics_client_id=self._physics_client_id)
            # Set block's color randomly
            update_object(blk.id, 
                          color=self._train_rng.choice(self._obj_colors),
                          physics_client_id=self._physics_client_id)
            # Re-initialize displacing to False
            self._block_is_displacing[blk] = False

        # Re-draw water
        self._create_or_update_water(force_redraw=True)

        vx = state.get(self._vessel, "x")
        wx = vx + self.CONTAINER_OPENING_LEN + self.CONTAINER_GAP / 2
        wy, wz = state.get(self._vessel, "y"), state.get(self._vessel, "z")
        create_water_body(size_x=self.CONTAINER_GAP,
                          size_y=self.TUBE_OPENING_LEN,
                          size_z=self.TUBE_OPENING_LEN,
                          base_position=(wx, wy, wz),
                          color=[0.5, 0.5, 1, 0.5],
                          physics_client_id=self._physics_client_id)

    def step(self, action: Action, render_obs: bool = False) -> State:
        next_state = super().step(action, render_obs=render_obs)
        # Check if blocks entering/exiting water changed its level
        changed = self._update_water_level_if_needed(next_state)
        if changed:
            self._create_or_update_water(force_redraw=True)
        # Keep light blocks floating on water surface
        self._float_light_blocks(next_state)

        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    def _float_light_blocks(self, state: State) -> None:
        """Force each light, unheld block in a container compartment to float
        at the surface."""
        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel.id, physicsClientId=self._physics_client_id)
        water_surface_z = vz + self._current_water_height

        for blk in self._blocks:
            # Skip blocks that are heavy or being held
            if blk.is_light < 0.5:
                continue
            if state.get(blk, "is_held") > 0.5:
                continue

            # Get latest position from PyBullet
            (bx, by, bz), orn = p.getBasePositionAndOrientation(
                blk.id, physicsClientId=self._physics_client_id)
            # Check if the block is inside either compartment
            if (self._is_in_left_compartment(bx, by)
                    or self._is_in_right_compartment(bx, by)):
                # Float it: set Z so that the entire block is above the water
                float_z = water_surface_z + self.block_size / 2.0
                p.resetBasePositionAndOrientation(
                    blk.id, (bx, by, float_z),
                    orn,
                    physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Water-Level Logic

    @property
    def _current_water_height(self) -> float:
        return getattr(self, "__current_water_height", 0.0)

    @_current_water_height.setter
    def _current_water_height(self, val: float) -> None:
        setattr(self, "__current_water_height", val)

    def _update_water_level_if_needed(self, state: State) -> bool:
        """Check if any block's top crosses the water line.

        If so, update water displacement and recalc water level. Returns
        True if the water level changed, else False.
        """
        if CFG.float_water_level_doesnt_raise:
            return False
        old_height = self._current_water_height
        # Start from total volume = 2 compartments * old_height * area
        old_volume = 2.0 * self.CONTAINER_AREA * old_height

        new_displaced_volume = 0.0
        # Track which blocks are displacing at end
        blocks_displacing_now = {}

        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel.id, physicsClientId=self._physics_client_id)

        for blk in self._blocks:
            # Get block's top (we approximate block top as its center + half)
            is_light = blk.is_light > 0.5
            bx = state.get(blk, "x")
            by = state.get(blk, "y")
            bz = state.get(blk, "z")
            top_z = bz + (self.block_size / 2.0)

            # Are we inside a compartment XY?
            in_left = self._is_in_left_compartment(bx, by)
            in_right = self._is_in_right_compartment(bx, by)
            if is_light or not (in_left or in_right):
                # Not above any container => no displacement
                blocks_displacing_now[blk] = False
                continue

            # Water surface world Z
            surface_z = vz + self._current_water_height

            was_displacing = self._block_is_displacing[blk]

            # Condition for "entering water" => block top is below water surface
            # (i.e. the entire block is now submerged)
            if top_z < surface_z:
                # => This block is fully submerged => displacing
                blocks_displacing_now[blk] = True
            else:
                # => Not fully submerged
                blocks_displacing_now[blk] = False

            # If we just toggled from not displacing to displacing,
            # or vice versa, we must recalc total volume
            # We'll do final calculation after we see who is displacing
        # end for

        # Now let's see how many blocks are displacing
        # Each displacing block adds block_size^3
        for blk, is_disp in blocks_displacing_now.items():
            if is_disp:
                new_displaced_volume += (self.block_size**3)

        # If the new_displaced_volume is the same as the old displaced volume
        # we had, the water height won't change. But we don't store old displaced
        # volume separately; we store old_height. We can compute old displaced
        # from ( old_volume - 2*area*lowest_level_of_water ).
        # Easiest approach: just see how many blocks were displacing before vs. now.

        old_num_displacing = sum(self._block_is_displacing.values())
        new_num_displacing = sum(blocks_displacing_now.values())
        if old_num_displacing == new_num_displacing:
            # => no change in net block submersion => water won't change
            # but let's confirm that none have changed...
            # If it's the same number of blocks but actually a different set,
            # that might matter. For identical blocks of same volume,
            # the net displacement is the same. So no water level change either.
            changed = False
        else:
            # old_volume was water + old displaced. We want to remove old
            # displaced and add new displaced. So new_volume = old_volume
            #                                  - old_displaced + new_displaced
            increase_factor = 2
            old_displaced_vol = old_num_displacing * (self.block_size**3)
            new_displaced_vol = new_num_displacing * (self.block_size**
                                                      3) * increase_factor
            new_volume = (old_volume - old_displaced_vol) + new_displaced_vol

            # water + blocks => water in 2 compartments => new_height
            new_height = new_volume / (2.0 * self.CONTAINER_AREA)
            new_height = np.clip(new_height, 0.0, self.z_ub_water)
            self._current_water_height = new_height
            changed = not np.isclose(new_height, old_height, atol=1e-5)

        # Update our record of who is displacing
        for blk in self._blocks:
            self._block_is_displacing[blk] = blocks_displacing_now[blk]

        return changed

    def _create_or_update_water(self, force_redraw: bool = False) -> None:
        """Draw water boxes for left & right compartments if water changed."""
        # If we only update water on changes, we can skip if !force_redraw
        if not force_redraw:
            return

        # Remove old boxes
        for side, wid in self._water_ids.items():
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
                self._water_ids[side] = None

        # If water height is zero => no water to draw
        if self._current_water_height <= 0:
            return

        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel.id, physicsClientId=self._physics_client_id)

        # Left
        lx = vx + self.CONTAINER_OPENING_LEN / 2
        ly = vy
        left_id = create_water_body(size_z=self._current_water_height,
                                    size_x=self.CONTAINER_OPENING_LEN,
                                    size_y=self.CONTAINER_OPENING_LEN,
                                    base_position=(lx, ly, vz),
                                    physics_client_id=self._physics_client_id)
        self._water_ids["left"] = left_id

        # Right
        rx_offset = (self.CONTAINER_OPENING_LEN + self.CONTAINER_GAP +
                     self.CONTAINER_OPENING_LEN / 2)
        rx = vx + rx_offset
        ry = vy
        right_id = create_water_body(size_z=self._current_water_height,
                                     size_x=self.CONTAINER_OPENING_LEN,
                                     size_y=self.CONTAINER_OPENING_LEN,
                                     base_position=(rx, ry, vz),
                                     physics_client_id=self._physics_client_id)
        self._water_ids["right"] = right_id

    # -------------------------------------------------------------------------
    # Geometry checks
    def _is_in_left_compartment(self, bx: float, by: float) -> bool:
        (vx, vy, _) = self._get_vessel_base_position()
        x_min = vx
        x_max = vx + self.CONTAINER_OPENING_LEN
        y_min = vy - self.CONTAINER_OPENING_LEN / 2
        y_max = vy + self.CONTAINER_OPENING_LEN / 2
        return (x_min <= bx <= x_max) and (y_min <= by <= y_max)

    def _is_in_right_compartment(self, bx: float, by: float) -> bool:
        (vx, vy, _) = self._get_vessel_base_position()
        x_min = vx + self.CONTAINER_OPENING_LEN + self.CONTAINER_GAP
        x_max = x_min + self.CONTAINER_OPENING_LEN
        y_min = vy - self.CONTAINER_OPENING_LEN / 2
        y_max = vy + self.CONTAINER_OPENING_LEN / 2
        return (x_min <= bx <= x_max) and (y_min <= by <= y_max)

    def _get_vessel_base_position(self) -> Tuple[float, float, float]:
        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel.id, physicsClientId=self._physics_client_id)
        return (vx, vy, vz)

    # -------------------------------------------------------------------------
    # Predicates
    @staticmethod
    def _InWater_holds(state: State, objects: Sequence[Object]) -> bool:
        (block, ) = objects
        return state.get(block, "in_water") > 0.5

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, block = objects
        return state.get(block, "is_held") > 0.5

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    # -------------------------------------------------------------------------
    # Helpers for tasks
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int,
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            # Robot
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }
            # Vessel
            vessel_dict = {
                "x": self.VESSEL_BASE_X,
                "y": self.VESSEL_BASE_Y,
                "z": self.z_lb,
                "water_height": self.initial_water_height,
            }
            # Blocks
            num_free_blocks = len(self._blocks) - 1
            block_xy_positions = sample_collision_free_2d_positions(
                num_samples=num_free_blocks,
                x_range=(self.VESSEL_BASE_X + self.CONTAINER_OPENING_LEN + \
                            self.block_size,
                         self.VESSEL_BASE_X + self.CONTAINER_OPENING_LEN +
                            self.CONTAINER_GAP - self.block_size),
                y_range=(self.y_lb + self.block_size * 2.5, 
                         self.VESSEL_BASE_Y + self.CONTAINER_OPENING_LEN / 2),
                shape_type="rectangle",
                shape_params=[self.block_size, self.block_size, 0],
                rng=self._train_rng,
            )
            # Adding z values
            block_positions = [(pos[0], pos[1], self.z_lb + self.block_size / 2)
                               for pos in block_xy_positions]
            # Add the block inside the vessel
            block_positions.append((
                self.VESSEL_BASE_X + self.CONTAINER_OPENING_LEN +
                    self.CONTAINER_GAP + self.CONTAINER_OPENING_LEN / 2,
                self.VESSEL_BASE_Y,
                self.initial_water_height + self.block_size / 2,
            ))

            init_dict = {self._robot: robot_dict, self._vessel: vessel_dict}
            for b_obj, b_pos in zip(self._blocks, block_positions):
                b_vals = {
                    "x": b_pos[0],
                    "y": b_pos[1],
                    "z": b_pos[2],
                    "in_water": 0.0,
                    "is_held": 0.0,
                }
                init_dict[b_obj] = b_vals

            init_state = utils.create_state_from_dict(init_dict)

            # e.g. goal: all blocks in water
            first_two_blocks = self._blocks[:2]
            goal_atoms = {
                GroundAtom(self._InWater, [b])
                for b in first_two_blocks
            }
            # goal_atoms = set()
            goal_atoms.add(
                GroundAtom(self._Holding, [self._robot, self._blocks[2]]))
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    import time

    CFG.seed = 0
    CFG.pybullet_sim_steps_per_action = 1
    env = PyBulletFloatEnv(use_gui=True)
    task = env._make_tasks(1, np.random.default_rng(0))[0]
    env._reset_state(task.init)

    while True:
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))
        env.step(action)
        time.sleep(0.01)
