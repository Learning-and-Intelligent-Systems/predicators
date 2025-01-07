"""
Optimized single-object communicating vessel example.

python predicators/main.py --approach oracle --env pybullet_float \
--seed 0 --num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_sim_steps_per_action 1 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --debug
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type


def create_water_body(size_z, size_x=0.2, size_y=0.2,
                      base_position=(0, 0, 0),
                      color=[0, 0, 1, 0.8],
                      physics_client_id=None):
    """Create a semi-transparent 'water' box in PyBullet."""
    water_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size_x / 2, size_y / 2, size_z / 2],
        rgbaColor=color,
        physicsClientId=physics_client_id
    )
    base_position = list(base_position)
    base_position[2] += size_z / 2  # shift up so box sits on base_position
    water_body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=water_visual,
        basePosition=base_position,
        physicsClientId=physics_client_id
    )
    return water_body_id


class PyBulletFloatEnv(PyBulletEnv):
    """Communicating vessel environment with a single 'vessel' object (plus blocks).
    The vessel has x, y, z, water_height. Internally, we treat two compartments
    but share a single water_height because the fluid is connected.

    Optimizations:
    - Only update water geometry if water level changes.
    - Water level changes only when a block "enters" or "exits" water.
    """

    # -------------------------------------------------------------------------
    # Vessel geometry / URDF config
    COMM_VESSEL_URDF: ClassVar[str] = "urdf/comm_vessel2.urdf"
    CONTAINER_OPENING_LEN: ClassVar[float] = 0.1
    CONTAINER_GAP: ClassVar[float] = 0.3
    TUBE_OPENING_LEN: ClassVar[float] = 0.05
    VESSEL_WALL_THICKNESS: ClassVar[float] = 0.01

    # Cross-sectional area for each compartment => total area is 2 * CONTAINER_AREA
    CONTAINER_AREA: ClassVar[float] = CONTAINER_OPENING_LEN**2

    # -------------------------------------------------------------------------
    # Table / workspace config
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = 0.2
    z_ub: ClassVar[float] = 0.75

    table_pos: ClassVar[Pose3D] = (0.75, 1.35, 0.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0., 0., np.pi/2])

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
    VESSEL_BASE_Y: ClassVar[float] = 1.3

    # Water
    initial_water_height: ClassVar[float] = 0.13
    z_ub_water: ClassVar[float] = 0.5

    # Blocks
    block_size: ClassVar[float] = 0.06
    block_mass: ClassVar[float] = 0.01

    # Types
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _vessel_type = Type("vessel", ["x", "y", "z", "water_height"])
    _block_type = Type("block", ["x", "y", "z", "in_water", "is_held"])

    def __init__(self, use_gui: bool = True) -> None:
        self._robot = Object("robot", self._robot_type)
        self._vessel = Object("vessel", self._vessel_type)
        self._block0 = Object("block0", self._block_type)
        self._block1 = Object("block1", self._block_type)
        self._block2 = Object("block2", self._block_type)
        self._block_objs = [self._block0, self._block1, self._block2]

        super().__init__(use_gui)

        self._InWater = Predicate("InWater", [self._block_type],
                                  self._InWater_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._block_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)


        # Track water geometry in PyBullet
        self._water_ids: Dict[str, Optional[int]] = {"left": None, "right": None}

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
        cls,
        using_gui: bool
    ) -> Tuple[int, Any, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui)

        # Table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id
        )
        bodies["table_id"] = table_id

        # Vessel
        vessel_id = create_object(
            asset_path=cls.COMM_VESSEL_URDF,
            position=(cls.VESSEL_BASE_X, cls.VESSEL_BASE_Y, cls.z_lb),
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id
        )
        bodies["vessel_id"] = vessel_id

        # Two blocks
        block_ids = []
        for i in range(3):
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[cls.block_size/2]*3,
                physicsClientId=physics_client_id
            )
            visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[cls.block_size/2]*3,
                rgbaColor=[1.0, 0.6, 0.0, 1.0],
                physicsClientId=physics_client_id
            )
            init_z = cls.z_lb + cls.block_size/2
            body_id = p.createMultiBody(
                baseMass=cls.block_mass,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=(0, 0, init_z),
                physicsClientId=physics_client_id
            )
            block_ids.append(body_id)
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._vessel.id = pybullet_bodies["vessel_id"]
        self._block0.id = pybullet_bodies["block_ids"][0]
        self._block1.id = pybullet_bodies["block_ids"][1]
        self._block2.id = pybullet_bodies["block_ids"][2]

    # -------------------------------------------------------------------------
    # State Management

    def _get_object_ids_for_held_check(self) -> List[int]:
        return [block_obj.id for block_obj in self._block_objs]

    def _get_state(self) -> State:
        state_dict: Dict[Object, Dict[str, float]] = {}

        # Robot
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
        _, tilt, wrist = p.getEulerFromQuaternion([qx, qy, qz, qw])
        state_dict[self._robot] = {
            "x": rx,
            "y": ry,
            "z": rz,
            "fingers": self._fingers_joint_to_state(self._pybullet_robot, rf),
            "tilt": tilt,
            "wrist": wrist
        }

        # Vessel
        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel.id, physicsClientId=self._physics_client_id)
        vessel_dict = {
            "x": vx,
            "y": vy,
            "z": vz,
            "water_height": self._current_water_height
        }
        state_dict[self._vessel] = vessel_dict

        # Blocks
        for blk in self._block_objs:
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                blk.id, physicsClientId=self._physics_client_id)
            in_water_val = 0.0
            # If block is within bounding region and top is below water surface
            if self._is_in_left_compartment(bx, by):
                if bz < self._current_water_height + vz:
                    in_water_val = 1.0
            elif self._is_in_right_compartment(bx, by):
                if bz < self._current_water_height + vz:
                    in_water_val = 1.0
            state_dict[blk] = {
                "x": bx,
                "y": by,
                "z": bz,
                "in_water": in_water_val
            }
            is_held_val = 1.0 if blk.id == self._held_obj_id else 0.0
            state_dict[blk]["is_held"] = is_held_val

        py_state = utils.create_state_from_dict(state_dict)
        joint_positions = self._pybullet_robot.get_joints()
        return utils.PyBulletState(
            py_state.data,
            simulator_state={"joint_positions": joint_positions})

    def _reset_state(self, state: State) -> None:
        super()._reset_state(state)
        # Initialize water level
        self._current_water_height = state.get(self._vessel, "water_height")
        # Clear old water
        for wid in self._water_ids.values():
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
        self._water_ids = {"left": None, "right": None}

        # Reset blocks
        for blk in self._block_objs:
            bx = state.get(blk, "x")
            by = state.get(blk, "y")
            bz = state.get(blk, "z")
            update_object(blk.id,
                          position=(bx, by, bz),
                          physics_client_id=self._physics_client_id)
            # Re-initialize displacing to False
            self._block_is_displacing[blk] = False

        # Re-attach blocks that have is_held=1
        for block_obj in self._block_objs:
            if state.get(block_obj, "is_held") > 0.5:
                self._attach(block_obj.id, self._pybullet_robot)
                self._held_obj_id = block_obj.id

        # Re-draw water
        self._create_or_update_water(force_redraw=True)

        vx = state.get(self._vessel, "x")
        wx = vx + self.CONTAINER_OPENING_LEN + self.CONTAINER_GAP/2
        wy, wz = state.get(self._vessel, "y"), state.get(self._vessel, "z")
        create_water_body(
            size_x=self.CONTAINER_GAP,
            size_y=self.TUBE_OPENING_LEN,
            size_z=self.TUBE_OPENING_LEN,
            base_position=(wx, wy, wz),
            color=[0.5, 0.5, 1, 0.5],
            physics_client_id=self._physics_client_id
        )

        # Check
        s2 = self._get_state()
        if not s2.allclose(state):
            logging.warning("Could not reconstruct state exactly!")

    def step(self, action: Action, render_obs: bool = False) -> State:
        next_state = super().step(action, render_obs=render_obs)
        changed = self._update_water_level_if_needed(next_state)
        # Only re-draw if changed
        if changed:
            self._create_or_update_water(force_redraw=True)
        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

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
        If so, update water displacement and recalc water level.
        Returns True if the water level changed, else False.
        """
        old_height = self._current_water_height
        # Start from total volume = 2 compartments * old_height * area
        old_volume = 2.0 * self.CONTAINER_AREA * old_height

        new_displaced_volume = 0.0
        # Track which blocks are displacing at end
        blocks_displacing_now = {}

        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel.id, physicsClientId=self._physics_client_id)

        for blk in self._block_objs:
            # Get block's top (we approximate block top as its center + half)
            bx = state.get(blk, "x")
            by = state.get(blk, "y")
            bz = state.get(blk, "z")
            top_z = bz + (self.block_size / 2.0)

            # Are we inside a compartment XY?
            in_left = self._is_in_left_compartment(bx, by)
            in_right = self._is_in_right_compartment(bx, by)
            if not (in_left or in_right):
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
                new_displaced_volume += (self.block_size ** 3)

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
            # => The number of submerging blocks changed => recalc water
            new_total_volume = 2.0 * self.CONTAINER_AREA * old_height \
                               - (old_num_displacing * (self.block_size**3)) \
                               + (new_num_displacing * (self.block_size**3))
            # ^ This approach: 
            #   2 * area * old_height is the water volume w/o blocks
            #   old_num_displacing * block_volume => old block total
            #   remove old block volumes, add new block volumes
            # Actually let's do it more directly:
            #   old_volume was water + old displaced. We want to remove old displaced 
            #   and add new displaced. So new_volume = old_volume 
            #                                  - old_displaced + new_displaced
            old_displaced_vol = old_num_displacing * (self.block_size**3)
            new_displaced_vol = new_num_displacing * (self.block_size**3)
            new_volume = (old_volume - old_displaced_vol) + new_displaced_vol

            # water + blocks => water in 2 compartments => new_height
            new_height = new_volume / (2.0 * self.CONTAINER_AREA)
            new_height = np.clip(new_height, 0.0, self.z_ub_water)
            self._current_water_height = new_height
            changed = not np.isclose(new_height, old_height, atol=1e-5)

        # Update our record of who is displacing
        for blk in self._block_objs:
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
        lx = vx + self.CONTAINER_OPENING_LEN/2
        ly = vy
        left_id = create_water_body(
            size_z=self._current_water_height,
            size_x=self.CONTAINER_OPENING_LEN,
            size_y=self.CONTAINER_OPENING_LEN,
            base_position=(lx, ly, vz),
            physics_client_id=self._physics_client_id
        )
        self._water_ids["left"] = left_id

        # Right
        rx_offset = (self.CONTAINER_OPENING_LEN
                     + self.CONTAINER_GAP
                     + self.CONTAINER_OPENING_LEN/2)
        rx = vx + rx_offset
        ry = vy
        right_id = create_water_body(
            size_z=self._current_water_height,
            size_x=self.CONTAINER_OPENING_LEN,
            size_y=self.CONTAINER_OPENING_LEN,
            base_position=(rx, ry, vz),
            physics_client_id=self._physics_client_id
        )
        self._water_ids["right"] = right_id

    # -------------------------------------------------------------------------
    # Geometry checks
    def _is_in_left_compartment(self, bx: float, by: float) -> bool:
        (vx, vy, _) = self._get_vessel_base_position()
        x_min = vx
        x_max = vx + self.CONTAINER_OPENING_LEN
        y_min = vy
        y_max = vy + self.CONTAINER_OPENING_LEN
        return (x_min <= bx <= x_max) and (y_min <= by <= y_max)

    def _is_in_right_compartment(self, bx: float, by: float) -> bool:
        (vx, vy, _) = self._get_vessel_base_position()
        x_min = vx + self.CONTAINER_OPENING_LEN + self.CONTAINER_GAP
        x_max = x_min + self.CONTAINER_OPENING_LEN
        y_min = vy
        y_max = vy + self.CONTAINER_OPENING_LEN
        return (x_min <= bx <= x_max) and (y_min <= by <= y_max)

    def _get_vessel_base_position(self) -> Tuple[float, float, float]:
        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel.id, physicsClientId=self._physics_client_id)
        return (vx, vy, vz)

    # -------------------------------------------------------------------------
    # Predicates
    @staticmethod
    def _InWater_holds(state: State, objects: Sequence[Object]) -> bool:
        (block,) = objects
        return state.get(block, "in_water") > 0.5

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, block = objects
        return state.get(block, "is_held") > 0.5

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.2

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
            # Blocks random
            block_dicts = []
            # for _block in self._block_objs:
            #     bx = rng.uniform(self.x_lb, self.x_ub)
            #     by = rng.uniform(self.y_lb + 0.05, 
            #                     self.VESSEL_BASE_Y - self.CONTAINER_OPENING_LEN)
            #     bz = self.z_lb + self.block_size/2
            #     block_dicts.append({"x": bx, "y": by, "z": bz, 
            #                         "in_water": 0.0,
            #                         "is_held": 0.0})
            block_dicts = [
                {"x": 0.6, "y": 1.16 , "z": self.z_lb + self.block_size/2,
                    "in_water": 0.0, "is_held": 0.0},
                {"x": 0.8, "y": 1.2 , "z": self.z_lb + self.block_size/2,
                    "in_water": 0.0, "is_held": 0.0},
                {"x": 1, "y": 1.16, "z": self.z_lb + self.block_size/2,
                    "in_water": 0.0, "is_held": 0.0},
            ]

            init_dict = {self._robot: robot_dict, self._vessel: vessel_dict}
            for b_obj, b_vals in zip(self._block_objs, block_dicts):
                init_dict[b_obj] = b_vals

            init_state = utils.create_state_from_dict(init_dict)

            # e.g. goal: all blocks in water
            first_two_blocks = self._block_objs[:2]
            goal_atoms = {
                GroundAtom(self._InWater, [b]) for b in first_two_blocks
            }
            goal_atoms.add(
                GroundAtom(self._Holding, [self._robot, self._block_objs[2]])
            )
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