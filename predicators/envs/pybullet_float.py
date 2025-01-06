"""
Single-object Communicating Vessel Example.

Here, we define one `vessel` object with features: [x, y, z, water_height].
Internally, we treat two compartments, but the single water_height applies to
both since the fluid is shared.

Blocks have [x, y, z, in_water]. 
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


def create_water_body(size_z, size_x=0.2, size_y=0.2, base_position=(0, 0, 0),
                      physics_client_id=None):
    """Create a semi-transparent 'water' box in PyBullet."""
    water_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size_x / 2, size_y / 2, size_z / 2],
        rgbaColor=[0, 0, 1, 0.5],
        physicsClientId=physics_client_id
    )
    base_position = [
        base_position[0],
        base_position[1],
        base_position[2] + size_z / 2  # shift up
    ]
    water_body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=water_visual,
        basePosition=base_position,
        physicsClientId=physics_client_id
    )
    return water_body_id


class PyBulletSingleVesselEnv(PyBulletEnv):
    """Communicating vessel environment with a single 'vessel' object (plus blocks).
    The vessel has x, y, z, water_height. Internally, we treat two compartments
    but share a single water_height because the fluid is connected.
    """

    # -------------------------------------------------------------------------
    # Vessel geometry / URDF config
    COMM_VESSEL_URDF: ClassVar[str] = "urdf/comm_vessel2.urdf"
    CONTAINER_OPENING_LEN: ClassVar[float] = 0.1  # each compartment is 0.2x0.2
    CONTAINER_GAP: ClassVar[float] = 0.3
    VESSEL_WALL_THICKNESS: ClassVar[float] = 0.01

    # One shared cross-sectional area for each compartment
    # => total area is 2 * CONTAINER_AREA
    CONTAINER_AREA: ClassVar[float] = CONTAINER_OPENING_LEN**2

    # -------------------------------------------------------------------------
    # Table / workspace config
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = 0.2
    z_ub: ClassVar[float] = 0.75

    # Table pose
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, 0.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler([0., 0., np.pi/2])

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

    # We'll place the entire vessel near (0.75, 1.3, 0.2)
    # The user can adjust these as needed
    VESSEL_BASE_X: ClassVar[float] = 0.55
    VESSEL_BASE_Y: ClassVar[float] = 1.3

    # Water config
    initial_water_height: ClassVar[float] = 0.1
    z_ub_water: ClassVar[float] = 0.5

    # Blocks
    block_size: ClassVar[float] = 0.05
    block_mass: ClassVar[float] = 0.1

    # Single vessel object has features: [x, y, z, water_height]
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _vessel_type = Type("vessel", ["x", "y", "z", "water_height"])
    _block_type = Type("block", ["x", "y", "z", "in_water"])

    # We'll keep references to the actual PyBullet bodies
    _water_ids: Dict[str, Optional[int]]  # e.g. {"left": ID, "right": ID}

    def __init__(self, use_gui: bool = True) -> None:

        # Define environment objects
        self._robot = Object("robot", self._robot_type)
        # A single vessel object
        self._vessel = Object("vessel", self._vessel_type)
        # We'll create a couple of blocks
        self._block0 = Object("block0", self._block_type)
        self._block1 = Object("block1", self._block_type)
        self._block_objs = [self._block0, self._block1]

        # For references to PyBullet
        self._vessel_id: Optional[int] = None

        super().__init__(use_gui)

        # Example predicate
        self._InWater = Predicate("InWater", [self._block_type],
                                  self._InWater_holds)

        # Track water box IDs for left/right compartments
        self._water_ids = {"left": None, "right": None}

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_single_vessel"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InWater}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return set()

    @property
    def types(self) -> Set[Type]:
        return {self._vessel_type, self._block_type}

    # -------------------------------------------------------------------------
    # PyBullet Setup

    @classmethod
    def initialize_pybullet(cls, using_gui: bool
                            ) -> Tuple[int, Any, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui)

        # Create table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id
        )
        bodies["table_id"] = table_id

        # Create vessel URDF
        vessel_id = create_object(
            asset_path=cls.COMM_VESSEL_URDF,
            position=(cls.VESSEL_BASE_X, cls.VESSEL_BASE_Y, cls.z_lb),
            use_fixed_base=True,
            scale=1.0,
            physics_client_id=physics_client_id
        )
        bodies["vessel_id"] = vessel_id

        # Create blocks
        block_ids = []
        for i in range(2):
            collision_id = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[cls.block_size/2]*3,
                physicsClientId=physics_client_id
            )
            visual_id = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[cls.block_size/2]*3,
                rgbaColor=[1, 0.6, 0, 1],
                physicsClientId=physics_client_id
            )
            init_z = cls.z_lb + cls.block_size / 2
            body_id = p.createMultiBody(
                baseMass=cls.block_mass,
                baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id,
                basePosition=(0.8, 1.3 + 0.05*i, init_z),
                physicsClientId=physics_client_id
            )
            block_ids.append(body_id)
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._vessel_id = pybullet_bodies["vessel_id"]
        self._block0.id = pybullet_bodies["block_ids"][0]
        self._block1.id = pybullet_bodies["block_ids"][1]

    # -------------------------------------------------------------------------
    # State Management

    def _get_object_ids_for_held_check(self) -> List[int]:
        # For picking logic, if needed
        return []

    def _get_state(self) -> State:
        """Return current environment State."""
        state_dict: Dict[Object, Dict[str, float]] = {}

        # 1) Robot
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
        _, tilt, wrist = p.getEulerFromQuaternion([qx, qy, qz, qw])
        state_dict[self._robot] = {
            "x": rx,
            "y": ry,
            "z": rz,
            "fingers": self._fingers_joint_to_state(self._pybullet_robot, rf),
            "tilt": tilt,
            "wrist": wrist,
        }
        # Not needed as a separate object unless you want a "robot" object 
        # in your domain. If you do, define a self._robot = Object(...).
        # For brevity, skip it or define it.

        # 2) Vessel
        # We'll read the base pose from PyBullet
        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel_id, physicsClientId=self._physics_client_id)
        vessel_dict = {
            "x": vx,
            "y": vy,
            "z": vz,
            "water_height": self._current_water_height
        }
        state_dict[self._vessel] = vessel_dict

        # 3) Blocks
        for blk in self._block_objs:
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                blk.id, physicsClientId=self._physics_client_id)
            in_water_val = 0.0
            # If block is within left or right sub-compartment bounding region
            # and below water top => in_water=1.0
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

        state = utils.create_state_from_dict(state_dict)
        joint_positions = self._pybullet_robot.get_joints()
        return utils.PyBulletState(
            state.data, simulator_state={"joint_positions": joint_positions})

    def _reset_state(self, state: State) -> None:
        """Reset environment from State."""
        super()._reset_state(state)

        # 1) Retrieve vessel water height
        self._current_water_height = state.get(self._vessel, "water_height")
        # Clear old water visuals
        for key, wid in self._water_ids.items():
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
                self._water_ids[key] = None

        # 2) Reposition blocks
        for blk in self._block_objs:
            bx = state.get(blk, "x")
            by = state.get(blk, "y")
            bz = state.get(blk, "z")
            update_object(blk.id,
                          position=(bx, by, bz),
                          physics_client_id=self._physics_client_id)

        # 3) Redraw water
        self._create_or_update_water()

        # Check reconstruction
        s2 = self._get_state()
        if not s2.allclose(state):
            logging.warning("Could not reconstruct state exactly!")

    def step(self, action: Action, render_obs: bool = False) -> State:
        next_state = super().step(action, render_obs=render_obs)
        self._update_water_level(next_state)
        self._create_or_update_water()
        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Water Logic

    @property
    def _current_water_height(self) -> float:
        """Helper property for the vessel's water level."""
        return getattr(self, "__current_water_height", 0.0)

    @_current_water_height.setter
    def _current_water_height(self, val: float) -> None:
        setattr(self, "__current_water_height", val)

    def _update_water_level(self, state: State) -> None:
        """Combine volumes in both compartments, use one water_height."""
        old_height = state.get(self._vessel, "water_height")
        # volume = height * cross_section(for each compartment) => total of 2 compartments
        initial_volume = 2.0 * self.CONTAINER_AREA * old_height

        # Displacement from blocks
        block_vol = self.block_size**3
        total_displaced = 0.0
        for blk in self._block_objs:
            if state.get(blk, "in_water") > 0.5:
                total_displaced += block_vol

        new_volume = initial_volume + total_displaced
        # new_height = new_volume / (area_of_two_compartments)
        new_height = new_volume / (2.0 * self.CONTAINER_AREA)
        new_height = max(0.0, min(new_height, self.z_ub_water))

        # update the vessel water_height in the State
        state.set(self._vessel, "water_height", new_height)
        self._current_water_height = new_height

    def _create_or_update_water(self) -> None:
        """Draw water boxes for left, right compartments at same water_height."""
        for side, wid in self._water_ids.items():
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
                self._water_ids[side] = None

        # If water height is 0, do nothing
        if self._current_water_height <= 0:
            return

        # We'll retrieve the vessel base from PyBullet in case it moved:
        (vx, vy, vz), _ = p.getBasePositionAndOrientation(
            self._vessel_id, physicsClientId=self._physics_client_id)

        # Left water
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

        # Right water
        rx_offset = self.CONTAINER_OPENING_LEN + self.CONTAINER_GAP + (self.CONTAINER_OPENING_LEN/2)
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
    # Bounding Checks (Python "properties" or methods)
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
            self._vessel_id, physicsClientId=self._physics_client_id)
        return (vx, vy, vz)

    # -------------------------------------------------------------------------
    # Predicates

    @staticmethod
    def _InWater_holds(state: State, objects: Sequence[Object]) -> bool:
        (block,) = objects
        return state.get(block, "in_water") > 0.5

    # -------------------------------------------------------------------------
    # Task Generation Example
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
            # Robot at center
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
            # Blocks randomly placed on table
            block_dicts = []
            for _block in self._block_objs:
                bx = rng.uniform(self.x_lb, self.x_ub)
                by = rng.uniform(self.y_lb, self.y_ub)
                bz = self.z_lb + self.block_size/2
                block_dicts.append({"x": bx, "y": by, "z": bz, "in_water": 0.0})

            # Combine into init_dict
            init_dict = {
                self._robot: robot_dict,
                self._vessel: vessel_dict
            }
            for b_obj, b_vals in zip(self._block_objs, block_dicts):
                init_dict[b_obj] = b_vals

            init_state = utils.create_state_from_dict(init_dict)

            # e.g. goal is all blocks in water
            goal_atoms = {
                GroundAtom(self._InWater, [b]) for b in self._block_objs
            }
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    import time

    CFG.seed = 0
    CFG.pybullet_sim_steps_per_action = 1

    env = PyBulletSingleVesselEnv(use_gui=True)
    task = env._make_tasks(1, np.random.default_rng(0))[0]
    env._reset_state(task.init)

    for _ in range(int(1000000)):
        # robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))
        env.step(action)
        time.sleep(0.01)