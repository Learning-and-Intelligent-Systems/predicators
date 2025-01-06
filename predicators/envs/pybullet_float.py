"""
Example PyBulletFloatEnv using comm_vessel2.urdf for the communicating vessel.
This URDF has two containers, each with a 0.2x0.2 opening, walls of thickness
0.01, and a 0.1 x-gap between them. The origin is at the middle of the bottom
left edge of the vessel.
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
    """Create a semi-transparent water box in PyBullet."""
    water_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size_x / 2, size_y / 2, size_z / 2],
        rgbaColor=[0, 0, 1, 0.5],
        physicsClientId=physics_client_id
    )
    # Shift up by half the height so it's "resting" at base_position in z=0
    base_position = [
        base_position[0],
        base_position[1],
        base_position[2] + size_z / 2
    ]
    water_body_id = p.createMultiBody(
        baseMass=0,
        baseVisualShapeIndex=water_visual,
        basePosition=base_position,
        physicsClientId=physics_client_id
    )
    return water_body_id


class PyBulletFloatEnv(PyBulletEnv):
    """Environment with a single URDF model (comm_vessel2.urdf) containing two
    connected containers. Blocks can be dropped in either container, which
    raises the water height in both due to the communicating vessel principle.
    """

    # -------------------------------------------------------------------------
    # Vessel geometry / URDF config
    COMM_VESSEL_URDF: ClassVar[str] = "urdf/comm_vessel2.urdf"
    VESSEL_WALL_THICKNESS: ClassVar[float] = 0.01
    CONTAINER_OPENING_LEN: ClassVar[float] = 0.2
    CONTAINER_GAP: ClassVar[float] = 0.1
    # The origin is at the middle of the bottom-left edge (in the URDF frame).
    # Adjust these if you want to place the vessel differently in the world.
    VESSEL_BASE_X: ClassVar[float] = 0.7
    VESSEL_BASE_Y: ClassVar[float] = 1.0

    # We will treat each container as having cross-sectional area = 0.2*0.2 = 0.04
    CONTAINER_AREA: ClassVar[float] = CONTAINER_OPENING_LEN**2

    # The distance between left and right container 'openings' along X
    # is CONTAINER_OPENING_LEN + CONTAINER_GAP + CONTAINER_OPENING_LEN
    # (from the left container's right edge to the right container's left edge).
    # If you measure differently in your URDF, adjust logic below.

    # -------------------------------------------------------------------------
    # Water config
    initial_water_height: ClassVar[float] = 0.1
    z_ub: ClassVar[float] = 1.0  # some max height to clamp water

    # -------------------------------------------------------------------------
    # Robot / block config (optional, or adjust as needed)
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.3
    y_lb: ClassVar[float] = 0.5
    y_ub: ClassVar[float] = 1.6

    # Table config
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, 0.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0., 0., np.pi / 2])

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

    block_size: ClassVar[float] = 0.02  # side of a small cube
    block_mass: ClassVar[float] = 0.1

    # We'll store references to water bodies so we can remove them each step
    _water_ids: Dict[str, Optional[int]]  # "left" => int, "right" => int

    # Define environment "Types"
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _container_type = Type("container", ["water_height"])
    _block_type = Type("block", ["x", "y", "z", "in_water"])

    def __init__(self, use_gui: bool = True) -> None:

        self._robot = Object("robot", self._robot_type)

        # We'll conceptualize two container objects for tracking water levels
        # even though they're part of a single URDF in PyBullet.
        self._left_container = Object("left_container", self._container_type)
        self._right_container = Object("right_container", self._container_type)

        # Vessel as a single PyBullet body, but no typed features needed
        # if you don't plan on referencing it in the state. We'll store it
        # in _vessel_id after creation.
        self._vessel_id: Optional[int] = None

        # We'll create multiple blocks
        self._block0 = Object("block0", self._block_type)
        self._block1 = Object("block1", self._block_type)
        self._block_objs = [self._block0, self._block1]

        super().__init__(use_gui)

        # Example predicate: a block is in water
        self._InWater = Predicate("InWater", [self._block_type],
                                  self._InWater_holds)

        self._water_ids = {"left": None, "right": None}

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_float"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InWater}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """No specific goals for demonstration."""
        return set()

    @property
    def types(self) -> Set[Type]:
        return {self._container_type, self._block_type}

    # -------------------------------------------------------------------------
    # PyBullet Setup

    @classmethod
    def initialize_pybullet(
        cls, using_gui: bool
    ) -> Tuple[int, Any, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui
        )

        # Add table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # Create the single URDF for the communicating vessel
        vessel_id = create_object(
            asset_path=cls.COMM_VESSEL_URDF,
            position=(cls.VESSEL_BASE_X, cls.VESSEL_BASE_Y, 0.0),
            use_fixed_base=True,
            physics_client_id=physics_client_id
        )
        bodies["vessel_id"] = vessel_id

        # Create blocks
        block_ids = []
        for _ in range(2):
            blk_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[cls.block_size / 2]*3,
                physicsClientId=physics_client_id
            )
            visual_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[cls.block_size / 2]*3,
                rgbaColor=[1.0, 0.6, 0.0, 1.0],
                physicsClientId=physics_client_id
            )
            block_body_id = p.createMultiBody(
                baseMass=cls.block_mass,
                baseCollisionShapeIndex=blk_id,
                baseVisualShapeIndex=visual_id,
                basePosition=(0.8, 0.8, 0.2),  # initial block position
                physicsClientId=physics_client_id
            )
            block_ids.append(block_body_id)
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet IDs for environment assets."""
        self._vessel_id = pybullet_bodies["vessel_id"]
        self._block0.id = pybullet_bodies["block_ids"][0]
        self._block1.id = pybullet_bodies["block_ids"][1]

    # -------------------------------------------------------------------------
    # State Management
    def _get_object_ids_for_held_check(self) -> List[int]:
        return []

    def _get_state(self) -> State:
        """Create a State object from PyBullet simulation."""
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
            "wrist": wrist,
        }

        # For each container object, store the water_height feature.
        # We'll keep these in internal variables updated at each step.
        # (Or you can store them in the state directly; here, we do both.)
        state_dict[self._left_container] = {
            "water_height": self._current_left_water_height
        }
        state_dict[self._right_container] = {
            "water_height": self._current_right_water_height
        }

        # Blocks
        for blk in self._block_objs:
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                blk.id, physicsClientId=self._physics_client_id
            )
            in_water_val = 0.0
            # We'll do a simplified bounding check: if the block is in the left
            # container region and below the left water surface => in_water=1.0
            # Similarly for the right container region.
            if self._is_in_left_container(bx, by):
                if bz < self._current_left_water_height:
                    in_water_val = 1.0
            elif self._is_in_right_container(bx, by):
                if bz < self._current_right_water_height:
                    in_water_val = 1.0

            state_dict[blk] = {
                "x": bx,
                "y": by,
                "z": bz,
                "in_water": in_water_val
            }

        state = utils.create_state_from_dict(state_dict)
        joint_positions = self._pybullet_robot.get_joints()
        pyb_state = utils.PyBulletState(
            state.data, simulator_state={"joint_positions": joint_positions})
        return pyb_state

    def _reset_state(self, state: State) -> None:
        """Reset PyBullet to the given State. Also reconstruct water boxes."""
        super()._reset_state(state)

        # Reinitialize the internal container water heights
        self._current_left_water_height = state.get(self._left_container,
                                                    "water_height")
        self._current_right_water_height = state.get(self._right_container,
                                                     "water_height")

        # Remove old water bodies
        for key in self._water_ids:
            wid = self._water_ids[key]
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
                self._water_ids[key] = None

        # Recreate blocks
        for blk in self._block_objs:
            bx = state.get(blk, "x")
            by = state.get(blk, "y")
            bz = state.get(blk, "z")
            update_object(blk.id,
                          position=(bx, by, bz),
                          physics_client_id=self._physics_client_id)

        # Now draw new water boxes
        self._create_or_update_water_bodies()

        # Sanity-check
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.warning("Could not reconstruct state exactly!")

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Simulate one step, then recompute water heights and re-draw water."""
        next_state = super().step(action, render_obs=render_obs)

        # Update water height from displacement
        self._update_water_levels(next_state)
        # Recreate water visuals
        self._create_or_update_water_bodies()

        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Water & Displacement Logic

    def _update_water_levels(self, state: State) -> None:
        """Compute new water heights from total displacement."""
        left_h = state.get(self._left_container, "water_height")
        right_h = state.get(self._right_container, "water_height")

        # Combine volumes from both containers:
        # total_volume = sum of (height * cross_section)
        # cross_section = CONTAINER_AREA = 0.2 * 0.2 = 0.04
        total_initial_volume = left_h * self.CONTAINER_AREA + \
                               right_h * self.CONTAINER_AREA

        # Block displacement
        block_volume = self.block_size**3
        total_displaced = 0.0
        for blk in self._block_objs:
            if state.get(blk, "in_water") > 0.5:
                # If "in_water" is 1, we assume entire block volume is displaced
                # (Simplification.)
                total_displaced += block_volume

        new_total_volume = total_initial_volume + total_displaced

        # The two compartments share the same final water level.
        # Combined area is 2 * CONTAINER_AREA
        new_height = new_total_volume / (2.0 * self.CONTAINER_AREA)
        new_height = max(0.0, min(new_height, self.z_ub))

        # Update the environment's water heights for both containers
        state.set(self._left_container, "water_height", new_height)
        state.set(self._right_container, "water_height", new_height)

        self._current_left_water_height = new_height
        self._current_right_water_height = new_height

    def _create_or_update_water_bodies(self) -> None:
        """Re-create the water boxes in PyBullet for each container."""
        # Remove old first
        for key, wid in self._water_ids.items():
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
                self._water_ids[key] = None

        # For demonstration, we place each water box based on
        # the vessel base plus offsets. In your URDF, measure precisely
        # where left vs. right container openings are.
        # Suppose the left container spans x in [0, 0.2], the right in [0.3, 0.5]
        # from the vessel's local origin. We add these to VESSEL_BASE_X.
        
        # Left water
        if self._current_left_water_height > 0:
            left_x = self.VESSEL_BASE_X + (self.CONTAINER_OPENING_LEN/2.0)
            left_y = self.VESSEL_BASE_Y + (self.CONTAINER_OPENING_LEN/2.0)
            self._water_ids["left"] = create_water_body(
                size_z=self._current_left_water_height,
                size_x=self.CONTAINER_OPENING_LEN,
                size_y=self.CONTAINER_OPENING_LEN,
                base_position=(left_x, left_y, 0.0),
                physics_client_id=self._physics_client_id
            )

        # Right water
        if self._current_right_water_height > 0:
            # The right container starts at x=0.2 + 0.1 gap => 0.3
            # so center is 0.4 if the opening is 0.2 wide
            right_x_offset = self.CONTAINER_OPENING_LEN + self.CONTAINER_GAP \
                             + self.CONTAINER_OPENING_LEN/2.0
            right_x = self.VESSEL_BASE_X + right_x_offset
            right_y = self.VESSEL_BASE_Y + (self.CONTAINER_OPENING_LEN/2.0)
            self._water_ids["right"] = create_water_body(
                size_z=self._current_right_water_height,
                size_x=self.CONTAINER_OPENING_LEN,
                size_y=self.CONTAINER_OPENING_LEN,
                base_position=(right_x, right_y, 0.0),
                physics_client_id=self._physics_client_id
            )

    # -------------------------------------------------------------------------
    # Predicates

    @staticmethod
    def _InWater_holds(state: State, objects: Sequence[Object]) -> bool:
        """Returns True if the block is flagged as 'in_water' > 0.5."""
        (block,) = objects
        return state.get(block, "in_water") > 0.5

    # -------------------------------------------------------------------------
    # Helpers

    def _is_in_left_container(self, bx: float, by: float) -> bool:
        """Crude bounding region for left container in XY, offset from VESSEL_BASE."""
        # If the vessel's local X-range for left container is [0, 0.2],
        # then the global range is [VESSEL_BASE_X, VESSEL_BASE_X+0.2].
        # Similarly for Y range. Adjust as needed.
        x_min = self.VESSEL_BASE_X
        x_max = x_min + self.CONTAINER_OPENING_LEN
        y_min = self.VESSEL_BASE_Y
        y_max = y_min + self.CONTAINER_OPENING_LEN
        return (x_min <= bx <= x_max) and (y_min <= by <= y_max)

    def _is_in_right_container(self, bx: float, by: float) -> bool:
        """Crude bounding region for right container in XY."""
        x_min = self.VESSEL_BASE_X + self.CONTAINER_OPENING_LEN + self.CONTAINER_GAP
        x_max = x_min + self.CONTAINER_OPENING_LEN
        y_min = self.VESSEL_BASE_Y
        y_max = y_min + self.CONTAINER_OPENING_LEN
        return (x_min <= bx <= x_max) and (y_min <= by <= y_max)

    # -------------------------------------------------------------------------
    # Task Generation (optional demo)

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

            # Start water in both containers
            left_container_dict = {"water_height": self.initial_water_height}
            right_container_dict = {"water_height": self.initial_water_height}

            # Place blocks randomly
            bdicts = []
            for _block_obj in self._block_objs:
                bx = np.random.uniform(self.x_lb, self.x_ub)
                by = np.random.uniform(self.y_lb, self.y_ub)
                bz = 0.15
                bdicts.append({"x": bx, "y": by, "z": bz, "in_water": 0.0})

            init_dict = {
                self._left_container: left_container_dict,
                self._right_container: right_container_dict,
                self._robot: robot_dict,
            }
            for block_obj, block_vals in zip(self._block_objs, bdicts):
                init_dict[block_obj] = block_vals

            init_state = utils.create_state_from_dict(init_dict)
            # Suppose we want both blocks in water
            goal_atoms = {
                GroundAtom(self._InWater, [b]) for b in self._block_objs
            }
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)

if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 0
    CFG.pybullet_sim_steps_per_action = 1
    env = PyBulletFloatEnv(use_gui=True)
    task = env._make_tasks(1, CFG.seed)[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))

        env.step(action)
        time.sleep(0.01)
