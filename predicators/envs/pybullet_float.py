"""
Example PyBulletFloatEnv where two containers are connected via a communicating
vessel. Dropping a block into either container raises water level in both.
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


# -----------------------------------------------------------------------------
# Utility for creating water bodies: as you provided
def create_water_body(size_z, size_x=0.2, size_y=0.2, base_position=(0, 0, 0)):
    """Create a semi-transparent water box in PyBullet."""
    water_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size_x / 2, size_y / 2, size_z / 2],
        rgbaColor=[0, 0, 1, 0.5]
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
        basePosition=base_position
    )
    return water_body_id


class PyBulletFloatEnv(PyBulletEnv):
    """A simple environment with two connected containers and a set of blocks.
    When a block is placed in either container, the water height in both
    containers increases according to total displaced volume / cross-sectional
    area. We represent the water in each container with a single box geometry.
    """

    # -------------------------------------------------------------------------
    # Workspace bounds (customize as needed)
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.3
    y_lb: ClassVar[float] = 0.5
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = 0.0
    z_ub: ClassVar[float] = 1.0
    init_padding: ClassVar[float] = 0.05

    # Container geometry / cross-sectional area
    container_opening: ClassVar[float] = 0.2  # each dimension => area = 0.04
    container_area: ClassVar[float] = container_opening * container_opening

    # Initial water heights (same in both containers)
    initial_water_height: ClassVar[float] = 0.1

    # Robot config (optional -- or remove if you don't want a robot)
    robot_init_x: ClassVar[float] = 0.8
    robot_init_y: ClassVar[float] = 0.7
    robot_init_z: ClassVar[float] = 0.2

    # Hard-coded block geometry (customize as desired)
    block_size: ClassVar[float] = 0.02  # side of a small cube
    block_mass: ClassVar[float] = 0.1

    # We store water IDs so we can remove + recreate them on every step if needed
    _water_ids: Dict[str, Optional[int]]  # "left" => int, "right" => int

    def __init__(self, use_gui: bool = True) -> None:
        # Define Types
        # Each container has a single water height feature
        self._container_type = Type("container", ["x", "y", "z", "water_height"])
        # Blocks have x,y,z positions (no orientation for simplicity),
        # plus a 'dropped' feature for whether it's in the water
        self._block_type = Type("block", ["x", "y", "z", "in_water"])

        # Create objects placeholders
        self._left_container = Object("left_container", self._container_type)
        self._right_container = Object("right_container", self._container_type)

        # We'll create multiple blocks
        self._block0 = Object("block0", self._block_type)
        self._block1 = Object("block1", self._block_type)

        # (optional) store them in a list for convenience
        self._block_objs = [self._block0, self._block1]

        super().__init__(use_gui)

        # Predicates (optional or define as needed)
        self._InWater = Predicate("InWater", [self._block_type],
                                  self._InWater_holds)

        # If you want the environment to track these for goals, etc.
        # This environment is just for demonstration.
        # So we won't define a specific "goal" predicate here.
        
        self._water_ids = {"left": None, "right": None}

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_float"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InWater}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """If you want a particular goal, define it here. For now, empty."""
        return set()

    @property
    def types(self) -> Set[Type]:
        return {self._container_type, self._block_type}

    # -------------------------------------------------------------------------
    # Initialization in PyBullet

    @classmethod
    def initialize_pybullet(
        cls, using_gui: bool
    ) -> Tuple[int, Any, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(
            using_gui)

        # For demonstration, create two open-topped container URDFs or just
        # boxes. (You could also do create_object with an .urdf if you prefer.)
        # We'll place them in front of the robot.
        # Alternatively, import a table, floor plane, etc. as needed.
        left_container_id = create_object(
            asset_path="urdf/box.urdf",  # or your custom open-top container
            position=(0.6, 1.1, 0.0),
            scale=0.4,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        right_container_id = create_object(
            asset_path="urdf/box.urdf",
            position=(1.0, 1.1, 0.0),
            scale=0.4,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["left_container_id"] = left_container_id
        bodies["right_container_id"] = right_container_id

        # Create a few blocks
        block_ids = []
        for _ in range(2):
            blk_id = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[cls.block_size / 2]*3,
                physicsClientId=physics_client_id
            )
            # Combine visual shape with the same half-extents
            visual_id = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[cls.block_size / 2]*3,
                rgbaColor=[1.0, 0.6, 0.0, 1.0],
                physicsClientId=physics_client_id
            )
            # Create multiBody
            block_body_id = p.createMultiBody(
                baseMass=cls.block_mass,
                baseCollisionShapeIndex=blk_id,
                baseVisualShapeIndex=visual_id,
                basePosition=(0.8, 0.8, 0.2),  # somewhere to start
                physicsClientId=physics_client_id
            )
            block_ids.append(block_body_id)
        bodies["block_ids"] = block_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to PyBullet IDs for environment assets."""
        self._left_container.id = pybullet_bodies["left_container_id"]
        self._right_container.id = pybullet_bodies["right_container_id"]
        self._block0.id = pybullet_bodies["block_ids"][0]
        self._block1.id = pybullet_bodies["block_ids"][1]

    # -------------------------------------------------------------------------
    # State Management: Get, Set, Step

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return IDs of jugs (since we can only hold jugs)."""
        return []

    def _get_state(self) -> State:
        """Create a State object from PyBullet simulation."""
        state_dict: Dict[Object, Dict[str, float]] = {}

        # Containers
        for container_obj in [self._left_container, self._right_container]:
            cid = container_obj.id
            (cx, cy, cz), _ = p.getBasePositionAndOrientation(
                cid, physicsClientId=self._physics_client_id
            )
            # We'll store the water height as a feature
            # We'll keep it in memory from our environment or from the container
            # For the sake of demonstration, let's read from the current state
            # or a placeholder. We'll store it under "water_height".
            # We'll retrieve it from our environment if we want a single source
            # of truth. For now, let's do the simpler approach of storing a
            # feature we can update as needed.
            if container_obj is self._left_container:
                wh = self._current_left_water_height
            else:
                wh = self._current_right_water_height

            state_dict[container_obj] = {
                "x": cx,
                "y": cy,
                "z": cz,
                "water_height": wh
            }

        # Blocks
        for blk in self._block_objs:
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                blk.id, physicsClientId=self._physics_client_id
            )
            # in_water = 1.0 if the block is partially submerged
            # We'll define a simple check: if the block's z < water surface
            # in either container bounding region, we say it's 'in_water'.
            # This is obviously a big simplification.
            in_water_val = 0.0
            if self._is_block_in_left_container(bx, by):
                if bz < self._current_left_water_height:
                    in_water_val = 1.0
            elif self._is_block_in_right_container(bx, by):
                if bz < self._current_right_water_height:
                    in_water_val = 1.0

            state_dict[blk] = {
                "x": bx,
                "y": by,
                "z": bz,
                "in_water": in_water_val
            }

        return utils.create_state_from_dict(state_dict)

    def _reset_state(self, state: State) -> None:
        """Reset PyBullet to the given state. Also reconstruct water geometry."""
        super()._reset_state(state)  # clear constraints, reset robot if present

        # Reconstruct the container positions and water heights
        for c_obj in [self._left_container, self._right_container]:
            cx = state.get(c_obj, "x")
            cy = state.get(c_obj, "y")
            cz = state.get(c_obj, "z")
            update_object(c_obj.id,
                          position=(cx, cy, cz),
                          physics_client_id=self._physics_client_id)

        self._current_left_water_height = state.get(self._left_container,
                                                    "water_height")
        self._current_right_water_height = state.get(self._right_container,
                                                     "water_height")

        # Recreate water bodies (remove old ones if any)
        for key in self._water_ids:
            wid = self._water_ids[key]
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
            self._water_ids[key] = None

        self._create_or_update_water_bodies()

        # Recreate blocks
        for blk in self._block_objs:
            bx = state.get(blk, "x")
            by = state.get(blk, "y")
            bz = state.get(blk, "z")
            update_object(blk.id,
                          position=(bx, by, bz),
                          physics_client_id=self._physics_client_id)

        # done
        reconstructed = self._get_state()
        if not reconstructed.allclose(state):
            logging.warning("Could not reconstruct state exactly!")

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Advance the simulation one step. Then check for water displacement."""
        next_state = super().step(action, render_obs=render_obs)
        # After PyBullet moves blocks, re-check if blocks are in water, and
        # update water heights accordingly.
        self._update_water_levels(next_state)
        # Re-draw water bodies
        self._create_or_update_water_bodies()
        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Water-level logic

    def _update_water_levels(self, state: State) -> None:
        """Compute new water levels in both containers based on blocks."""
        # Start from base volume (area * initial_heights)
        # or from the current water heights in the state.
        left_h = state.get(self._left_container, "water_height")
        right_h = state.get(self._right_container, "water_height")

        # Combined water volume in the entire system
        # (In a truly connected vessel, the water level is the same in both
        # containers, so we combine volume, then compute new level, then
        # set that level in both containers.)
        total_initial_volume = (left_h + right_h) * (self.container_area / 2.0)
        # ^ If you prefer each container to have container_area = 0.04,
        # then total cross-section is 0.08. Or treat them individually
        # but ensure final heights are same. We'll do a simple approach:
        #   total_initial_volume = left_h*0.04 + right_h*0.04
        # so let's do that:
        total_initial_volume = left_h * self.container_area \
                             + right_h * self.container_area

        # Check each block's submersion volume
        # We'll use a crude check: if block is in either container and
        # below water surface, assume entire block volume is displaced.
        # You can refine for partial submersion, etc.
        block_volume = self.block_size**3

        total_displaced = 0.0
        for blk in self._block_objs:
            in_water = state.get(blk, "in_water")
            if in_water > 0.5:
                total_displaced += block_volume

        # The new total volume = water + displaced
        new_total_volume = total_initial_volume + total_displaced
        # Recompute the new uniform water height (since connected)
        # container_area is for each container. If we assume the same cross-
        # sectional area for left and right, total effective cross-section
        # is 2 * container_area. Then water in each container is the same
        # level, so total volume = new_height * 2 * container_area
        new_height = new_total_volume / (2.0 * self.container_area)

        # Make sure we don't go below some minimal or above some max
        new_height = max(new_height, 0.0)
        new_height = min(new_height, self.z_ub)

        # Update state with new water height for both containers
        state.set(self._left_container, "water_height", new_height)
        state.set(self._right_container, "water_height", new_height)

        self._current_left_water_height = new_height
        self._current_right_water_height = new_height

    def _create_or_update_water_bodies(self) -> None:
        """Create or update the water geom boxes in PyBullet for each container."""
        # Remove old water bodies if they exist, then recreate
        for key, wid in self._water_ids.items():
            if wid is not None:
                p.removeBody(wid, physicsClientId=self._physics_client_id)
                self._water_ids[key] = None

        # Recreate from scratch for left container
        lx, ly, _ = p.getBasePositionAndOrientation(
            self._left_container.id, physicsClientId=self._physics_client_id
        )[0]
        left_water_h = self._current_left_water_height
        # Because the container has cross-section 0.2 x 0.2, we can call
        # create_water_body with size_z=left_water_h, etc.
        if left_water_h > 0:
            self._water_ids["left"] = create_water_body(
                size_z=left_water_h,
                size_x=self.container_opening,
                size_y=self.container_opening,
                base_position=(lx, ly, 0.0)
            )

        # Recreate for right container
        rx, ry, _ = p.getBasePositionAndOrientation(
            self._right_container.id, physicsClientId=self._physics_client_id
        )[0]
        right_water_h = self._current_right_water_height
        if right_water_h > 0:
            self._water_ids["right"] = create_water_body(
                size_z=right_water_h,
                size_x=self.container_opening,
                size_y=self.container_opening,
                base_position=(rx, ry, 0.0)
            )

    # -------------------------------------------------------------------------
    # Predicates
    @staticmethod
    def _InWater_holds(state: State, objects: Sequence[Object]) -> bool:
        """Check if block is marked as in water."""
        (block,) = objects
        return state.get(block, "in_water") > 0.5

    # -------------------------------------------------------------------------
    # Helpers
    def _is_block_in_left_container(self, bx: float, by: float) -> bool:
        """Simple bounding check for left container region in XY plane."""
        # Adjust these to match the actual container geometry
        # Here, we assume the container is near (0.6, 1.1).
        container_center = (0.6, 1.1)
        half = 0.2  # bounding region for a container of scale=0.4
        if (abs(bx - container_center[0]) <= half and
                abs(by - container_center[1]) <= half):
            return True
        return False

    def _is_block_in_right_container(self, bx: float, by: float) -> bool:
        """Simple bounding check for right container region in XY plane."""
        container_center = (1.0, 1.1)
        half = 0.2
        if (abs(bx - container_center[0]) <= half and
                abs(by - container_center[1]) <= half):
            return True
        return False

    # -------------------------------------------------------------------------
    # Task Generation (optional)
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
            # Example: place containers in default positions, blocks random
            left_container_dict = {
                "x": 0.6,
                "y": 1.1,
                "z": 0.0,
                "water_height": self.initial_water_height
            }
            right_container_dict = {
                "x": 1.0,
                "y": 1.1,
                "z": 0.0,
                "water_height": self.initial_water_height
            }
            # Random block positions
            bx0 = rng.uniform(self.x_lb, self.x_ub)
            by0 = rng.uniform(self.y_lb, self.y_ub)
            bz0 = 0.15
            bx1 = rng.uniform(self.x_lb, self.x_ub)
            by1 = rng.uniform(self.y_lb, self.y_ub)
            bz1 = 0.15

            block0_dict = {
                "x": bx0,
                "y": by0,
                "z": bz0,
                "in_water": 0.0,
            }
            block1_dict = {
                "x": bx1,
                "y": by1,
                "z": bz1,
                "in_water": 0.0,
            }

            init_dict = {
                self._left_container: left_container_dict,
                self._right_container: right_container_dict,
                self._block0: block0_dict,
                self._block1: block1_dict,
            }
            init_state = utils.create_state_from_dict(init_dict)
            # You can define a goal if you want. Example: both blocks in water.
            goal_atoms = {
                GroundAtom(self._InWater, [self._block0]),
                GroundAtom(self._InWater, [self._block1]),
            }
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)

if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 0
    env = PyBulletFloatEnv(use_gui=True)
    task = env._make_tasks(1, CFG.seed)[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))

        env.step(action)
        time.sleep(0.01)
