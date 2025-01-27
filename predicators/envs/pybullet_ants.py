from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.objects import (create_object, update_object,
                                            sample_collision_free_2d_positions)
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, \
    State, Type, GroundAtom


class PyBulletAntsEnv(PyBulletEnv):
    """A PyBullet environment with:

    - A single robot.
    - Multiple food blocks of varying colors (some 'attractive').
    - Several ant objects that move toward an attractive food with noise.
    """

    # -------------------------------------------------------------------------
    # Table / workspace config
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Tuple[float, float,
                              float]] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Tuple[float, float, float,
                              float]] = p.getQuaternionFromEuler(
                                  [0., 0., np.pi / 2])

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    one_third_x: ClassVar[float] = x_lb + (x_ub - x_lb) / 3
    two_third_x: ClassVar[float] = x_lb + 2 * (x_ub - x_lb) / 3
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    padding: ClassVar[float] = 0.01

    # Robot init
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub
    robot_base_pos: ClassVar[Tuple[float, float, float]] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Tuple[float, float, float, float]] =\
        p.getQuaternionFromEuler([0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2
    held_tol: ClassVar[float] = 0.5

    # Camera
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # Define how many ants and how many food blocks
    num_ants: ClassVar[int] = 6
    num_food: ClassVar[int] = 6
    num_colors: ClassVar[int] = 3
    num_attractive_colors: ClassVar[int] = 2

    # Food shape (could vary size, color)
    food_half_extents: ClassVar[Tuple[float, float, float]] =\
        (0.03, 0.03, 0.03)
    food_size: ClassVar[float] = food_half_extents[0] * 2
    food_mass: ClassVar[float] = 0.1

    # Ant shape
    ant_half_extents: ClassVar[Tuple[float, float, float]] =\
        (0.015, 0.01, 0.01)
    ant_mass: ClassVar[float] = 0.05
    ant_step_size: ClassVar[float] = 0.005

    # -------------------------------------------------------------------------
    # Types
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])

    # Food has color channels + "attractive" as 0.0 or 1.0
    _food_type = Type(
        "food", ["x", "y", "z", "rot", "is_held", "attractive", "r", "g", "b"],
        sim_features=["id", "attractive"])

    # Each ant might have orientation, but minimal for demonstration
    _ant_type = Type("ants", ["x", "y", "z", "rot", "target_food"],
                     sim_features=["id", "target_food"])

    def __init__(self,
                 use_gui: bool = True,
                 debug_layout: bool = True) -> None:
        # Create single robot
        self._robot = Object("robot", self._robot_type)

        # Create N food blocks
        self._blocks: List[Object] = []
        for i in range(self.num_food):
            name = f"food_{i}"
            food_obj = Object(name, self._food_type)
            self._blocks.append(food_obj)

        # Create M ants
        self._ants: List[Object] = []
        for i in range(self.num_ants):
            name = f"ant_{i}"
            ant_obj = Object(name, self._ant_type)
            self._ants.append(ant_obj)
        
        if CFG.ants_ants_attracted_to_points:
            self._ants_to_xy = dict()

        super().__init__(use_gui)
        self._debug_layout = debug_layout

        # Define predicates if needed (some are placeholders)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._food_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._On = Predicate("On", [self._food_type, self._food_type],
                             self._On_holds)
        self._OnTable = Predicate("OnTable", [self._food_type],
                                  self._OnTable_holds)
        self._Clear = Predicate("Clear", [self._food_type], self._Clear_holds)
        self._InSortedRegion = Predicate("InSortedRegion", 
                                               [self._food_type],
                                                self._InSortedRegion_holds)
        self._Attractive = Predicate("Attractive", [self._food_type],
                                    self._Attractive_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_ants"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Holding, self._HandEmpty, self._On, self._OnTable,
                self._Clear, self._InSortedRegion, self._Attractive}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return set()

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._food_type, self._ant_type}

    # -------------------------------------------------------------------------
    # Environment Setup

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Add a simple table
        table_id = create_object(asset_path="urdf/table.urdf",
                                 position=cls.table_pos,
                                 orientation=cls.table_orn,
                                 scale=1.0,
                                 use_fixed_base=True,
                                 physics_client_id=physics_client_id)
        bodies["table_id"] = table_id

        # Create the food objects
        food_ids = []
        for _ in range(cls.num_food):
            fid = create_pybullet_block(
                color=(0.5, 0.5, 0.5, 1.0),  # We’ll override color later
                half_extents=cls.food_half_extents,
                mass=cls.food_mass,
                friction=0.5,
                orientation=[0.0, 0.0, 0.0],
                physics_client_id=physics_client_id,
            )
            food_ids.append(fid)

        # Create the ants (small cuboids)
        ant_ids = []
        for _ in range(cls.num_ants):
            aid = create_pybullet_block(
                color=(0.3, 0.3, 0.3, 1.0),
                half_extents=cls.ant_half_extents,
                mass=cls.ant_mass,
                friction=0.5,
                orientation=[0.0, 0.0, 0.0],
                physics_client_id=physics_client_id,
            )
            ant_ids.append(aid)

        bodies["food_ids"] = food_ids
        bodies["ant_ids"] = ant_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        # Keep references for IDs
        for fobj, fid in zip(self._blocks, pybullet_bodies["food_ids"]):
            fobj.id = fid
        for aobj, aid in zip(self._ants, pybullet_bodies["ant_ids"]):
            aobj.id = aid

    # -------------------------------------------------------------------------
    # State Management

    def _get_object_ids_for_held_check(self) -> List[int]:
        # If we support robot picking up food blocks, return those IDs.
        return [f.id for f in self._blocks]
    
    def _create_task_specific_objects(self, state: State) -> None:
        pass

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._food_type:
            if feature == "attractive":
                return obj.attractive
        if obj.type == self._ant_type:
            if feature == "target_food":
                return obj.target_food.id

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _reset_custom_env_state(self, state: State) -> None:

        if CFG.ants_ants_attracted_to_points:
            self._ant_to_xy = dict()
            for ant_obj in state.get_objects(self._ant_type):
                self._ants_to_xy[ant_obj] = (
                    self._train_rng.uniform(self.one_third_x, self.two_third_x), 
                    self._train_rng.uniform(self.y_lb, self.y_ub))

        # Hide irrelevant objects
        oov_x, oov_y = self._out_of_view_xy
        block_objs = state.get_objects(self._food_type)
        for i in range(len(block_objs), len(self._blocks)):
            # Hide the remaining blocks
            update_object(self._blocks[i].id,
                          position=(oov_x, oov_y, self.z_lb),
                          physics_client_id=self._physics_client_id)
        
        ant_objs = state.get_objects(self._ant_type)
        for i in range(len(ant_objs), len(self._ants)):
            # Hide the remaining ants
            update_object(self._ants[i].id,
                          position=(oov_x, oov_y, self.z_lb),
                          physics_client_id=self._physics_client_id)

        for food in state.get_objects(self._food_type):
            r = state.get(food, "r")
            g = state.get(food, "g")
            b = state.get(food, "b")
            attractive = state.get(food, "attractive")
            update_object(food.id,
                          color=(r, g, b, 1.0),
                          physics_client_id=self._physics_client_id)
            food.attractive = attractive
        
        # Set ant's attractive food
        for ant_obj in state.get_objects(self._ant_type):
            food_id = state.get(ant_obj, "target_food")
            for food_obj in state.get_objects(self._food_type):
                if food_obj.id == food_id:
                    ant_obj.target_food = food_obj
                    break

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Override to (1) do usual robot step, (2) move ants toward attracted
        food with noise, and then (3) return the final state."""
        # Step the robot normally
        next_state = super().step(action, render_obs=render_obs)

        # Move ants. For each ant, find a target food object that is "attractive."
        # If there's more than one attractive block, pick the one it’s “assigned” to,
        # or the first in the list. Then move a small step toward it with noise.
        self._update_ant_positions(next_state)

        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    def _update_ant_positions(self, state: State) -> None:
        """For each ant, move it a small step toward its assigned attractive
        food."""
        for ant_obj in self._ants:

            # Move a small step toward it with noise
            ax = state.get(ant_obj, "x")
            ay = state.get(ant_obj, "y")
            if CFG.ants_ants_attracted_to_points:
                fx, fy = self._ants_to_xy[ant_obj]
            else:
                # Retrieve this ant’s assigned food
                target_food_obj = None
                for food_obj in state.get_objects(self._food_type):
                    if food_obj.id == state.get(ant_obj, "target_food"):
                        target_food_obj = food_obj
                        break
                if target_food_obj is None:
                    continue
                fx = state.get(target_food_obj, "x")
                fy = state.get(target_food_obj, "y")
            dist = np.sqrt((fx - ax)**2 + (fy - ay)**2)

            noise = 0.002
            if dist > 1e-6:
                dxn = (fx - ax) / dist
                dyn = (fy - ay) / dist
                new_x = ax + self.ant_step_size * dxn + np.random.uniform(
                    -noise, noise)
                new_y = ay + self.ant_step_size * dyn + np.random.uniform(
                    -noise, noise)
                new_rot = np.arctan2(new_y - ay, new_x - ax)
            else:
                new_x = ax
                new_y = ay
                new_rot = state.get(ant_obj, "rot")

            az = state.get(ant_obj, "z")
            update_object(
                ant_obj.id,
                position=(new_x, new_y, az),
                orientation=p.getQuaternionFromEuler([0.0, 0.0, new_rot]),
                physics_client_id=self._physics_client_id,
            )

    # -------------------------------------------------------------------------
    # Predicates

    @classmethod
    def _Holding_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        # For demonstration, check if food is_held = 1.0
        _, food = objects
        return state.get(food, "is_held") > 0.5

    @classmethod
    def _HandEmpty_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        # E.g. open fingers threshold
        robot, = objects
        return state.get(robot, "fingers") > 0.03

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        if state.get(block1, "is_held") >= self.held_tol or \
           state.get(block2, "is_held") >= self.held_tol:
            return False
        x1 = state.get(block1, "x")
        y1 = state.get(block1, "y")
        z1 = state.get(block1, "z")
        x2 = state.get(block2, "x")
        y2 = state.get(block2, "y")
        z2 = state.get(block2, "z")
        return np.allclose([x1, y1, z1], [x2, y2, z2 + self.food_size],
                           atol=0.01)

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        z = state.get(block, "z")
        desired_z = self.table_height + self.food_half_extents[2]
        return (state.get(block, "is_held") < 0.5) and \
            (desired_z-0.1 < z < desired_z+0.1)
    
    def _InSortedRegion_holds(self, state: State, 
                              objects: Sequence[Object]) -> bool:
        """
        Defined as none attractive food blocks in the first third region (left)
        and attractive food blocks in the last third region (right).
        """
        blocks, = objects
        # Check if the blocks are in a sorted region
        x = state.get(blocks, "x")
        held = state.get(blocks, "is_held")
        attractive = state.get(blocks, "attractive")

        if held > 0.5:
            return False
        if attractive < 0.5:
            return x < self.one_third_x
        else:
            return x > self.two_third_x
    
    def _Attractive_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "attractive") > 0.5

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, [self._robot] + objects):
            return False
        block, = objects
        for other_block in state.get_objects(self._food_type):
            if self._On_holds(state, [other_block, block]):
                return False
        return True

    # -------------------------------------------------------------------------
    # Task Generation

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
            init_dict = {}
            block_by_color = {}
            attractive_food_objs = []

            task_color_palette = rng.choice(self._obj_colors_main, 
                                            size=self.num_colors, 
                                            replace=False)
            task_color_palette = [tuple(c) for c in task_color_palette]
            attractive_colors = rng.choice(task_color_palette, 
                                           size=self.num_attractive_colors, 
                                           replace=False)
            attractive_colors = [tuple(c) for c in attractive_colors]

            # 1) Robot
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }
            init_dict[self._robot] = robot_dict

            # 2) Food
            num_blocks = 4
            num_ants = 6
            blocks_positions = sample_collision_free_2d_positions(
                num_blocks, 
                x_range=(self.x_lb, self.one_third_x), 
                y_range=(self.y_lb + 2 * self.food_size, 
                         self.y_ub - 2 * self.food_size),
                shape_type="rectangle",
                shape_params=(1.3 * self.food_size, 
                              1.3 * self.food_size, 0.0),
                rng=rng,
            )

            for i in range(num_blocks):
                pos = blocks_positions[i]
                fobj = self._blocks[i]
                # Random position
                rot = rng.uniform(-np.pi/2, np.pi/2)
                # Pick color
                if i < self.num_colors:
                    color_rgba = task_color_palette[i]
                else:
                    color_rgba = tuple(rng.choice(task_color_palette))

                attractive = color_rgba in attractive_colors
                init_dict[fobj] = {
                    "x": pos[0],
                    "y": pos[1],
                    "z": self.z_lb + self.food_half_extents[2],  # on table
                    "rot": rot,
                    "is_held": 0.0,
                    "attractive": float(attractive),
                    "r": color_rgba[0],
                    "g": color_rgba[1],
                    "b": color_rgba[2],
                }

                block_by_color.setdefault(color_rgba, []).append(fobj)
                if attractive:
                    attractive_food_objs.append(fobj) 

            # 3) Ants
            for i in range(num_ants):
                aobj = self._ants[i]
                x = rng.uniform(self.x_ub - self.padding, self.x_ub)
                y = rng.uniform(self.y_lb, self.y_ub)
                rot = rng.uniform(-np.pi, np.pi)
                init_dict[aobj] = {
                    "x": x,
                    "y": y,
                    "z": self.z_lb + self.ant_half_extents[2],
                    "rot": rot,
                    "target_food": rng.choice(attractive_food_objs).id,
                }

            init_state = utils.create_state_from_dict(init_dict)

            # The goal is to have all the blocks of the same color in a tower.
            # First sort the blocks by color
            goal_atoms = set()

            for c, blocks_group in block_by_color.items():
                goal_atoms.add(GroundAtom(self._InSortedRegion, 
                                         [blocks_group[0]]))
                if len(blocks_group) > 1:
                    # Base block on table
                    for j in range(len(blocks_group) - 1):
                        goal_atoms.add(GroundAtom(self._On, [blocks_group[j+1], 
                                                             blocks_group[j]]))
            tasks.append(EnvironmentTask(init_state, goal_atoms))
        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 1
    CFG.env = "pybullet_ants"
    CFG.pybullet_sim_steps_per_action = 1
    env = PyBulletAntsEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(1, rng)[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))

        env.step(action)
        time.sleep(0.01)
