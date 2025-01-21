from typing import Any, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, \
    State, Type


class PyBulletAntEnv(PyBulletEnv):
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

    # Camera
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # Define how many ants and how many food blocks
    num_ants: ClassVar[int] = 4
    num_food: ClassVar[int] = 6

    # Food shape (could vary size, color)
    food_half_extents: ClassVar[Tuple[float, float, float]] =\
        (0.03, 0.03, 0.03)
    food_mass: ClassVar[float] = 0.1

    # Ant shape
    ant_half_extents: ClassVar[Tuple[float, float, float]] =\
        (0.01, 0.015, 0.01)
    ant_mass: ClassVar[float] = 0.05
    ant_step_size: ClassVar[float] = 0.0002

    # Color palette: e.g. 3 basic colors
    color_palette: ClassVar[List[Tuple[float, float, float, float]]] = [
        (1.0, 0.0, 0.0, 1.0),  # red
        (0.0, 1.0, 0.0, 1.0),  # green
        (0.0, 0.0, 1.0, 1.0),  # blue
    ]

    # -------------------------------------------------------------------------
    # Types
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])

    # Food has color channels + "attractive" as 0.0 or 1.0
    _food_type = Type(
        "food", ["x", "y", "z", "rot", "is_held", "attractive", "r", "g", "b"],
        sim_features=["id", "r", "g", "b", "attractive"])

    # Each ant might have orientation, but minimal for demonstration
    _ant_type = Type("ant", ["x", "y", "z", "rot"],
                     sim_features=["id", "target_food"])

    def __init__(self,
                 use_gui: bool = True,
                 debug_layout: bool = True) -> None:
        # Create single robot
        self._robot = Object("robot", self._robot_type)

        # Create N food blocks
        self.food: List[Object] = []
        for i in range(self.num_food):
            name = f"food_{i}"
            food_obj = Object(name, self._food_type)
            self.food.append(food_obj)

        # Create M ants
        self.ants: List[Object] = []
        for i in range(self.num_ants):
            name = f"ant_{i}"
            ant_obj = Object(name, self._ant_type)
            self.ants.append(ant_obj)

        super().__init__(use_gui)
        self._debug_layout = debug_layout

        # Define predicates if needed (some are placeholders)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._food_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_ant"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Holding, self._HandEmpty}

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
        for fobj, fid in zip(self.food, pybullet_bodies["food_ids"]):
            fobj.id = fid
        for aobj, aid in zip(self.ants, pybullet_bodies["ant_ids"]):
            aobj.id = aid

    # -------------------------------------------------------------------------
    # State Management

    def _get_object_ids_for_held_check(self) -> List[int]:
        # If we support robot picking up food blocks, return those IDs.
        return [f.id for f in self.food]

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._food_type:
            if feature == "attractive":
                return obj.attractive
            elif feature == "r":
                return obj.r
            elif feature == "g":
                return obj.g
            elif feature == "b":
                return obj.b

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _reset_custom_env_state(self, state: State) -> None:
        for food in self.food:
            r = state.get(food, "r")
            g = state.get(food, "g")
            b = state.get(food, "b")
            update_object(food.id,
                          color=(r, g, b, 1.0),
                          physics_client_id=self._physics_client_id)

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
        for ant_obj in self.ants:
            # Retrieve this ant’s assigned food
            target_food_obj = getattr(ant_obj, "target_food", None)
            if target_food_obj is None:
                continue

            # Move a small step toward it with noise
            ax = state.get(ant_obj, "x")
            ay = state.get(ant_obj, "y")
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
        return state.get(robot, "fingers") > 0.2

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

            num_attractive_colors = 2
            attractive_colors_id = rng.integers(0, len(self.color_palette),
                                                num_attractive_colors)
            attractive_colors = [
                self.color_palette[i] for i in attractive_colors_id
            ]

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
            for i, fobj in enumerate(self.food):
                # Random position
                one_third_line = (self.x_lb + self.x_ub) / 3
                two_third_line = 2 * one_third_line
                # Draw a debug ling for two_third_line
                if self._debug_layout:
                    p.addUserDebugLine(
                        [two_third_line, self.y_lb, self.z_lb + 0.01],
                        [two_third_line, self.y_ub, self.z_lb + 0.01],
                        [1, 0, 0],
                        lineWidth=3)
                x = rng.uniform(self.x_lb, one_third_line)
                y = rng.uniform(self.y_lb, self.y_ub)
                rot = rng.uniform(-np.pi, np.pi)
                # Pick color
                # First, prepare a list of color indices to distribute blocks evenly among colors
                if i == 0:  # do this once before assigning colors
                    blocks_per_color = self.num_food // len(self.color_palette)
                    remainder = self.num_food % len(self.color_palette)
                    self._color_indices = []
                    for c in range(len(self.color_palette)):
                        count = blocks_per_color + (1 if c < remainder else 0)
                        self._color_indices += [c] * count
                    rng.shuffle(self._color_indices)

                # Then assign the color from this prepared list
                color_idx = self._color_indices[i]
                color_rgba = self.color_palette[color_idx]
                # Store color in object attributes
                fobj.r = color_rgba[0]
                fobj.g = color_rgba[1]
                fobj.b = color_rgba[2]
                # If color is in attractive_colors, set "attractive"=1
                if color_rgba in attractive_colors:
                    fobj.attractive = 1.0
                else:
                    fobj.attractive = 0.0

                init_dict[fobj] = {
                    "x": x,
                    "y": y,
                    "z": self.z_lb + self.food_half_extents[2],  # on table
                    "rot": rot,
                    "is_held": 0.0,
                    "attractive": fobj.attractive,
                    "r": fobj.r,
                    "g": fobj.g,
                    "b": fobj.b,
                }

            # Collect the "attractive" foods for random assignment
            attractive_food_objs = [
                f for f in self.food if f.attractive == 1.0
            ]

            # 3) Ants
            for i, aobj in enumerate(self.ants):
                x = rng.uniform(self.x_ub - self.padding, self.x_ub)
                y = rng.uniform(self.y_lb, self.y_ub)
                rot = rng.uniform(-np.pi, np.pi)
                init_dict[aobj] = {
                    "x": x,
                    "y": y,
                    "z": self.z_lb + self.ant_half_extents[2],
                    "rot": rot,
                }
                # Assign a random attractive block if any exist
                # (store that choice as an attribute in the Python object)
                if attractive_food_objs:
                    aobj.target_food = rng.choice(attractive_food_objs)
                else:
                    aobj.target_food = None

            init_state = utils.create_state_from_dict(init_dict)
            goal_atoms = set()
            tasks.append(EnvironmentTask(init_state, goal_atoms))

        return self._add_pybullet_state_to_tasks(tasks)


if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    # Make a task
    CFG.seed = 1
    CFG.pybullet_sim_steps_per_action = 1
    env = PyBulletAntEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(1, rng)[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))

        env.step(action)
        time.sleep(0.01)
