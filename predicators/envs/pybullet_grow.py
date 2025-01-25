"""Grow plants with fertalizers.

python predicators/main.py --approach oracle --env pybullet_grow --seed 1 \
--num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --make_test_videos \
--sesame_check_expected_atoms False
"""

import logging
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletGrowEnv(PyBulletEnv):
    """A PyBullet environment with cups and jugs, where pouring matching-color
    liquid into a cup grows a 'plant'. The goal is to have both cups grown.

    We want the 'growth' of both cups to exceed some threshold as a goal.
    from PyBullet Coffee domain.
    x: cup <-> jug,
    y: robot <-> machine
    z: up <-> down
    """

    # -------------------------------------------------------------------------
    # Global configuration / geometry

    # How many cups and jugs to create
    num_cups: ClassVar[int] = 2
    num_jugs: ClassVar[int] = 2

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

    # robot config
    _finger_action_tol: ClassVar[float] = 5e-3
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.72, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2
    tilt_lb: ClassVar[float] = robot_init_tilt
    tilt_ub: ClassVar[float] = tilt_lb - np.pi / 4

    # jug/cup geometry
    jug_height: ClassVar[float] = 0.12
    jug_init_z: ClassVar[float] = z_lb + jug_height / 2
    jug_init_rot: ClassVar[float] = -np.pi / 2

    # For no-collision sampling
    collision_padding: ClassVar[float] = 0.15
    small_padding: ClassVar[float] = 0.1  # just for spacing in XY checks

    # Growth logic
    growth_height: ClassVar[float] = 0.3
    growth_color: ClassVar[Tuple[float]] = (0.35, 1, 0.3, 0.8)

    pour_rate: ClassVar[float] = 0.1

    # Tolerance
    place_jug_tol: ClassVar[float] = 1e-3

    # Camera
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -38  # 0: low <-> -90: high
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Types now include r, g, b features for color
    _robot_type = Type("robot", ["x", "y", "z", "fingers", "tilt", "wrist"])
    _cup_type = Type("cup", ["x", "y", "z", "growth", "r", "g", "b"])
    _jug_type = Type("jug", ["x", "y", "z", "rot", "is_held", "r", "g", "b"])

    def __init__(self, use_gui: bool = True) -> None:
        # Create the single robot Object
        self._robot = Object("robot", self._robot_type)

        # Create containers for cups and jugs
        self._cups: List[Object] = []
        for i in range(self.num_cups):
            cup_name = f"cup{i}"
            self._cups.append(Object(cup_name, self._cup_type))

        self._jugs: List[Object] = []
        for i in range(self.num_jugs):
            jug_name = f"jug{i}"
            self._jugs.append(Object(jug_name, self._jug_type))

        # For tracking the "liquid bodies" we create for each cup
        self._cup_to_liquid_id: Dict[Object, Optional[int]] = {}

        super().__init__(use_gui)

        # Define Predicates
        self._Grown = Predicate("Grown", [self._cup_type], self._Grown_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._jug_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._JugOnTable = Predicate("JugOnTable", [self._jug_type],
                                     self._JugOnTable_holds)
        self._CupOnTable = Predicate("CupOnTable", [self._cup_type],
                                     self._CupOnTable_holds)
        self._SameColor = Predicate("SameColor",
                                    [self._cup_type, self._jug_type],
                                    self._SameColor_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_grow"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Grown, self._Holding, self._HandEmpty, self._JugOnTable,
            self._SameColor, self._CupOnTable
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Grown}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._cup_type, self._jug_type}

    # -------------------------------------------------------------------------
    # Environment Setup

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Create the PyBullet environment and the robot."""
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Add a table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id

        # Create the cups
        cup_ids = []
        for _ in range(cls.num_cups):
            # For now, just give a placeholder color; we'll update color below
            cup_id = create_object(asset_path="urdf/pot-pixel.urdf",
                                   mass=50,
                                   physics_client_id=physics_client_id)
            cup_ids.append(cup_id)
        bodies["cup_ids"] = cup_ids

        # Create the jugs
        jug_ids = []
        for _ in range(cls.num_jugs):
            jug_id = create_object(asset_path="urdf/jug-pixel.urdf",
                                   physics_client_id=physics_client_id)
            jug_ids.append(jug_id)
        bodies["jug_ids"] = jug_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references (IDs) to cups and jugs inside self._cups,
        self._jugs."""
        for i, cup in enumerate(self._cups):
            cup.id = pybullet_bodies["cup_ids"][i]
        for i, jug in enumerate(self._jugs):
            jug.id = pybullet_bodies["jug_ids"][i]

    # -------------------------------------------------------------------------
    # State Management

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return IDs of jugs (since we can only hold jugs)."""
        jug_ids = [jug.id for jug in self._jugs if jug.id is not None]
        return jug_ids

    def _create_task_specific_objects(self, state: State) -> None:
        """No extra objects to create beyond cups and jugs."""
        pass

    def _extract_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        # For growth, we look up the height of the liquid body
        if obj.type == self._cup_type and feature == "growth":
            liquid_id = self._cup_to_liquid_id.get(obj, None)
            if liquid_id is not None:
                shape_data = p.getVisualShapeData(
                    liquid_id, physicsClientId=self._physics_client_id)
                if shape_data:  # (handle the case shape_data might be empty)
                    # shape_data[0][3][2] is the Z dimension of the box half-extents*2, etc.
                    height = shape_data[0][3][2]
                    return height
            return 0.0

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _reset_custom_env_state(self, state: State) -> None:
        """Called in _reset_state to handle any custom resetting."""
        # Remove existing "liquid bodies"
        for liquid_id in self._cup_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id,
                             physicsClientId=self._physics_client_id)
        self._cup_to_liquid_id.clear()

        # Recreate the liquid bodies as needed
        for cup in self._cups:
            liquid_id = self._create_pybullet_liquid_for_cup(cup, state)
            self._cup_to_liquid_id[cup] = liquid_id

        # Also update the PyBullet color on each cup/jug to match the (r,g,b) in the state
        for cup in self._cups:
            if cup.id is not None:
                r = state.get(cup, "r")
                g = state.get(cup, "g")
                b = state.get(cup, "b")
                update_object(cup.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)
        for jug in self._jugs:
            if jug.id is not None:
                r = state.get(jug, "r")
                g = state.get(jug, "g")
                b = state.get(jug, "b")
                update_object(jug.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Pouring logic

    def step(self, action: Action, render_obs: bool = False) -> State:
        """Let parent handle the robot stepping, then apply custom pouring
        logic."""
        next_state = super().step(action, render_obs=render_obs)

        # If a jug is in the robot's hand, and tilt is large, check if over a color-matching cup
        if self._held_obj_id is not None:
            # Which jug is being held?
            jug_obj = self.get_object_by_id(self._held_obj_id)
            if jug_obj is not None:
                tilt = next_state.get(self._robot, "tilt")
                # If tilt near a "pouring" angle
                if abs(tilt - np.pi / 4) < 0.1:
                    jug_r = next_state.get(jug_obj, "r")
                    jug_g = next_state.get(jug_obj, "g")
                    jug_b = next_state.get(jug_obj, "b")
                    jug_x = next_state.get(jug_obj, "x")
                    jug_y = next_state.get(jug_obj, "y")

                    # Check if over a cup with the same (r,g,b)
                    for cup_obj in self._cups:
                        cx = next_state.get(cup_obj, "x")
                        cy = next_state.get(cup_obj, "y")
                        dist = np.hypot(jug_x - cx, jug_y - cy)
                        # If close enough
                        if dist < 0.15:
                            cup_r = next_state.get(cup_obj, "r")
                            cup_g = next_state.get(cup_obj, "g")
                            cup_b = next_state.get(cup_obj, "b")

                            # If colors match (within small tolerance)
                            if (abs(jug_r - cup_r) < 1e-3
                                    and abs(jug_g - cup_g) < 1e-3
                                    and abs(jug_b - cup_b) < 1e-3):
                                # Increase growth
                                current_growth = next_state.get(
                                    cup_obj, "growth")
                                new_growth = min(
                                    1.0, current_growth + self.pour_rate)

                                # Remove old liquid body, set new growth
                                old_liquid_id = self._cup_to_liquid_id[cup_obj]
                                if old_liquid_id is not None:
                                    p.removeBody(old_liquid_id,
                                                 physicsClientId=self.
                                                 _physics_client_id)

                                next_state.set(cup_obj, "growth", new_growth)
                                self._cup_to_liquid_id[cup_obj] = \
                                    self._create_pybullet_liquid_for_cup(cup_obj, next_state)

        final_state = self._get_state()
        self._current_observation = final_state
        return final_state

    # -------------------------------------------------------------------------
    # Predicates

    @staticmethod
    def _Grown_holds(state: State, objects: Sequence[Object]) -> bool:
        """A cup is "grown" if growth > growth_height."""
        (cup, ) = objects
        return state.get(cup, "growth") > PyBulletGrowEnv.growth_height

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        (robot, jug) = objects
        return state.get(jug, "is_held") > 0.5

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        (robot, ) = objects
        return state.get(robot, "fingers") > 0.02

    def _InTableBoundry(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        if x < self.x_lb or x > self.x_ub or y < self.y_lb or y > self.y_ub:
            return False
        return True

    def _JugOnTable_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        (jug, ) = objects
        # If being held, it's not "on the table"
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        return self._InTableBoundry(state, [jug])

    def _CupOnTable_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        return self._InTableBoundry(state, objects)

    @staticmethod
    def _SameColor_holds(state: State, objects: Sequence[Object]) -> bool:
        (cup, jug) = objects
        eps = 1e-3
        if abs(state.get(cup, "r") - state.get(jug, "r")) > eps:
            return False
        if abs(state.get(cup, "g") - state.get(jug, "g")) > eps:
            return False
        if abs(state.get(cup, "b") - state.get(jug, "b")) > eps:
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
            # We'll create an initial state dictionary
            init_dict = {}

            # Robot at center
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist
            }
            init_dict[self._robot] = robot_dict

            # Keep track of where we've placed objects so far
            existing_xys: Set[Tuple[float, float]] = set()

            jug_colors = []
            # Sample positions and colors for jugs
            for jug_obj in self._jugs:
                x, y = self._sample_table_xy(rng, existing_xys)
                # Make sure we don't sample the same color twice
                while True:
                    c = list(rng.choice(self._obj_colors))
                    if c not in jug_colors:
                        break
                jug_colors.append(c)
                r_col, g_col, b_col, _ = c
                jug_dict = {
                    "x": x,
                    "y": y,
                    "z": self.jug_init_z,
                    "rot": self.jug_init_rot,
                    "is_held": 0.0,
                    "r": r_col,
                    "g": g_col,
                    "b": b_col,
                }
                init_dict[jug_obj] = jug_dict

            # Sample positions and colors for cups
            for i, cup_obj in enumerate(self._cups):
                x, y = self._sample_table_xy(rng, existing_xys)
                # Sample a color (r, g, b, a)
                if i < len(jug_colors):
                    r_col, g_col, b_col, _ = jug_colors[i]
                else:
                    r_col, g_col, b_col, _ = rng.choice(jug_colors)
                cup_dict = {
                    "x": x,
                    "y": y,
                    "z": self.jug_init_z,  # small offset so it sits on table
                    "growth": 0.0,
                    "r": r_col,
                    "g": g_col,
                    "b": b_col,
                }
                init_dict[cup_obj] = cup_dict

            # Build the initial State
            init_state = utils.create_state_from_dict(init_dict)

            # The goal is that all cups are grown
            goal_atoms = set()
            for cup_obj in self._cups:
                goal_atoms.add(GroundAtom(self._Grown, [cup_obj]))
                goal_atoms.add(GroundAtom(self._CupOnTable, [cup_obj]))
            # # plus jugs are on the table
            # for jug_obj in self._jugs:
            #     goal_atoms.add(GroundAtom(self._JugOnTable, [jug_obj]))

            task = EnvironmentTask(init_state, goal_atoms)
            tasks.append(task)

        return self._add_pybullet_state_to_tasks(tasks)

    # -------------------------------------------------------------------------
    # Sampling helpers

    def _table_xy_is_clear(self, x: float, y: float,
                           existing_xys: Set[Tuple[float, float]]) -> bool:
        """Ensure we don't place objects too close along x or y."""
        # If (x, y) is sufficiently far from all existing positions,
        # return True; otherwise False.
        for (other_x, other_y) in existing_xys:
            # Check square distance
            if np.sqrt((x - other_x)**2 + (y - other_y)**2) <\
                    self.small_padding:
                return False
        return True

    def _sample_table_xy(
            self, rng: np.random.Generator,
            existing_xys: Set[Tuple[float, float]]) -> Tuple[float, float]:
        max_tries = 1000
        for _ in range(max_tries):
            x = rng.uniform(self.x_lb + self.small_padding,
                            self.x_ub - self.small_padding)
            y = rng.uniform(self.y_lb + 2 * self.small_padding,
                            self.y_ub - 2 * self.small_padding)
            if self._table_xy_is_clear(x, y, existing_xys):
                existing_xys.add((x, y))
                return (x, y)
        raise RuntimeError(
            "Could not sample a valid table (x, y) without crowding.")

    # -------------------------------------------------------------------------
    # Liquid creation

    def _create_pybullet_liquid_for_cup(
        self,
        cup: Object,
        state: State,
        growth_color: Tuple[float, float, float, float] = growth_color
    ) -> Optional[int]:
        """Given a cup's 'growth' feature, create (or None) a small PyBullet
        body."""
        current_liquid = state.get(cup, "growth")
        if current_liquid <= 0:
            return None

        # Make a box that sits inside the cup
        liquid_height = current_liquid
        half_extents = [0.03, 0.03, liquid_height / 2]
        cx = state.get(cup, "x")
        cy = state.get(cup, "y")
        cz = self.z_lb + liquid_height / 2  # sits on table

        if CFG.grow_plant_same_color_as_cup:
            color = (state.get(cup, "r"), state.get(cup,
                                                    "g"), state.get(cup,
                                                                    "b"), 0.8)
        else:
            color = growth_color
        return create_pybullet_block(color=color,
                                     half_extents=half_extents,
                                     mass=10.0,
                                     friction=0.5,
                                     position=(cx, cy, cz),
                                     physics_client_id=self._physics_client_id)


if __name__ == "__main__":
    """Run a simple simulation to test the environment."""
    import time

    CFG.env = "pybullet_grow"
    CFG.seed = 0
    CFG.pybullet_sim_steps_per_action = 1

    env = PyBulletGrowEnv(use_gui=True)
    rng = np.random.default_rng(CFG.seed)
    task = env._make_tasks(1, rng)[0]
    env._reset_state(task.init)

    while True:
        # Robot does nothing
        action = Action(np.array(env._pybullet_robot.initial_joint_positions))
        env.step(action)
        time.sleep(0.01)
