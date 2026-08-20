"""Grow plants with fertalizers.

python predicators/main.py --approach oracle --env pybullet_grow --seed 1 \
--num_test_tasks 1 --use_gui --debug --num_train_tasks 0 \
--sesame_max_skeletons_optimized 1  --make_failure_videos --video_fps 20 \
--pybullet_camera_height 900 --pybullet_camera_width 900 --make_test_videos \
--sesame_check_expected_atoms False
"""

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, update_object
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

    # Table / workspace config
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0., 0., np.pi / 2])

    x_lb: ClassVar[float] = 0.45
    x_ub: ClassVar[float] = 1.05
    y_lb: ClassVar[float] = 1.15
    y_ub: ClassVar[float] = 1.55
    y_mid: ClassVar[float] = (y_lb + y_ub) / 2
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2

    # robot config
    # grasp_tol_small and _finger_action_tol used to be overridden here
    # (5e-2 / 5e-3, legacy tuning for the pre-skill-factory options); the
    # base-class values work for the jug handle grasp and keep grasp
    # detection consistent with every other domain.
    pour_pos_tol_factor: ClassVar[float] = 1.8
    pour_pos_tol: ClassVar[float] = 0.005 * pour_pos_tol_factor
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
    jug_handle_height: ClassVar[float] = 0.1
    jug_radius: ClassVar[float] = 0.1
    jug_handle_offset: ClassVar[float] = 1.05 * jug_radius
    cup_radius: ClassVar[float] = jug_radius
    cup_capacity_ub: ClassVar[float] = 1

    # For no-collision sampling
    collision_padding: ClassVar[float] = 0.10
    small_padding: ClassVar[float] = 0.1  # just for spacing in XY checks

    # Growth logic
    growth_height: ClassVar[float] = 0.3
    max_growth_height: ClassVar[float] = 0.3
    growth_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.35, 1, 0.3, 0.8)

    pour_rate: ClassVar[float] = 0.005
    pour_x_offset: ClassVar[float] = cup_radius
    pour_y_offset: ClassVar[float] = -3 * (cup_radius + jug_radius)
    pour_z_offset: ClassVar[float] = 2.5 * (cup_capacity_ub + jug_height -\
                                            jug_handle_height)

    # Tolerance
    place_jug_tol: ClassVar[float] = 1e-3

    # Camera
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -38  # 0: low <-> -90: high
    _camera_target: ClassVar[Pose3D] = (0.75, 1.25, 0.42)

    # Types now include r, g, b features for color
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])
    _cup_type = Type("cup", ["x", "y", "z", "growth", "r", "g", "b"])
    _jug_type = Type("jug", ["x", "y", "z", "rot", "is_held", "r", "g", "b"],
                     sim_features=["id", "init_x", "init_y", "init_z"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # Create the single robot Object
        self._robot = Object("robot", self._robot_type)

        # Create containers for cups and jugs (create enough for max needed)
        max_cups = max(max(CFG.grow_num_cups_train),
                       max(CFG.grow_num_cups_test))
        max_jugs = max(max(CFG.grow_num_jugs_train),
                       max(CFG.grow_num_jugs_test))

        self._cups: List[Object] = []
        for i in range(max_cups):
            cup_name = f"cup{i}"
            self._cups.append(Object(cup_name, self._cup_type))

        self._jugs: List[Object] = []
        for i in range(max_jugs):
            jug_name = f"jug{i}"
            self._jugs.append(Object(jug_name, self._jug_type))

        # For tracking the "liquid bodies" we create for each cup
        self._cup_to_liquid_id: Dict[Object, Optional[int]] = {}

        super().__init__(use_gui, **kwargs)

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
        self._JugAboveCup = Predicate("JugAboveCup",
                                      [self._jug_type, self._cup_type],
                                      self._JugAboveCup_holds)
        self._NotAboveCup = Predicate("NotAboveCup",
                                      [self._robot_type, self._jug_type],
                                      self._NotAboveCup_holds)
        self._HandTilted = Predicate("HandTilted", [self._robot_type],
                                     self._HandTilted_holds)

    def get_extra_collision_ids(self) -> Sequence[int]:
        """Return liquid body IDs so motion planning avoids grown plants."""
        return [
            lid for lid in self._cup_to_liquid_id.values() if lid is not None
        ]

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_grow"

    @classmethod
    def _get_jug_handle_grasp(cls, state: State,
                              jug: Object) -> Tuple[float, float, float]:
        """Get the grasp position for the jug handle."""
        rot = state.get(jug, "rot")
        target_x = state.get(jug, "x") + np.cos(rot) * cls.jug_handle_offset
        target_y = state.get(jug, "y") + np.sin(rot) * cls.jug_handle_offset
        target_z = cls.z_lb + cls.jug_handle_height
        return (target_x, target_y, target_z)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Grown, self._Holding, self._HandEmpty, self._JugOnTable,
            self._SameColor, self._CupOnTable, self._JugAboveCup,
            self._NotAboveCup, self._HandTilted
        }

    @property
    def target_predicates(self) -> Set[Predicate]:
        target_predicates = self.predicates.copy()
        target_predicates.remove(self._HandTilted)
        return target_predicates

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

        # Create the cups (create enough for max needed)
        max_cups = max(max(CFG.grow_num_cups_train),
                       max(CFG.grow_num_cups_test))
        max_jugs = max(max(CFG.grow_num_jugs_train),
                       max(CFG.grow_num_jugs_test))

        cup_ids = []
        for _ in range(max_cups):
            # For now, just give a placeholder color; we'll update color below
            cup_id = create_object(
                asset_path="urdf/pot-pixel.urdf",
                physics_client_id=physics_client_id,
                use_fixed_base=True,
            )
            cup_ids.append(cup_id)
        bodies["cup_ids"] = cup_ids

        # Create the jugs
        jug_ids = []
        for _ in range(max_jugs):
            jug_id = create_object(asset_path="urdf/jug-pixel.urdf",
                                   physics_client_id=physics_client_id)
            jug_ids.append(jug_id)
        bodies["jug_ids"] = jug_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references (IDs) to cups and jugs inside self._cups,
        self._jugs."""
        self._table_ids = [pybullet_bodies["table_id"]]
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

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        # For growth, we look up the height of the liquid body
        if obj.type == self._cup_type and feature == "growth":
            liquid_id = self._cup_to_liquid_id.get(obj, None)
            if liquid_id is not None:
                shape_data = p.getVisualShapeData(
                    liquid_id, physicsClientId=self._physics_client_id)
                if shape_data:  # (handle the case shape_data might be empty)
                    # shape_data[0][3][2] is the Z dimension of the box
                    # half-extents*2, etc.
                    height = shape_data[0][3][2]
                    return height
            return 0.0

        raise ValueError(f"Unknown feature {feature} for object {obj}")

    def _set_domain_specific_state(self, state: State) -> None:
        """Set out-of-view positioning, jug init positions, liquid bodies, and
        cup/jug colors."""
        cups = state.get_objects(self._cup_type)
        jugs = state.get_objects(self._jug_type)

        # Store jug initial positions
        for jug in jugs:
            jug.init_x = state.get(jug, "x")
            jug.init_y = state.get(jug, "y")
            jug.init_z = state.get(jug, "z")

        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(cups), len(self._cups)):
            update_object(self._cups[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)
        for i in range(len(jugs), len(self._jugs)):
            update_object(self._jugs[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)

        # Remove existing liquid bodies
        for liquid_id in self._cup_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id,
                             physicsClientId=self._physics_client_id)
        self._cup_to_liquid_id.clear()

        # Recreate the liquid bodies as needed
        for cup in cups:
            liquid_id = self._create_pybullet_liquid_for_cup(cup, state)
            self._cup_to_liquid_id[cup] = liquid_id

        # Update colors
        for cup in cups:
            if cup.id is not None:
                r = state.get(cup, "r")
                g = state.get(cup, "g")
                b = state.get(cup, "b")
                update_object(cup.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)
        for jug in jugs:
            if jug.id is not None:
                r = state.get(jug, "r")
                g = state.get(jug, "g")
                b = state.get(jug, "b")
                update_object(jug.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Pouring logic

    def _domain_specific_step(self) -> None:
        """Apply custom pouring logic."""
        state = self._get_state()
        self._handle_pouring(state)

    def _handle_pouring(self, state: State) -> None:
        if self._held_obj_id is None:
            return
        if abs(state.get(self._robot, "tilt") -
               self.tilt_ub) < PyBulletCoffeeEnv.pour_angle_tol:
            # Identify which cup (if any) is being poured into
            cup = self._get_cup_to_pour(state)
            if cup is None:
                return

            # Get the jug being held
            jug = self.get_object_by_id(self._held_obj_id)

            # Check if jug and cup colors match
            if not self._SameColor_holds(state, [cup, jug]):
                return  # No growth if colors don't match

            current_growth = state.get(cup, "growth")
            new_growth = min(self.max_growth_height,
                             current_growth + self.pour_rate)

            # Remove old liquid body, set new growth
            old_liquid_id = self._cup_to_liquid_id[cup]
            if old_liquid_id is not None:
                p.removeBody(old_liquid_id,
                             physicsClientId=self._physics_client_id)

            state.set(cup, "growth", new_growth)
            self._cup_to_liquid_id[cup] = \
                    self._create_pybullet_liquid_for_cup(cup, state)

    def _get_cup_to_pour(self, state: State) -> Optional[Object]:
        # Which jug is being held?
        assert self._held_obj_id is not None
        jug_obj = self.get_object_by_id(self._held_obj_id)
        jug_x = state.get(jug_obj, "x")
        jug_y = state.get(jug_obj, "y")
        jug_z = self._get_jug_z(state, jug_obj)
        jug_pos = (jug_x, jug_y, jug_z)
        closest_cup = None
        closest_cup_dist = float("inf")
        for cup in state.get_objects(self._cup_type):
            target = PyBulletCoffeeEnv._get_pour_position(state, cup)  # pylint: disable=protected-access
            sq_dist = np.sum(np.subtract(jug_pos, target)**2)
            if sq_dist < self.pour_pos_tol and sq_dist < closest_cup_dist:
                closest_cup = cup
                closest_cup_dist = sq_dist
        return closest_cup

    def _get_jug_z(self, state: State, jug: Object) -> float:
        if state.get(jug, "is_held") > 0.5:
            # Offset to account for handle.
            return state.get(self._robot, "z") -\
                PyBulletCoffeeEnv.jug_handle_height()
        # On the table.
        return self.z_lb

    # -------------------------------------------------------------------------
    # Predicates

    @staticmethod
    def _Grown_holds(state: State, objects: Sequence[Object]) -> bool:
        """A cup is "grown" if growth > growth_height."""
        (cup, ) = objects
        return state.get(cup, "growth") >= PyBulletGrowEnv.growth_height

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, jug = objects
        return state.get(jug, "is_held") > 0.5

    def _HandEmpty_holds(self, state: State,
                         _objects: Sequence[Object]) -> bool:
        # (robot, ) = objects
        # return state.get(robot, "fingers") > 0.02
        # using a more robust check
        jugs = state.get_objects(self._jug_type)
        for jug in jugs:
            if self._Holding_holds(state, [self._robot, jug]):
                return False
        return True

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

    def _JugAboveCup_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        jug, cup = objects
        if not self._Holding_holds(state, [self._robot, jug]):
            return False
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        jug_z = state.get(self._robot, "z") -\
            PyBulletCoffeeEnv.jug_handle_height()
        jug_pos = (jug_x, jug_y, jug_z)

        # Find the closest cup to the jug; can only be above one cup at a time
        closest_cup = None
        closest_cup_dist = float("inf")
        for cup_target in state.get_objects(self._cup_type):
            pour_pos = PyBulletCoffeeEnv._get_pour_position(state, cup_target)  # pylint: disable=protected-access
            sq_dist_to_pour = np.sum(np.subtract(jug_pos, pour_pos)**2)
            if sq_dist_to_pour < self.pour_pos_tol and \
                sq_dist_to_pour < closest_cup_dist:
                closest_cup = cup_target
                closest_cup_dist = sq_dist_to_pour
        # Can only be above one cup at a time
        if closest_cup is None or closest_cup != cup:
            return False
        return True

    def _NotAboveCup_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        _, jug = objects
        for cup in state.get_objects(self._cup_type):
            if self._JugAboveCup_holds(state, [jug, cup]):
                return False
        return True

    def _HandTilted_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        robot, = objects
        tilt = np.abs(state.get(robot, "tilt") - self.tilt_ub)
        return tilt < 0.1

    # -------------------------------------------------------------------------
    # Task Generation

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_cups_lst=CFG.grow_num_cups_train,
                               num_jugs_lst=CFG.grow_num_jugs_train,
                               rng=self._train_rng,
                               is_train=True)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_cups_lst=CFG.grow_num_cups_test,
                               num_jugs_lst=CFG.grow_num_jugs_test,
                               rng=self._test_rng,
                               is_train=False)

    def _get_tasks(self,
                   num: int,
                   num_cups_lst: List[int],
                   num_jugs_lst: List[int],
                   rng: np.random.Generator,
                   is_train: bool = False) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num):
            # Determine number of cups for this task
            num_cups = num_cups_lst[rng.choice(len(num_cups_lst))]
            num_jugs = num_jugs_lst[rng.choice(len(num_jugs_lst))]

            # Use only the subset of cups/jugs needed for this task
            cups = self._cups[:num_cups]
            jugs = self._jugs[:num_jugs]
            # We'll create an initial state dictionary
            init_dict = {}

            # Robot at center
            robot_dict = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "roll": self.robot_init_roll,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist
            }
            init_dict[self._robot] = robot_dict

            # Generate all object positions at once, satisfying the constraints.
            object_positions = self._sample_object_positions(rng, jugs, cups)

            jug_colors = []
            # Sample positions and colors for jugs
            for jug_obj in jugs:
                # Make sure we don't sample the same color twice
                while True:
                    c = list(rng.choice(self._obj_colors))
                    if c not in jug_colors:
                        break
                jug_colors.append(c)
                r_col, g_col, b_col, _ = c
                # Get the pre-sampled position
                x, y = object_positions[jug_obj]
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
            for i, cup_obj in enumerate(cups):
                # Get the pre-sampled position
                x, y = object_positions[cup_obj]
                # Sample a color (r, g, b, a)
                if i < len(jug_colors) and is_train:
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
            for cup_obj in cups:
                goal_atoms.add(GroundAtom(self._Grown, [cup_obj]))
                # goal_atoms.add(GroundAtom(self._CupOnTable, [cup_obj]))
            # # plus jugs are on the table
            # for jug_obj in jugs:
            #     goal_atoms.add(GroundAtom(self._JugOnTable, [jug_obj]))

            task = EnvironmentTask(init_state, goal_atoms)
            tasks.append(task)

        return self._add_pybullet_state_to_tasks(tasks)

    # -------------------------------------------------------------------------
    # Sampling helpers
    def _sample_object_positions(
        self,
        rng: np.random.Generator,
        jugs: List[Any],
        cups: List[Any],
    ) -> Dict[Any, Tuple[float, float]]:
        """Samples (x, y) positions for jugs and cups in separate y-regions.

        The x-positions are sampled first to be ordered from left-to-right
        and guaranteed to be `collision_padding` apart.

        - Jug y-positions are sampled from [y_lb, y_mid].
        - Cup y-positions are sampled from [y_mid, y_ub].

        The generated (x, y) coordinates are then randomly assigned to the
        corresponding objects.
        """
        all_objects = jugs + cups
        num_objects = len(all_objects)

        # 1. Generate spaced-out X coordinates for all objects
        total_x_range = self.x_ub - self.x_lb - self.small_padding
        required_padding_space = (num_objects - 1) * self.collision_padding

        if required_padding_space > total_x_range:
            raise ValueError(
                f"Cannot fit {num_objects} objects with padding "
                f"{self.collision_padding} in x-range {total_x_range}.")

        random_x_space = total_x_range - required_padding_space
        x_offsets = np.sort(rng.uniform(0, random_x_space, size=num_objects))

        x_coords = [
            self.x_lb + 0.5 * self.small_padding + x_offsets[i] +
            i * self.collision_padding for i in range(num_objects)
        ]

        # 2. Generate Y coordinates in separate regions for jugs and cups
        jug_y_coords = rng.uniform(self.y_lb + 1.5 * self.small_padding,
                                   self.y_mid,
                                   size=len(jugs))
        cup_y_coords = rng.uniform(self.y_mid + 0.2 * self.small_padding,
                                   self.y_ub - 1.5 * self.small_padding,
                                   size=len(cups))

        # 3. Randomly assign X and Y coordinates to objects
        # Shuffle the x-coordinates to assign them randomly to any object.
        rng.shuffle(x_coords)

        positions = {}
        # Assign a random x and a jug-specific y to each jug
        for i, jug in enumerate(jugs):
            positions[jug] = (x_coords.pop(), jug_y_coords[i])

        # Assign a random x and a cup-specific y to each cup
        for i, cup in enumerate(cups):
            positions[cup] = (x_coords.pop(), cup_y_coords[i])

        return positions

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
        half_extents = (0.03, 0.03, liquid_height / 2)
        cx = state.get(cup, "x")
        cy = state.get(cup, "y")
        cz = self.z_lb + liquid_height / 2  # sits on table

        if CFG.grow_plant_same_color_as_cup:
            color = (state.get(cup, "r"), state.get(cup,
                                                    "g"), state.get(cup,
                                                                    "b"), 0.8)
        else:
            color = growth_color
        return create_pybullet_block(
            color=color,
            half_extents=half_extents,
            #  mass=10.0,
            mass=0.0,
            friction=0.5,
            position=(cx, cy, cz),
            physics_client_id=self._physics_client_id)


if __name__ == "__main__":
    # Run a simple simulation to test the environment.
    import time

    CFG.env = "pybullet_grow"
    CFG.seed = 0
    CFG.pybullet_sim_steps_per_action = 1

    env = PyBulletGrowEnv(use_gui=True)
    _rng = np.random.default_rng(CFG.seed)
    _task = env._get_tasks(  # pylint: disable=protected-access
        1, CFG.grow_num_cups_test, CFG.grow_num_jugs_test, _rng)[0]
    env._set_state(_task.init)  # pylint: disable=protected-access

    while True:
        # Robot does nothing
        _joints = env._pybullet_robot.initial_joint_positions  # pylint: disable=protected-access
        _act = Action(np.array(_joints))
        env.step(_act)
        time.sleep(0.01)
