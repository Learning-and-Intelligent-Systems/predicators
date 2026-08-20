"""Example command:

python predicators/envs/pybullet_boil.py
"""
import random
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers import retry_pybullet_call
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import cap_switch_joint_travel, \
    create_object, create_pybullet_block, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, DerivedPredicate, EnvironmentTask, \
    GroundAtom, Object, Predicate, State, Type


class PyBulletBoilEnv(PyBulletEnv):
    """A PyBullet environment that simulates boiling water in jugs using
    multiple burners and filling water from a faucet.

    - Jugs can be placed under a faucet to be filled with water (blue color).
    - Jugs can be placed on burners to heat water toward a red color.
    - Each burner and the faucet has a corresponding switch that can be toggled.
    - Spillage occurs if there is no jug under the faucet while
      the faucet is on.
    """

    # -------------------------------------------------------------------------
    # Table / workspace config
    # -------------------------------------------------------------------------
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])

    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    y_ub: ClassVar[float] = 1.6
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    x_mid: ClassVar[float] = (x_lb + x_ub) / 2
    y_mid: ClassVar[float] = (y_lb + y_ub) / 2

    # -------------------------------------------------------------------------
    # Robot config
    # -------------------------------------------------------------------------
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    robot_init_y: ClassVar[float] = (y_lb + y_ub) * 0.5
    robot_init_z: ClassVar[float] = z_ub - 0.1
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.65, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2])
    robot_init_tilt: ClassVar[float] = np.pi / 2
    robot_init_wrist: ClassVar[float] = -np.pi / 2

    # -------------------------------------------------------------------------
    # Camera
    # -------------------------------------------------------------------------
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 60
    _camera_pitch: ClassVar[float] = -38
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # -------------------------------------------------------------------------
    jug_height: ClassVar[float] = 0.12
    jug_handle_height: ClassVar[float] = jug_height * 3 / 4
    jug_handle_offset: ClassVar[float] = 0.08
    jug_init_z: ClassVar[float] = table_height + jug_height / 2
    small_gap: ClassVar[float] = 0.05
    burner_x_gap: ClassVar[float] = 3 * small_gap
    burner_y: ClassVar[float] = y_mid - small_gap * 1.1
    faucet_x: ClassVar[float] = x_mid + 6 * small_gap
    faucet_y: ClassVar[float] = y_mid + 5 * small_gap
    faucet_x_len: ClassVar[float] = 0.15
    # Faucet water outlet, expressed as a 2-D offset in the faucet's local
    # frame and mapped to the world by the standard rotation matrix R(rot):
    #     outlet = (faucet_x, faucet_y) + R(rot) @ (local_dx, local_dy)
    # This is the general rotation-matrix parameterization the learned
    # simulators use (their `_faucet_anchor_dist`), rather than the previous
    # single-distance-along-(cos, -sin) special case. The spout points along
    # the faucet's local -x axis, so the along-spout offset is -faucet_x_len;
    # the outlet sits on the spout centerline, so the lateral (local-y) offset
    # is 0. With the faucet's fixed rot = pi/2 this reproduces the original
    # outlet (faucet_x, faucet_y - faucet_x_len).
    faucet_outlet_local_dx: ClassVar[float] = -faucet_x_len
    faucet_outlet_local_dy: ClassVar[float] = 0.0
    switch_y: ClassVar[float] = y_lb + small_gap

    # -------------------------------------------------------------------------
    # Jug sampling boundaries
    # -------------------------------------------------------------------------
    jug_sample_x_margin: ClassVar[float] = 0.05
    jug_sample_y_margin_bot: ClassVar[float] = 0.4  # margin from y_lb
    jug_sample_y_margin_top: ClassVar[float] = 0.05  # margin from y_ub
    jug_sample_x_min: ClassVar[float] = x_mid
    jug_sample_x_max: ClassVar[float] = x_mid + jug_sample_x_margin * 3
    jug_sample_y_min: ClassVar[float] = y_lb + jug_sample_y_margin_bot
    jug_sample_y_max: ClassVar[float] = y_ub - jug_sample_y_margin_top

    # -------------------------------------------------------------------------
    # Domain-specific config
    # -------------------------------------------------------------------------
    # Speeds / rates
    water_height_to_level_ratio: ClassVar[float] = 10
    # how fast water_volume increases per step — read from CFG at runtime
    @property
    def water_fill_speed(self) -> float:
        """Water fill speed."""
        return CFG.boil_water_fill_speed * self.water_height_to_level_ratio

    water_filled_height: ClassVar[float] = 0.08 * water_height_to_level_ratio
    # When capacity is 1, it is harder to learn the right process for
    # WaterSpilled process because for water to spill on the table is happens
    # immediately (after a period t) while for overflow, it takes requires it
    # to first reach the max capacity, then overflow (also after a period t).
    # But when capacity is say 0.083, it wouldn't allow plans that waits after
    # TurnerFaucetOn, which is not the most efficient, but also reasonable.
    # Experiment with capacity at 0.83, if this doesn't allow it to learn
    # the right process, then fallback to 0.95; -> it doesn't
    # Another idea is to change the environment to be that water doesn't
    # over flow. (magic water like in grow)
    max_jug_water_capacity: ClassVar[
        float] = 0.13 * water_height_to_level_ratio
    # float] = 0.093 * water_height_to_level_ratio # the value it get if it
    # wait then TurnOff
    max_water_spill_width: ClassVar[float] = 0.3
    water_color = (0.0, 0.0, 1.0, 0.9)  # blue
    heating_speed: ClassVar[
        float] = 0.03  # how fast the jug's "heat_level" goes up per step
    happy_speed: ClassVar[float] = 0.05

    # Partial-observability projection: bubbling_level = clip(
    #   (heat_level - BUBBLING_THRESHOLD) * BUBBLING_RAMP, 0, 1).
    BUBBLING_THRESHOLD: ClassVar[float] = 0.85
    BUBBLING_RAMP: ClassVar[float] = 1.0 / (1.0 - BUBBLING_THRESHOLD)  # ≈6.67
    # Goal threshold on the observable bubbling_level, used by
    # WaterBoiled in partial-observability mode (where heat_level is
    # not observable). bubbling_level reaches 1.0 exactly when
    # heat_level reaches the fully-observable boil point (1.0), so 0.99
    # fires at essentially the same instant while staying robust to
    # float rounding in the ramp.
    BUBBLING_BOIL_THRESHOLD: ClassVar[float] = 0.99

    # Colors for switches and faucet
    burner_switch_color: ClassVar[Tuple[float, float, float,
                                        float]] = (1.0, 0.5, 0.0, 1.0
                                                   )  # orange
    faucet_switch_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.0, 0.7, 1.0, 1.0
                                                   )  # light blue
    faucet_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.6, 0.6, 1.0)  # gray

    # Burner plate colors
    burner_off_color: ClassVar[Tuple[float, float, float,
                                     float]] = (0.7, 0.7, 0.7, 1.0
                                                )  # gray (off)
    burner_on_color: ClassVar[Tuple[float, float, float,
                                    float]] = (1.0, 0.3, 0.0, 1.0
                                               )  # red-orange (on)

    # Dist thresholds
    faucet_align_threshold: ClassVar[
        float] = 0.1  # if jug is within this distance of faucet
    burner_align_threshold: ClassVar[float] = 0.05
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5  # fraction of the joint range
    switch_height: ClassVar[float] = 0.08

    # We'll store a separate 'heat' feature for jugs in the environment
    # (0.0 => fully cold/blue, 1.0 => fully hot/red).
    # We'll produce a color in step() from that.
    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"])

    # `bubbling_level` is a derived observable: ramp from 0 to 1 as
    # internal heat crosses BUBBLING_THRESHOLD. Present in the State
    # schema in both fully- and partially-observable modes.
    #
    # Two jug types: the fully-observable `_jug_type` carries
    # `heat_level` as an observable feature, while the partially-
    # observable `_jug_type_po` drops it entirely so the agent never
    # sees a feature named `heat_level` (it must infer the hidden
    # heating process from the derived `bubbling_level`). In both,
    # `heat_level` stays a `sim_feature` so the `jug.heat_level` Python
    # attribute — the internal source of truth for the heating
    # dynamics — keeps working. `__init__` swaps `self._jug_type` to
    # the PO variant when `CFG.partially_observable` is set.
    _jug_type = Type("jug", [
        "x", "y", "z", "rot", "is_held", "water_volume", "heat_level",
        "bubbling_level", "r", "g", "b"
    ],
                     sim_features=["id", "heat_level", "water_id"])
    _jug_type_po = Type("jug", [
        "x", "y", "z", "rot", "is_held", "water_volume", "bubbling_level", "r",
        "g", "b"
    ],
                        sim_features=["id", "heat_level", "water_id"])
    _burner_type = Type("burner", ["x", "y", "z", "is_on"],
                        sim_features=["id", "switch_id", "prev_on"])
    _switch_type = Type("switch", ["x", "y", "z", "rot", "is_on"])
    # _spilled_level is initialized to be 0.04 smaller. This creates a delay
    # for spill to occur while allows the WaterSpill predicate to have an
    # intuitive >0.0 definition, instead of >0.04
    _faucet_type = Type(
        "faucet", ["x", "y", "z", "rot", "is_on", "spilled_level"],
        sim_features=["id", "switch_id", "_spilled_level", "prev_on"])
    _human_type = Type("human", ["happiness_level"],
                       sim_features=["id", "happiness_level"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # In partial-observability mode, swap the jug type to the
        # variant without `heat_level` *before* any jugs/predicates are
        # built off `self._jug_type`, so the reduced type propagates to
        # the objects, every jug predicate, the `types` property, and
        # thus the agent-facing inspect tools.
        if CFG.partially_observable:
            self._jug_type = self._jug_type_po

        # Create the robot as an Object
        self._robot = Object("robot", self._robot_type)

        # Create jugs
        self._jugs: List[Object] = []
        max_jugs = max(max(CFG.boil_num_jugs_train),
                       max(CFG.boil_num_jugs_test))
        for i in range(max_jugs):
            jug_obj = Object(f"jug{i}", self._jug_type)
            self._jugs.append(jug_obj)
        self._jug_to_liquid_id: Dict[Object, Optional[int]] = {}

        # Create burners + a corresponding switch for each
        self._burners: List[Object] = []
        self._burner_switches: List[Object] = []
        max_burners = max(max(CFG.boil_num_burner_train),
                          max(CFG.boil_num_burner_test))
        for i in range(max_burners):
            burn_obj = Object(f"burner{i}", self._burner_type)
            self._burners.append(burn_obj)

            sw_obj = Object(f"burner_switch{i}", self._switch_type)
            self._burner_switches.append(sw_obj)

        # Create one faucet + a corresponding switch
        self._faucet = Object("faucet", self._faucet_type)
        self._faucet_switch = Object("faucet_switch", self._switch_type)

        # Create humans - one for each possible jug
        self._humans: List[Object] = []
        max_humans = max_jugs  # Same as max jugs
        for i in range(max_humans):
            human_obj = Object(f"human{i}", self._human_type)
            self._humans.append(human_obj)

        # Keep track of the spilled water block (None if no spill yet)
        self._spilled_water_id: Optional[int] = None

        super().__init__(use_gui, **kwargs)

        # Optionally, define some relevant predicates
        self._JugFilled = Predicate("JugFilled", [self._jug_type],
                                    self._JugFilled_holds)
        self._JugNotFilled = Predicate(
            "JugNotFilled", [self._jug_type],
            lambda s, o: not self._JugFilled_holds(s, o))
        self._JugAtCapacity = Predicate("JugAtCapacity", [self._jug_type],
                                        self._JugAtCapacity_holds)
        self._WaterBoiled = Predicate("WaterBoiled", [self._jug_type],
                                      self._WaterBoiled_holds)
        self._BurnerOn = Predicate("BurnerOn", [self._burner_type],
                                   self._BurnerOn_holds)
        self._FaucetOn = Predicate("FaucetOn", [self._faucet_type],
                                   self._FaucetOn_holds)
        self._BurnerOff = Predicate(
            "BurnerOff", [self._burner_type],
            lambda s, o: not self._BurnerOn_holds(s, o))
        self._FaucetOff = Predicate(
            "FaucetOff", [self._faucet_type],
            lambda s, o: not self._FaucetOn_holds(s, o))
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._jug_type],
                                  self._Holding_holds)
        self._JugAtBurner = Predicate("JugAtBurner",
                                      [self._jug_type, self._burner_type],
                                      self._JugOnBurner_holds)
        self._JugAtFaucet = Predicate("JugAtFaucet",
                                      [self._jug_type, self._faucet_type],
                                      self._JugAtFaucet_holds)
        self._JugNotAtBurnerOrFaucet = Predicate(
            "JugNotAtBurnerOrFaucet", [self._jug_type],
            self._JugNotAtBurnerOrFaucet_holds)
        self._NoJugAtFaucet = Predicate("NoJugAtFaucet", [self._faucet_type],
                                        self._NoJugAtFaucet_holds)
        self._NoJugAtBurner = Predicate("NoJugAtBurner", [self._burner_type],
                                        self._NoJugAtBurner_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._WaterSpilled = Predicate("WaterSpilled", [],
                                       self._WaterSpilled_holds)
        self._NoWaterSpilled = Predicate("NoWaterSpilled", [],
                                         self._NoWaterSpilled_holds)
        self._HumanHappy = Predicate(
            "HumanHappy",
            [self._human_type, self._jug_type, self._burner_type],
            self._HumanHappy_holds)
        self._TaskCompleted = Predicate("TaskCompleted", [],
                                        self._TaskCompleted_holds)
        self._NoJugAtFaucetOrJugAtFaucetAndReachedCapacity = DerivedPredicate(
            "NoJugAtFaucetOrAtFaucetAndReachedCapacity",
            [self._jug_type, self._faucet_type],
            self._NoJugAtFaucetOrJugAtFaucetAndReachedCapacity_holds,
            auxiliary_predicates={
                self._JugAtFaucet, self._JugAtCapacity, self._NoJugAtFaucet
            })
        self._NoJugAtFaucetOrJugAtFaucetAndFilled = DerivedPredicate(
            "NoJugAtFaucetOrAtFaucetAndFilled",
            [self._jug_type, self._faucet_type],
            self._NoJugAtFaucetOrJugAtFaucetAndFilled_holds,
            auxiliary_predicates={
                self._JugAtFaucet, self._JugFilled, self._NoJugAtFaucet
            })

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_boil"

    @property
    def predicates(self) -> Set[Predicate]:
        """Return a set of domain-specific predicates that might be used for
        planning."""
        predicates = {
            self._JugFilled,
            # self._JugNotFilled,
            self._WaterBoiled,
            self._BurnerOn,
            self._FaucetOn,
            self._BurnerOff,
            self._FaucetOff,
            self._Holding,
            self._JugAtBurner,
            self._JugAtFaucet,
            self._JugNotAtBurnerOrFaucet,
            self._HandEmpty,
            # self._WaterSpilled,
            self._NoJugAtFaucet,
            self._NoJugAtBurner,
            self._NoWaterSpilled,
        }
        if CFG.boil_add_jug_reached_capacity_predicate:
            predicates.add(self._JugAtCapacity)
        if CFG.boil_goal == "human_happy":
            predicates.add(self._HumanHappy)
        elif CFG.boil_goal == "task_completed":
            predicates.add(self._TaskCompleted)
        if CFG.boil_use_derived_predicates:
            if CFG.boil_add_jug_reached_capacity_predicate:
                predicates.add(
                    self._NoJugAtFaucetOrJugAtFaucetAndReachedCapacity)
            else:
                predicates.add(self._NoJugAtFaucetOrJugAtFaucetAndFilled)
        return predicates

    @property
    def types(self) -> Set[Type]:
        """All custom types in this environment."""
        return {
            self._robot_type, self._jug_type, self._burner_type,
            self._switch_type, self._faucet_type, self._human_type
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Which predicates might appear in goals."""
        if CFG.boil_goal == "human_happy":
            return {self._HumanHappy}
        if CFG.boil_goal == "task_completed":
            return {self._TaskCompleted}
        if CFG.boil_goal == "simple":
            return {
                self._WaterBoiled, self._JugFilled, self._NoWaterSpilled,
                self._BurnerOff
            }  # Example
        raise ValueError(f"Unknown goal type {CFG.boil_goal}.")

    # @property
    # def agent_goal_predicates(self) -> Set[Predicate]:
    #     return {self._BurnerOff, self._HumanHappy}

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # 1) Create a table
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id
        # add another table for more space to place jugs and burners
        table_id2 = create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0],
                      cls.table_pos[1] + (cls.y_ub - cls.y_lb) / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        bodies["table_id2"] = table_id2

        # 2) Create jugs
        jug_ids = []
        max_jugs = max(max(CFG.boil_num_jugs_train),
                       max(CFG.boil_num_jugs_test))
        all_white_jugs = False
        for _ in range(max_jugs):
            # Example placeholder URDF for a jug
            jug_id = create_object(asset_path="urdf/jug-pixel.urdf",
                                   color=(1, 1, 1, 1) if all_white_jugs else
                                   random.choice(cls._obj_colors_main),
                                   use_fixed_base=False,
                                   physics_client_id=physics_client_id)
            jug_ids.append(jug_id)
        bodies["jug_ids"] = jug_ids

        # 3) Create burners
        burner_ids = []
        max_burners = max(max(CFG.boil_num_burner_train),
                          max(CFG.boil_num_burner_test))
        for _ in range(max_burners):
            burner_id = create_pybullet_block(
                color=cls.burner_off_color,
                half_extents=(0.07, 0.07, 0.0001),
                mass=0,
                friction=0.5,
                physics_client_id=physics_client_id)
            burner_ids.append(burner_id)
        bodies["burner_ids"] = burner_ids

        # 4) Create burner switches
        burner_switch_ids = []
        for _ in range(max_burners):
            switch_id = create_object(
                asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
                scale=1.0,
                use_fixed_base=True,
                physics_client_id=physics_client_id)
            # Color only the base (link -1), not the slider
            p.changeVisualShape(switch_id,
                                -1,
                                rgbaColor=cls.burner_switch_color,
                                physicsClientId=physics_client_id)
            cls._cap_switch_joint_travel(switch_id, physics_client_id)
            burner_switch_ids.append(switch_id)
        bodies["burner_switch_ids"] = burner_switch_ids

        # 5) Create faucet and faucet switch
        faucet_id = create_object(
            asset_path="urdf/partnet_mobility/faucet/1488/mobility.urdf",
            color=cls.faucet_color,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        bodies["faucet_id"] = faucet_id

        faucet_switch_id = create_object(
            asset_path="urdf/partnet_mobility/switch/102812/switch.urdf",
            scale=1.0,
            use_fixed_base=True,
            physics_client_id=physics_client_id)
        # Color only the base (link -1), not the slider
        p.changeVisualShape(faucet_switch_id,
                            -1,
                            rgbaColor=cls.faucet_switch_color,
                            physicsClientId=physics_client_id)
        cls._cap_switch_joint_travel(faucet_switch_id, physics_client_id)
        bodies["faucet_switch_id"] = faucet_switch_id

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to all PyBullet IDs in the environment objects."""
        self._table_ids = [
            pybullet_bodies["table_id"], pybullet_bodies["table_id2"]
        ]
        self._robot.id = self._pybullet_robot.robot_id
        # Jugs
        for i, jug_obj in enumerate(self._jugs):
            jug_obj.id = pybullet_bodies["jug_ids"][i]

        # Burners
        for i, burner_obj in enumerate(self._burners):
            burner_obj.id = pybullet_bodies["burner_ids"][i]

        # Burner switches
        for i, sw_obj in enumerate(self._burner_switches):
            sw_obj.id = pybullet_bodies["burner_switch_ids"][i]

        # Faucet
        self._faucet.id = pybullet_bodies["faucet_id"]
        # Faucet switch
        self._faucet_switch.id = pybullet_bodies["faucet_switch_id"]

        # Get a fresh id for humans
        max_id = float('-inf')
        for value in pybullet_bodies.values():
            if isinstance(value, list):
                for v in value:
                    if isinstance(v, int):
                        max_id = max(max_id, v)
            elif isinstance(value, int):
                max_id = max(max_id, value)

        # Assign IDs to humans
        for i, human_obj in enumerate(self._humans):
            human_obj.id = max_id + 1 + i

        # Draw debug boundary lines if enabled
        if CFG.pybullet_draw_debug:
            self._draw_sampling_boundary_debug_lines()

    # -------------------------------------------------------------------------
    # State Creation / Feature Extraction
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Only jugs can be held in the robot's gripper here."""
        jug_ids = [j.id for j in self._jugs if j.id is not None]
        return jug_ids

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Map from environment object + feature name -> a float feature in the
        State."""
        # Faucet
        if obj.type == self._faucet_type:
            if feature == "is_on":
                return float(self._is_switch_on(self._faucet_switch.id))
            if feature == "spilled_level":
                # Return the environment's internal record
                # (analogous to jug.heat_level).
                # We'll just store it in the object itself
                # (similar to jug.heat_level).
                # If it doesn't exist, default to 0.
                spill = self._faucet._spilled_level  # pylint: disable=protected-access
                return max(0.0, spill)
                # if self._spilled_water_id is None:
                #     return 0.0
                # shape_data = p.getVisualShapeData(
                #     self._spilled_water_id,
                #     physicsClientId=self._physics_client_id)
                # if not shape_data:
                #     return 0.0
                # # shape_data[0][3] is a tuple of the half-extents (x, y, z).
                # # Since it's a square "sheet," take x*2 as side length:
                # half_extents = shape_data[0][3]  # (hx, hy, hz)
                # side_len = half_extents[0] * 2.0
                # return side_len

        # Burner
        elif obj.type == self._burner_type:
            if feature == "is_on":
                idx = int(obj.name.replace("burner", ""))
                sw_obj = self._burner_switches[idx]
                return float(self._is_switch_on(sw_obj.id))

        # Switch
        elif obj.type == self._switch_type:
            if feature == "is_on":
                return float(self._is_switch_on(obj.id))

        # Jug
        elif obj.type == self._jug_type:
            if feature == "water_volume":
                liquid_id = self._jug_to_liquid_id.get(obj, None)
                if liquid_id is not None:
                    shape_data = p.getVisualShapeData(
                        liquid_id, physicsClientId=self._physics_client_id)
                    if shape_data:  # handle the case shape_data might be empty
                        # shape_data[0][3] => half-extents, e.g.
                        # shape_data[0][3][2] is half in z
                        height = shape_data[0][3][2]
                        return height * self.water_height_to_level_ratio
                return 0.0
            if feature == "heat_level":
                return obj.heat_level
            if feature == "bubbling_level":
                # Derived observable only meaningful in PO mode. In
                # fully-observable mode it stays at 0 so existing
                # approaches don't see a phantom observable; in PO
                # mode it ramps from 0 to 1 once internal heat
                # crosses BUBBLING_THRESHOLD.
                if not CFG.partially_observable:
                    return 0.0
                h = obj.heat_level
                if np.isnan(h):  # NaN guard
                    return 0.0
                return float(
                    max(
                        0.0,
                        min(1.0, (h - self.BUBBLING_THRESHOLD) *
                            self.BUBBLING_RAMP)))

        elif obj.type == self._human_type:
            if feature == "happiness_level":
                return obj.happiness_level

        # Otherwise, rely on defaults (like the base PyBulletEnv) for x,y,z,...
        raise ValueError(f"Unknown feature {feature} for object {obj}.")

    def _get_state(self, _render_obs: bool = False) -> State:
        """PyBullet -> State, plus the privileged (hidden) heat block.

        In partially-observable mode `heat_level` is not an observable
        feature, so snapshot each jug's true internal heat into
        ``state.privileged`` — the env-only channel the agent never sees
        (it is excluded from ``feature_names``/``__hash__``/``allclose``
        and from every data-based inspect tool). This keeps the env's
        ground-truth state self-contained per State, so backtracking
        restores each search node's own heat, without exposing it in the
        observation. Fully-observable mode leaves ``privileged`` as None
        (heat is an ordinary observable feature there).
        """
        state = super()._get_state(_render_obs)
        if CFG.partially_observable:
            state.privileged = {
                jug.name: {
                    "heat_level": float(jug.heat_level or 0.0)
                }
                for jug in state.get_objects(self._jug_type)
            }
        return state

    def _set_domain_specific_state(self, state: State) -> None:
        """Called in _set_state to do any environment-specific resetting.

        This environment only supports resetting the state at the
        beginning, because the state dict doesn't include all features
        (e.g., faucet prev_is_on) to reset the simulator state exactly.
        """
        # Programmatically set burner switches on/off
        burners = state.get_objects(self._burner_type)
        for i, burner_obj in enumerate(burners):
            on_val = state.get(burner_obj, "is_on")
            burner_obj.switch_id = self._burner_switches[i].id
            burner_obj.prev_on = 0.0
            self._set_switch_on(self._burner_switches[i].id,
                                bool(on_val > 0.5))

        # Remove existing jug liquid bodies if they exist
        for liquid_id in self._jug_to_liquid_id.values():
            if liquid_id is not None:
                p.removeBody(liquid_id,
                             physicsClientId=self._physics_client_id)
        self._jug_to_liquid_id.clear()

        # Recreate the liquid bodies as needed
        jugs = state.get_objects(self._jug_type)
        for jug in jugs:
            if "heat_level" in jug.type.feature_names:
                # Fully observable: heat_level is an observable feature,
                # so restore the internal attribute directly from it.
                jug.heat_level = state.get(jug, "heat_level")
            else:
                # Partially observable: heat_level is hidden from the
                # observation, so restore the env's true heat from the
                # State's privileged block (an env-only channel the agent
                # never sees). task.init carries each jug's initial heat
                # there, and states from _get_state snapshot the running
                # value, so backtracking restores each node's own heat.
                # Defaults to 0.0 when absent (e.g. a State built without
                # a privileged block).
                priv = state.privileged or {}
                jug.heat_level = float(
                    priv.get(jug.name, {}).get("heat_level", 0.0))
            liquid_id = self._create_liquid_for_jug(jug, state)
            self._jug_to_liquid_id[jug] = liquid_id

        self._update_liquid_colors(state)

        # Update jug body colors from state
        for jug in jugs:
            if jug.id is not None:
                r = state.get(jug, "r")
                g = state.get(jug, "g")
                b = state.get(jug, "b")
                update_object(jug.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

        # Faucet on/off
        self._faucet.switch_id = self._faucet_switch.id
        self._faucet.prev_on = 0.0
        f_on = state.get(self._faucet, "is_on")
        self._set_switch_on(self._faucet_switch.id, bool(f_on > 0.5))

        # Spilled water reset: remove old block if any
        if self._spilled_water_id is not None:
            p.removeBody(self._spilled_water_id,
                         physicsClientId=self._physics_client_id)
            self._spilled_water_id = None

        # Initialize to take 10 steps for spill to occur
        # pylint: disable=protected-access
        self._faucet._spilled_level = -self.water_fill_speed * 20
        spilled_level = max(0.0, self._faucet._spilled_level)
        # pylint: enable=protected-access
        if spilled_level > 0.0:
            self._spilled_water_id = self._create_spilled_water_block(
                spilled_level, state)

        # Human
        humans = state.get_objects(self._human_type)
        for human_obj in humans:
            human_obj.happiness_level = state.get(human_obj, "happiness_level")

        # Move irrelevant jugs and burners out of the way
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(jugs), len(self._jugs)):
            update_object(self._jugs[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)
        for i in range(len(burners), len(self._burners)):
            update_object(self._burners[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)
            update_object(self._burner_switches[i].id,
                          position=(oov_x, oov_y, self.switch_height),
                          physics_client_id=self._physics_client_id)

        # Update burner colors to match their initial on/off state
        self._update_burner_colors(state)

    # -------------------------------------------------------------------------
    # Step Logic
    # -------------------------------------------------------------------------
    def _domain_specific_step(self) -> None:
        """Handle water filling/spillage, heating, and happiness."""
        state = self._get_state()
        self._handle_faucet_logic(state)
        self._handle_heating_logic(state)
        self._update_liquid_colors(state)
        self._update_liquid_positions(state)
        self._update_burner_colors(state)
        self._update_human_happiness(state)
        self._update_prev_on_states(state)

    def _handle_faucet_logic(self, state: State) -> None:
        """If faucet is on, fill any jug that is properly aligned; otherwise,
        grow the spill block on the table.

        Additionally, if a jug is already full (water_volume >=
        self.max_jug_water_capacity) but stays under the faucet, water
        spills.
        """
        faucet_on = self._is_switch_on(self._faucet_switch.id)
        faucet_prev_on = self._faucet.prev_on > 0.5

        # Only process if faucet is on AND it was on in the previous step
        # (transition from off to on)
        if not (faucet_on and faucet_prev_on):
            return

        # Find jugs under the faucet
        jugs = state.get_objects(self._jug_type)
        jugs_under = [
            jug for jug in jugs
            if self._JugAtFaucet_holds(state, [jug, self._faucet])
        ]

        # ----------------------------------------------------------------------
        # NO JUG UNDER FAUCET => SPILL
        # ----------------------------------------------------------------------
        if not jugs_under:
            old_spill = self._faucet._spilled_level  # pylint: disable=protected-access
            self._increment_spillage(old_spill, state)

        # THERE IS AT LEAST ONE JUG UNDER THE FAUCET
        else:
            for jug_obj in jugs_under:
                old_level = state.get(jug_obj, "water_volume")
                if old_level < self.max_jug_water_capacity:
                    # If jug is NOT yet full => fill
                    self._fill_jug_water(jug_obj, old_level, state)
                else:
                    # Jug is already full => overflow
                    old_spill = self._faucet._spilled_level  # pylint: disable=protected-access
                    self._increment_spillage(old_spill, state)

    def _increment_spillage(self, old_spill: float, state: State) -> None:
        """Increment the spilled water level and recreate the PyBullet
        block."""
        _new_spill = min(self.max_water_spill_width,
                         old_spill + self.water_fill_speed)
        self._faucet._spilled_level = _new_spill  # pylint: disable=protected-access
        new_spill = max(0.0, _new_spill)
        state.set(self._faucet, "spilled_level", new_spill)

        # Remove any existing spill block
        if self._spilled_water_id is not None:
            p.removeBody(self._spilled_water_id,
                         physicsClientId=self._physics_client_id)

        # Recreate spill block with updated size
        self._spilled_water_id = self._create_spilled_water_block(
            new_spill, state)

    def _fill_jug_water(self, jug_obj: Object, old_level: float,
                        state: State) -> None:
        """Increment the jug’s water level (up to max) and recreate the liquid
        block."""
        new_level = old_level + self.water_fill_speed
        if new_level > self.max_jug_water_capacity:
            new_level = self.max_jug_water_capacity

        state.set(jug_obj, "water_volume", new_level)

        # Remove old liquid block
        old_liquid_id = self._jug_to_liquid_id.get(jug_obj, None)
        if old_liquid_id is not None:
            p.removeBody(old_liquid_id,
                         physicsClientId=self._physics_client_id)

        # Create new liquid block at updated level
        self._jug_to_liquid_id[jug_obj] = self._create_liquid_for_jug(
            jug_obj, state)

    def _handle_heating_logic(self, state: State) -> None:
        """If a jug with water is on a turned-on burner, increment jug 'heat'
        up to 1.0."""
        # Note: should also check jug is not held by the robot
        burners = state.get_objects(self._burner_type)
        jugs = state.get_objects(self._jug_type)
        for i, burner_obj in enumerate(burners):
            burner_on = self._is_switch_on(self._burner_switches[i].id)
            burner_prev_on = burner_obj.prev_on > 0.5

            # Only process if burner is on AND it wasn't on in the previous step
            # (transition from off to on)
            if not (burner_on and burner_prev_on):
                continue
            bx = state.get(burner_obj, "x")
            by = state.get(burner_obj, "y")
            for jug_obj in jugs:
                jug_x = state.get(jug_obj, "x")
                jug_y = state.get(jug_obj, "y")
                dist = np.hypot(bx - jug_x, by - jug_y)
                if dist < self.burner_align_threshold:
                    # Jug is on top of an active burner => increase heat.
                    # Read the `jug.heat_level` attribute (the internal
                    # source of truth) rather than the State array: in PO
                    # mode `heat_level` is not an observable feature, and
                    # in FO mode the array merely mirrors this attribute.
                    old_heat = jug_obj.heat_level
                    if CFG.boil_require_jug_full_to_heatup:
                        required_vol = self.water_filled_height
                    else:
                        required_vol = 0.0

                    if state.get(jug_obj, "water_volume") > required_vol and\
                        not self._Holding_holds(state, [self._robot, jug_obj]):
                        new_heat = min(1.0, old_heat + self.heating_speed)
                        jug_obj.heat_level = new_heat

    def _update_liquid_colors(self, state: State) -> None:
        """Simple linear interpolation from blue (0.0) to red (1.0) based on
        jug.heat."""
        jugs = state.get_objects(self._jug_type)
        for jug_obj in jugs:
            jug_id = jug_obj.id
            water_id = self._jug_to_liquid_id[jug_obj]
            if jug_id is None or water_id is None:
                continue
            heat = jug_obj.heat_level
            # Weighted interpolation from (0,0,1) => (1,0,0)
            r = heat
            g = 0.0
            b = 1.0 - heat
            alpha = 0.9
            update_object(water_id,
                          color=(r, g, b, alpha),
                          physics_client_id=self._physics_client_id)

    def _update_liquid_positions(self, state: State) -> None:
        """Teleport each liquid body to follow its jug.

        The liquid bodies are visual-only (collision filter mask=0, see
        ``_create_liquid_for_jug``) so they don't get carried by the
        jug's grasp constraint. Re-teleport them each step from the
        jug's current pose so the visualization stays inside the jug
        when the jug is picked up, placed, or rotated.
        """
        for jug_obj in state.get_objects(self._jug_type):
            water_id = self._jug_to_liquid_id.get(jug_obj)
            if water_id is None or jug_obj.id is None:
                continue
            volume = state.get(jug_obj, "water_volume")
            if volume <= 0:
                continue
            cx, cy, cz, orn = self._liquid_pose_for_jug(
                (state.get(jug_obj, "x"), state.get(jug_obj, "y"),
                 state.get(jug_obj, "z"), state.get(jug_obj, "rot")),
                volume,
            )
            p.resetBasePositionAndOrientation(
                water_id, (cx, cy, cz),
                orn,
                physicsClientId=self._physics_client_id)

    def _update_burner_colors(self, state: State) -> None:
        """Update burner plate colors based on their on/off state."""
        burners = state.get_objects(self._burner_type)
        for i, burner_obj in enumerate(burners):
            burner_id = burner_obj.id
            if burner_id is None:
                continue
            burner_on = self._is_switch_on(self._burner_switches[i].id)
            color = self.burner_on_color if burner_on else self.burner_off_color
            update_object(burner_id,
                          color=color,
                          physics_client_id=self._physics_client_id)

    def _update_human_happiness(self, state: State) -> None:
        """Update each human's happiness based on their corresponding jug."""
        humans = state.get_objects(self._human_type)
        jugs = state.get_objects(self._jug_type)
        burners = state.get_objects(self._burner_type)

        # Each human corresponds to a jug by index
        for i, human_obj in enumerate(humans):
            if i < len(jugs):
                jug = jugs[i]
                # Determine which burner this human cares about
                # If more humans than burners, some share the same burner
                burner_idx = i % len(burners) if burners else 0
                burner = burners[burner_idx] if burners else None

                # Check if this human's conditions are met
                jug_filled = self._JugFilled_holds(state, [jug])
                water_boiled = self._WaterBoiled_holds(state, [jug])
                no_water_spilled = self._NoWaterSpilled_holds(state, [])

                conditions = [jug_filled, water_boiled, no_water_spilled]
                if CFG.boil_goal_require_burner_off:
                    burner_off = True  # Default if no burner
                    if burner is not None:
                        burner_off = not self._BurnerOn_holds(state, [burner])
                    conditions.append(burner_off)

                if all(conditions):
                    old_happiness_level = state.get(human_obj,
                                                    "happiness_level")
                    new_happiness_level = min(
                        1.0, old_happiness_level + self.happy_speed)
                    human_obj.happiness_level = new_happiness_level

    def _update_prev_on_states(self, state: State) -> None:
        """Update the prev_on sim_features for burners and faucet to track
        their current on/off state for the next step."""
        # Update burner prev_on states
        burners = state.get_objects(self._burner_type)
        for i, burner_obj in enumerate(burners):
            burner_on = self._is_switch_on(self._burner_switches[i].id)
            burner_obj.prev_on = float(burner_on)

        # Update faucet prev_on state
        faucet_on = self._is_switch_on(self._faucet_switch.id)
        self._faucet.prev_on = float(faucet_on)

    def _faucet_outlet_xy(self, state: State,
                          faucet: Object) -> Tuple[float, float]:
        """World (x, y) of the faucet's water outlet.

        General form shared by the fill check and the spill block:
            outlet = (faucet_x, faucet_y) + R(rot) @ (local_dx, local_dy)
        with R(rot) the standard rotation matrix. Mirrors the learned
        simulators' `_faucet_anchor_dist`.
        """
        faucet_x = state.get(faucet, "x")
        faucet_y = state.get(faucet, "y")
        faucet_rot = state.get(faucet, "rot")
        cos_r, sin_r = np.cos(faucet_rot), np.sin(faucet_rot)
        dx, dy = self.faucet_outlet_local_dx, self.faucet_outlet_local_dy
        output_x = faucet_x + cos_r * dx - sin_r * dy
        output_y = faucet_y + sin_r * dx + cos_r * dy
        return output_x, output_y

    def _create_spilled_water_block(self, spilled_size: float,
                                    state: State) -> int:
        """Create a very short block on the table to represent spilled water.

        The side length is 'spilled_size'.
        """
        # Center the spill where the faucet output is.
        output_x, output_y = self._faucet_outlet_xy(state, self._faucet)

        half_len = spilled_size / 2.0
        # Keep it very thin in Z
        half_extents = (half_len, half_len, 0.001)

        block_id = create_pybullet_block(
            color=(0.0, 0.0, 1.0, 0.5),
            half_extents=half_extents,
            mass=0,
            friction=0.5,
            position=(output_x, output_y, self.table_height),
            physics_client_id=self._physics_client_id)
        return block_id

    # -------------------------------------------------------------------------
    # Switch Helpers
    # -------------------------------------------------------------------------
    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if a switch's main joint is above a threshold."""
        if switch_id < 0:
            return False
        j_id = self._get_joint_id(switch_id, "joint_0",
                                  self._physics_client_id)
        if j_id < 0:
            return False
        j_pos, _, _, _ = retry_pybullet_call(
            p.getJointState,
            switch_id,
            j_id,
            physicsClientId=self._physics_client_id)
        info = retry_pybullet_call(p.getJointInfo,
                                   switch_id,
                                   j_id,
                                   physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, power_on: bool) -> None:
        """Programmatically toggle the switch to on/off by resetting its joint
        state."""
        j_id = self._get_joint_id(switch_id, "joint_0",
                                  self._physics_client_id)
        if j_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              j_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = (j_max if power_on else j_min) * self.switch_joint_scale
        p.resetJointState(switch_id,
                          j_id,
                          target_val,
                          physicsClientId=self._physics_client_id)

    @staticmethod
    def _get_joint_id(obj_id: int,
                      joint_name: str,
                      physics_client_id: int = 0) -> int:
        """Helper to find a joint by name in a URDF."""
        num_joints = retry_pybullet_call(p.getNumJoints,
                                         obj_id,
                                         physicsClientId=physics_client_id)
        for j in range(num_joints):
            info = retry_pybullet_call(p.getJointInfo,
                                       obj_id,
                                       j,
                                       physicsClientId=physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    @classmethod
    def _cap_switch_joint_travel(cls, switch_id: int,
                                 physics_client_id: int) -> None:
        """Cap this env's switch so a push can't over-extend it past "on".

        Resolves ``joint_0`` and delegates to the shared
        :func:`cap_switch_joint_travel` (see its docstring for the why).
        """
        j_id = cls._get_joint_id(switch_id, "joint_0", physics_client_id)
        cap_switch_joint_travel(switch_id, j_id, cls.switch_joint_scale,
                                physics_client_id)

    def _draw_sampling_boundary_debug_lines(self) -> None:
        """Draw debug lines showing the boundaries where objects can be sampled
        in _sample_xy."""
        # Use the class variables for sampling boundaries
        x_min = self.jug_sample_x_min
        x_max = self.jug_sample_x_max
        y_min = self.jug_sample_y_min
        y_max = self.jug_sample_y_max
        z_height = self.table_height + 0.01  # Slightly above table surface

        # Draw a rectangle on the table showing the sampling boundaries
        # Bottom edge (y_min)
        p.addUserDebugLine(
            lineFromXYZ=[x_min, y_min, z_height],
            lineToXYZ=[x_max, y_min, z_height],
            lineColorRGB=[1, 0, 0],  # Red color
            lineWidth=3,
            physicsClientId=self._physics_client_id)

        # Top edge (y_max)
        p.addUserDebugLine(
            lineFromXYZ=[x_min, y_max, z_height],
            lineToXYZ=[x_max, y_max, z_height],
            lineColorRGB=[1, 0, 0],  # Red color
            lineWidth=3,
            physicsClientId=self._physics_client_id)

        # Left edge (x_min)
        p.addUserDebugLine(
            lineFromXYZ=[x_min, y_min, z_height],
            lineToXYZ=[x_min, y_max, z_height],
            lineColorRGB=[1, 0, 0],  # Red color
            lineWidth=3,
            physicsClientId=self._physics_client_id)

        # Right edge (x_max)
        p.addUserDebugLine(
            lineFromXYZ=[x_max, y_min, z_height],
            lineToXYZ=[x_max, y_max, z_height],
            lineColorRGB=[1, 0, 0],  # Red color
            lineWidth=3,
            physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Example Predicates
    # -------------------------------------------------------------------------
    @classmethod
    def _JugFilled_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        (jug, ) = objects
        return state.get(jug, "water_volume") >= cls.water_filled_height

    @classmethod
    def _JugAtCapacity_holds(cls, state: State,
                             objects: Sequence[Object]) -> bool:
        """Jug is at capacity if it has water_volume >=
        max_jug_water_capacity."""
        (jug, ) = objects
        return state.get(jug, "water_volume") >= cls.max_jug_water_capacity

    def _WaterSpilled_holds(self, state: State,
                            _objects: Sequence[Object]) -> bool:
        """Example: say water is spilled if the faucet is on with no jug
        underneath, or if we want any other condition. Modify as needed."""
        # # If faucet is on and no jug is under it, there's spillage
        # faucet_on = self._FaucetOn_holds(state, [self._faucet])
        # no_jug_under = self._NoJugUnderFaucet_holds(state, [self._faucet])
        # if faucet_on and no_jug_under:
        #     return True

        # A hack to achieve spill is 1 step after the faucet is on.
        return state.get(self._faucet,
                         "spilled_level") > 0  # self.water_fill_speed * 20:
        #     return True
        # return False

    def _NoWaterSpilled_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        return not self._WaterSpilled_holds(state, objects)

    @classmethod
    def _WaterBoiled_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        (jug, ) = objects
        if CFG.partially_observable:
            # heat_level is not observable in PO mode; read the derived
            # observable bubbling_level instead (it reaches 1.0 exactly
            # when heat_level hits the boil point).
            bubbling = state.get(jug, "bubbling_level")
            return bubbling >= cls.BUBBLING_BOIL_THRESHOLD
        return state.get(jug, "heat_level") >= 1.0

    @staticmethod
    def _BurnerOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (burner, ) = objects
        return state.get(burner, "is_on") > 0.5

    @staticmethod
    def _FaucetOn_holds(state: State, objects: Sequence[Object]) -> bool:
        (faucet, ) = objects
        return state.get(faucet, "is_on") > 0.5

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        (_robot, jug) = objects
        return state.get(jug, "is_held") > 0.5

    def _JugOnBurner_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        (jug, burner) = objects
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        burner_x = state.get(burner, "x")
        burner_y = state.get(burner, "y")
        dist = np.hypot(jug_x - burner_x, jug_y - burner_y)
        return dist < self.burner_align_threshold

    def _JugAtFaucet_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        (jug, faucet) = objects
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        jug_x = state.get(jug, "x")
        jug_y = state.get(jug, "y")
        output_x, output_y = self._faucet_outlet_xy(state, faucet)
        dist = np.hypot(jug_x - output_x, jug_y - output_y)
        return dist < self.faucet_align_threshold

    def _JugNotAtBurnerOrFaucet_holds(self, state: State,
                                      objects: Sequence[Object]) -> bool:
        """Jug on table but in area outside of burner or faucet."""
        (jug, ) = objects
        if self._Holding_holds(state, [self._robot, jug]):
            return False
        faucets = state.get_objects(self._faucet_type)
        burners = state.get_objects(self._burner_type)
        for faucet in faucets:
            if self._JugAtFaucet_holds(state, [jug, faucet]):
                return False
        for burner in burners:
            if self._JugOnBurner_holds(state, [jug, burner]):
                return False
        return True

    def _NoJugAtFaucet_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        (faucet, ) = objects
        jugs = state.get_objects(self._jug_type)
        for jug in jugs:
            if self._JugAtFaucet_holds(state, [jug, faucet]):
                return False
        return True

    def _NoJugAtBurner_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        (burner, ) = objects
        jugs = state.get_objects(self._jug_type)
        for jug in jugs:
            if self._JugOnBurner_holds(state, [jug, burner]):
                return False
        return True

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        (robot, ) = objects
        jugs = state.get_objects(self._jug_type)
        for jug in jugs:
            if self._Holding_holds(state, [robot, jug]):
                return False
        return True

    def _HumanHappy_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        """A predicate design mainly for experimenting with inventing predicate
        to describe the preimage of effects."""
        # Check if the specific human is happy about their jug and burner
        (human, _jug, _burner) = objects
        return state.get(human, "happiness_level") >= 1.0

    def _task_objective_holds(self, state: State) -> bool:
        """A simple task objective: all jugs are filled, no water spilled, all
        jugs are boiled, and all burners are off."""
        # Only check jugs and burners that are actually in the current state
        jugs_in_state = state.get_objects(self._jug_type)
        burners_in_state = state.get_objects(self._burner_type)

        all_filled = all(
            self._JugFilled_holds(state, [jug]) for jug in jugs_in_state)
        no_spill = self._NoWaterSpilled_holds(state, [])
        all_boiled = all(
            self._WaterBoiled_holds(state, [jug]) for jug in jugs_in_state)
        burner_off = all(not self._BurnerOn_holds(state, [burner])
                         for burner in burners_in_state)
        # logging.debug(f"all_filled: {all_filled}, no_spill: {no_spill}, "
        #               f"all_boiled: {all_boiled}, burner_off: {burner_off}")
        if CFG.boil_goal_simple_human_happy:
            return all_filled
        conditions = [all_filled, no_spill, all_boiled]
        if CFG.boil_goal_require_burner_off:
            conditions.append(burner_off)
        return all(conditions)

    def _robot_at_init_pose(self, state: State) -> bool:
        """Completion is declared when it's at a particular pose (e.g. the
        init)"""
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_z = state.get(self._robot, "z")
        robot_tilt = state.get(self._robot, "tilt")
        robot_wrist = state.get(self._robot, "wrist")
        return (np.isclose(robot_x, self.robot_init_x, atol=1e-1)
                and np.isclose(robot_y, self.robot_init_y, atol=1e-1)
                and np.isclose(robot_z, self.robot_init_z, atol=1e-1)
                and np.isclose(robot_tilt, self.robot_init_tilt, atol=1e-1)
                and np.isclose(robot_wrist, self.robot_init_wrist, atol=1e-1))

    def _TaskCompleted_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        """A task is completed when the robot is at the initial pose and the
        simple task objective holds."""
        del objects
        return self._robot_at_init_pose(state) and \
            self._task_objective_holds(state)

    def _NoJugAtFaucetOrJugAtFaucetAndFilled_holds(
            self, atoms: Set[GroundAtom], objects: Sequence[Object]) -> bool:
        """A jug is not at the faucet, or if it is, it is filled."""
        (jug, faucet) = objects

        no_jug_at_faucet = False
        jug_at_faucet = False
        jug_filled = False

        for atom in atoms:
            if atom.predicate == self._NoJugAtFaucet:
                no_jug_at_faucet = True
            elif atom.predicate == self._JugAtFaucet and \
                atom.objects == [jug, faucet]:
                jug_at_faucet = True
            elif atom.predicate.name in ["JugFilled", "JugIsFull",
                                         "JugFull", "JugHasWater"] and\
                atom.objects == [jug]:
                jug_filled = True

        return no_jug_at_faucet or (jug_at_faucet and jug_filled)

    def _NoJugAtFaucetOrJugAtFaucetAndReachedCapacity_holds(
            self, atoms: Set[GroundAtom], objects: Sequence[Object]) -> bool:
        """A jug is not at the faucet, or if it is, it is filled."""
        (jug, faucet) = objects

        no_jug_at_faucet = False
        jug_at_faucet = False
        jug_filled = False

        for atom in atoms:
            if atom.predicate == self._NoJugAtFaucet:
                no_jug_at_faucet = True
            elif atom.predicate == self._JugAtFaucet and \
                atom.objects == [jug, faucet]:
                jug_at_faucet = True
            elif atom.predicate.name in ["JugReachedCapacity"] and\
                atom.objects == [jug]:
                jug_filled = True

        return no_jug_at_faucet or (jug_at_faucet and jug_filled)

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                possible_num_jugs=CFG.boil_num_jugs_train,
                                possible_num_burners=CFG.boil_num_burner_train,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                possible_num_jugs=CFG.boil_num_jugs_test,
                                possible_num_burners=CFG.boil_num_burner_test,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int, possible_num_jugs: List[int],
                    possible_num_burners: List[int],
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        """Randomly place jugs, burners, faucet, etc.

        for each task.
        """
        tasks = []
        for _ in range(num_tasks):
            # Sample the number of jugs and burners for this task
            num_jugs = rng.choice(possible_num_jugs)
            num_burners = rng.choice(possible_num_burners)
            num_burners = min(num_jugs, num_burners)  # Limit to num_jugs

            init_dict = {}

            # Robot
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

            # For random placements
            used_xy: Set[Tuple[float, float]] = set()
            burner_2_x = self.x_mid - self.small_gap * 6

            # Jugs (only place the number needed for this task)
            for i in range(num_jugs):
                j_obj = self._jugs[i]
                if i == 0:
                    x, y = self._sample_xy(rng, used_xy)
                if i == 1:
                    x, y = burner_2_x, self.burner_y
                color_idx = rng.integers(len(self._obj_colors_main))
                r_col, g_col, b_col, _ = self._obj_colors_main[color_idx]
                init_dict[j_obj] = {
                    "x": x,
                    "y": y,
                    "z": self.jug_init_z,
                    "rot": -np.pi / 2,
                    "is_held": 0.0,
                    "water_volume": 0.0,
                    "heat_level": 0.0,
                    "bubbling_level": 0.0,
                    "r": r_col,
                    "g": g_col,
                    "b": b_col,
                }
                used_xy.add((x, y))

            # Burners (only place the number needed for this task)
            for i in range(num_burners):
                b_obj = self._burners[i]
                burner_x = self.x_mid - self.small_gap -\
                            (i + 0.5) * self.small_gap * 3
                init_dict[b_obj] = {
                    "x": burner_x,
                    "y": self.burner_y,
                    "z": self.table_height,
                    "is_on": 0.0
                }
                # Switch for burner
                sw_obj = self._burner_switches[i]
                init_dict[sw_obj] = {
                    "x": burner_x,
                    "y": self.switch_y,
                    "z": self.table_height,
                    "rot": 0.0,
                    "is_on": 0.0
                }

            # Faucet
            init_dict[self._faucet] = {
                "x": self.faucet_x,
                "y": self.faucet_y,
                "z": self.table_height + 0.15,
                "rot": np.pi / 2,
                "is_on": 0.0,
                "spilled_level": 0.0  # Initialize spillage to 0
            }
            # Faucet switch
            init_dict[self._faucet_switch] = {
                "x": self.faucet_x,
                "y": self.switch_y,
                "z": self.table_height,
                "rot": 0.0,
                "is_on": 0.0
            }
            # Humans - one for each jug used in this task. Only included
            # when the goal references human happiness, so other goal
            # modes don't expose the irrelevant `happiness_level` feature
            # to the agent.
            if CFG.boil_goal == "human_happy":
                for i in range(num_jugs):
                    human_obj = self._humans[i]
                    init_dict[human_obj] = {"happiness_level": 0.0}

            init_state = utils.create_state_from_dict(init_dict)
            if CFG.partially_observable:
                # heat_level is hidden from the observation; carry each
                # jug's initial heat in the privileged block so reset can
                # restore the env's true starting heat (the agent never
                # sees it). This is what lets tasks start with a non-zero
                # hidden heat without leaking it into the observation.
                init_state.privileged = {
                    j.name: {
                        "heat_level": feats["heat_level"]
                    }
                    for j, feats in init_dict.items()
                    if j.type == self._jug_type
                }

            # Example goal: Water boiled, no water spilled, etc.
            goal_atoms = set()
            goal_nl: str

            if CFG.boil_goal == "human_happy":
                # Add goal for each human used in this task
                for i in range(num_jugs):
                    human_obj = self._humans[i]
                    jug_obj = self._jugs[i]
                    # Determine which burner this human cares about
                    burner_idx = i % num_burners if num_burners > 0 else 0
                    burner_obj = self._burners[
                        burner_idx] if num_burners > 0 else self._burners[0]
                    goal_atoms.add(
                        GroundAtom(self._HumanHappy,
                                   [human_obj, jug_obj, burner_obj]))
                goal_nl = ("Make the human happy by serving them boiled "
                           "water — fill a jug at the faucet, heat it on "
                           "the burner until it boils, and turn the burner "
                           "off, all without spilling water.")
            elif CFG.boil_goal == "task_completed":
                goal_atoms.add(GroundAtom(self._TaskCompleted, []))
                goal_nl = ("Complete the boiling task — boil the water in "
                           "the jug.")
            elif CFG.boil_goal == "simple":
                goal_atoms.add(GroundAtom(self._NoWaterSpilled, []))
                # Only add goals for the jugs and burners used in this task
                for i in range(num_jugs):
                    j_obj = self._jugs[i]
                    goal_atoms.add(GroundAtom(self._WaterBoiled, [j_obj]))
                    goal_atoms.add(GroundAtom(self._JugFilled, [j_obj]))
                for i in range(num_burners):
                    b_obj = self._burners[i]
                    goal_atoms.add(GroundAtom(self._BurnerOff, [b_obj]))
                jug_word = "the jug" if num_jugs == 1 else "every jug"
                goal_nl = (f"Boil a full jug of water on the burner without "
                           f"spilling any water, turn the burner off "
                           f"once {jug_word} has finished boiling.")
            else:
                raise ValueError(f"Unknown goal type {CFG.boil_goal}.")

            tasks.append(
                EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl))

        return self._add_pybullet_state_to_tasks(tasks)

    def _sample_xy(self, rng: np.random.Generator,
                   used_xy: Set[Tuple[float, float]]) -> Tuple[float, float]:
        """Sample a random (x,y) on the table that doesn't collide with
        existing objects."""
        for _ in range(1000):
            x = rng.uniform(self.jug_sample_x_min, self.jug_sample_x_max)
            y = rng.uniform(self.jug_sample_y_min, self.jug_sample_y_max)
            if all((np.hypot(x - ux, y - uy) > 0.10) for (ux, uy) in used_xy):
                used_xy.add((x, y))
                return x, y
        raise RuntimeError("Failed to sample a collision-free (x, y).")

    # Vertical offset of the jug's inner-bottom surface below jug.z.
    # The jug-pixel URDF places its base box at z=-0.25 local, so with
    # the default scale=0.2 the base bottom sits 0.06 m below the jug
    # origin and the inner-bottom surface (top of the 0.1 m base box)
    # sits 0.04 m below; add a small clearance so the liquid box
    # doesn't z-fight the base.
    _LIQUID_OFFSET_BELOW_JUG: ClassVar[float] = 0.04

    def _liquid_pose_for_jug(
        self,
        jug_xy_z_rot: Tuple[float, float, float, float],
        water_volume: float,
    ) -> Tuple[float, float, float, Tuple[float, float, float, float]]:
        """Compute the liquid body's world pose given the jug's pose and
        current water_volume.

        Anchored to ``jug.z`` (not the table) so the liquid stays inside
        the jug when the jug is lifted.
        """
        jx, jy, jz, jrot = jug_xy_z_rot
        liquid_height = water_volume / self.water_height_to_level_ratio
        cz = jz - self._LIQUID_OFFSET_BELOW_JUG + liquid_height / 2
        orn = p.getQuaternionFromEuler([0.0, 0.0, jrot])
        return jx, jy, cz, orn

    def _create_liquid_for_jug(
        self,
        jug: Object,
        state: State,
    ) -> Optional[int]:
        """Given the jug's water_volume, create (or None) a small PyBullet body
        to represent the liquid."""
        current_liquid = state.get(jug, "water_volume")
        if current_liquid <= 0:
            return None

        liquid_height = current_liquid / self.water_height_to_level_ratio
        half_extents = (0.03, 0.03, liquid_height / 2)
        jug_xy_z_rot = (state.get(jug, "x"), state.get(jug, "y"),
                        state.get(jug, "z"), state.get(jug, "rot"))
        cx, cy, cz, orientation = self._liquid_pose_for_jug(
            jug_xy_z_rot, current_liquid)

        color = self.water_color
        liquid_id = create_pybullet_block(
            color=color,
            half_extents=half_extents,
            mass=0.01,
            friction=0.5,
            position=(cx, cy, cz),
            orientation=orientation,
            physics_client_id=self._physics_client_id)
        # The liquid block is purely a visualization of the water level.
        # Leaving its collision shape active causes the jug to drift
        # several cm when the body is recreated/repositioned inside the
        # jug (e.g. fill ticks during Wait). Disable collisions so only
        # the visual remains; physics-side it's a ghost.
        p.setCollisionFilterGroupMask(liquid_id,
                                      -1,
                                      collisionFilterGroup=0,
                                      collisionFilterMask=0,
                                      physicsClientId=self._physics_client_id)
        return liquid_id


if __name__ == "__main__":

    def _main() -> None:  # pylint: disable=too-many-locals
        """Run a simple simulation to test the environment."""
        # # pylint: disable=protected-access
        # from predicators.ground_truth_models import \
        #     get_gt_options  # pylint: disable=import-outside-toplevel
        CFG.seed = 0
        CFG.env = "pybullet_boil"
        CFG.pybullet_sim_steps_per_action = 1
        CFG.pybullet_draw_debug = True
        CFG.coffee_use_pixelated_jug = True
        CFG.boil_num_jugs_train = [1]
        CFG.boil_num_jugs_test = [2]
        CFG.boil_num_burner_train = [1]
        CFG.boil_num_burner_test = [2]
        env = PyBulletBoilEnv(use_gui=True)
        rng = np.random.default_rng(CFG.seed)
        tasks = env._make_tasks(1,
                                possible_num_jugs=[1],
                                possible_num_burners=[1],
                                rng=rng)

        # env_options = get_gt_options(env.get_name())
        # pick = utils.get_parameterized_option_by_name(env_options, "PickJug")
        # place_on_burner = utils.get_parameterized_option_by_name(
        #     env_options, "PlaceOnBurner")
        # place_under_faucet = utils.get_parameterized_option_by_name(
        #     env_options, "PlaceUnderFaucet")
        # switch_faucet_on = utils.get_parameterized_option_by_name(
        #     env_options, "SwitchFaucetOn")
        # switch_faucet_off = utils.get_parameterized_option_by_name(
        #     env_options, "SwitchFaucetOff")
        # switch_burner_on = utils.get_parameterized_option_by_name(
        #     env_options, "SwitchBurnerOn")
        # wait_opt = utils.get_parameterized_option_by_name(env_options, "Wait")
        # robot = env._robot
        # jug1 = env._jugs[0]
        # burner1 = env._burners[0]
        # faucet = env._faucet

        # # Keep references to suppress unused-variable warnings
        # _ = (pick, place_on_burner, place_under_faucet, switch_faucet_on,
        #      switch_faucet_off, switch_burner_on, wait_opt, robot, jug1,
        #      burner1, faucet)

        for task in tasks:
            env._set_state(task.init)
            for _ in range(20000):
                action = Action(
                    np.array(env._pybullet_robot.initial_joint_positions))
                env.step(action)
            print("Final state: "
                  f"{env._current_observation.pretty_str()}")

    _main()
