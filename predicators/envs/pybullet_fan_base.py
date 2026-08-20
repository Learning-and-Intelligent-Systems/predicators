"""Observable simulation core of the fan environment.

This module is the fan env's BASE SIM: scene geometry and physical
constants, object and body construction, state read/write, and the
switch/fan mechanics - everything needed to run rigid-body rollouts of
the arena. It deliberately contains NO residual dynamics (how the wind
moves the ball lives in the ``PyBulletFanEnv`` subclass's
``_domain_specific_step``), no task generation, and no predicate /
goal semantics.

That boundary is a visibility contract, enforced structurally rather
than by redaction: when ``CFG.agent_sim_provide_base_sim_source`` is
on, THIS FILE is copied verbatim into the learning agent's sandbox as
reference material ("the robot knows its own simulator"), so the file
the agent reads is byte-identical to the code its base-sim rollouts
execute. Anything that would leak the learning target - the wind force
law and its constants, the task distribution, goal thresholds - must
live in ``pybullet_fan.py`` (the concrete subclass), never here.
"""
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np
import pybullet as p

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import cap_switch_joint_travel, \
    create_object, create_pybullet_block, create_pybullet_sphere, \
    update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Object, State, Type


class PyBulletFanBaseEnv(PyBulletEnv):
    """Sim core of the fan arena: a ball on a walled grid table, four banks of
    fans, and four switches.

    Abstract on purpose - it defines no name, predicates, tasks, or
    domain-specific step, so env discovery skips it; the concrete env
    is ``PyBulletFanEnv``.
    """

    @classmethod
    def get_base_sim_source_files(cls) -> List[str]:
        # This module IS the visible sim core (see the module docstring's
        # visibility contract); pybullet_env.py is the generic engine it
        # is built on. pybullet_fan.py (residual dynamics, task
        # generation, predicates) must never be listed here.
        return [
            "predicators/envs/pybullet_fan_base.py",
            "predicators/envs/pybullet_env.py",
        ]

    # =========================================================================
    # WORKSPACE & ENVIRONMENT CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Table / Workspace Dimensions
    # -------------------------------------------------------------------------
    table_height: ClassVar[float] = 0.4
    table_pos: ClassVar[Pose3D] = (0.75, 1.35, table_height / 2.0)
    table_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])
    table_scale: ClassVar[float] = 1.0
    # Two tables side by side for extra workspace (mirrors pybullet_domino).
    # The second table is offset by +table_width/2 in y.
    table_width: ClassVar[float] = 1.0

    # Workspace bounds
    x_lb: ClassVar[float] = 0.4
    x_ub: ClassVar[float] = 1.1
    y_lb: ClassVar[float] = 1.1
    # Two tables span y in [1.1, 2.1]; y_ub is the upper workspace bound used
    # to clamp the fan-blown ball. Must cover the full grid (up to
    # loc_y_ub = up_fan_y - 0.05 ~= 1.97), so 2.1 (single-table 1.6 would clip
    # the ball at the upper cells). robot_init_y / switch_y below are anchored
    # to the front (y_lb) so they don't drift up into the grid.
    y_ub: ClassVar[float] = 2.1
    z_lb: ClassVar[float] = table_height
    z_ub: ClassVar[float] = 0.75 + table_height / 2
    init_padding: float = 0.05

    # -------------------------------------------------------------------------
    # Grid Layout Configuration
    # -------------------------------------------------------------------------
    # Grid dimensions will be set dynamically based on train/test mode
    pos_gap: ClassVar[float] = 0.08  # Distance between grid positions

    # -------------------------------------------------------------------------
    # Camera Configuration
    # -------------------------------------------------------------------------
    _camera_distance: ClassVar[float] = 1.3
    _camera_yaw: ClassVar[float] = 70
    _camera_pitch: ClassVar[float] = -50
    _camera_target: ClassVar[Tuple[float, float, float]] = (0.75, 1.25, 0.42)

    # =========================================================================
    # ROBOT CONFIGURATION
    # =========================================================================
    robot_init_x: ClassVar[float] = (x_lb + x_ub) * 0.5
    # Front-anchored (robot only reaches the front switches, not the grid).
    robot_init_y: ClassVar[float] = y_lb - 0.02
    robot_init_z: ClassVar[float] = z_ub - 0.3
    robot_base_pos: ClassVar[Pose3D] = (0.75, 0.62, 0.0)
    robot_base_orn: ClassVar[Quaternion] = p.getQuaternionFromEuler(
        [0.0, 0.0, np.pi / 2.0])
    robot_init_tilt: ClassVar[float] = np.pi / 2.0
    robot_init_wrist: ClassVar[float] = -np.pi / 2.0

    # =========================================================================
    # FAN SYSTEM CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Fan Count & Layout
    # -------------------------------------------------------------------------
    num_left_fans: ClassVar[int] = 5
    num_right_fans: ClassVar[int] = 5
    num_back_fans: ClassVar[int] = 5
    num_front_fans: ClassVar[int] = 5

    # -------------------------------------------------------------------------
    # Fan Physical Properties
    # -------------------------------------------------------------------------
    fan_scale: ClassVar[float] = 0.08
    fan_x_len: ClassVar[float] = 0.2 * fan_scale  # Length of fan blades
    fan_y_len: ClassVar[float] = 1.5 * fan_scale  # Width of fan blades
    fan_z_len: ClassVar[float] = 1.5 * fan_scale  # Height of fan base

    # -------------------------------------------------------------------------
    # Fan Positioning
    # -------------------------------------------------------------------------
    left_fan_x: ClassVar[float] = x_lb - fan_x_len * 5
    right_fan_x: ClassVar[float] = x_ub + fan_x_len * 5
    # Front (far) fan row sits at the upper edge of the second table. The two
    # tables together span y in [1.1, 2.1]; keep the fan body just inside that
    # far edge. Deepening the arena this way gives the left/right sides room
    # for 5 evenly-spaced fans, and re-centers the grid (loc_y_mid, the
    # midpoint of down_fan_y/up_fan_y) between the top and bottom fan rows.
    # Both rows are pushed this much further from the robot than the
    # geometry above would otherwise place them. The down row's rotor
    # link reaches ~0.093 behind the switch row, and a SwitchOff push -
    # whose approach waypoint sits at switch_y + approach_distance, on
    # the far side of the switch - wedges the wrist against it for
    # approach distances the params space advertises as legal (0.08
    # stalls, 0.06 clears). Shifting BOTH rows keeps the arena's
    # y-extent the same size and just translates it, spending the
    # headroom at the far edge (y_ub - up_fan_y = 0.08, less the rotor's
    # ~0.015 overhang). Everything downstream - fan_y_lb/ub and the
    # loc_* grid bounds - is derived from these two, so the grid
    # translates with the fans.
    fan_row_y_shift: ClassVar[float] = 0.03
    up_fan_y: ClassVar[float] = 2.02 + fan_row_y_shift
    down_fan_y: ClassVar[float] = y_lb + fan_x_len / 2 + 0.1 + fan_row_y_shift

    # Fan placement boundaries
    fan_y_lb: ClassVar[
        float] = down_fan_y + fan_x_len / 2 + fan_y_len / 2 + 0.01
    fan_y_ub: ClassVar[float] = up_fan_y - fan_x_len / 2 - fan_y_len / 2 - 0.01
    fan_x_lb: ClassVar[
        float] = left_fan_x + fan_x_len / 2 + fan_y_len / 2 + 0.01
    fan_x_ub: ClassVar[
        float] = right_fan_x - fan_x_len / 2 - fan_y_len / 2 - 0.01

    # =========================================================================
    # SWITCH CONFIGURATION
    # =========================================================================
    switch_scale: ClassVar[float] = 1.0
    switch_joint_scale: ClassVar[float] = 0.1
    switch_on_threshold: ClassVar[float] = 0.5  # Fraction of joint range
    switch_x_len: ClassVar[float] = 0.10  # Length of switch
    switch_height: ClassVar[float] = 0.08

    # Switch positioning: front-anchored so the switches stay at the near edge
    # (out of the grid), independent of the workspace upper bound y_ub.
    switch_y: ClassVar[float] = y_lb  # Y position of switches
    switch_base_x: ClassVar[float] = 0.60  # Base X position for first switch
    switch_x_spacing: ClassVar[float] = 0.08  # Spacing between switches

    # =========================================================================
    # OBJECT PHYSICS CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Ball Properties
    # -------------------------------------------------------------------------
    ball_radius: ClassVar[float] = 0.04
    ball_mass: ClassVar[float] = 0.01
    ball_friction: ClassVar[float] = 10.0
    ball_height_offset: ClassVar[float] = ball_radius
    # High linear damping acts as the ball's air/rolling resistance: it
    # sets the terminal speed under a continuously held force. The wind
    # (a held 0.06 N, see PyBulletFanEnv) terminal-velocities at the
    # ~0.00224 m/action free-field speed the domain is tuned around,
    # while staying far enough above the ~0.036 N stiction/seam creep
    # threshold to roll reliably from rest and across the table seam.
    ball_linear_damping: ClassVar[float] = 120.0
    ball_angular_damping: ClassVar[float] = 10.0
    ball_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.0, 0.0, 1.0, 1)

    # -------------------------------------------------------------------------
    # Wall Properties
    # -------------------------------------------------------------------------
    # Obstacle walls
    num_walls: ClassVar[int] = 4
    # wall_x_len: ClassVar[float] = 0.05
    # wall_y_len: ClassVar[float] = 0.04
    wall_x_len: ClassVar[float] = pos_gap - 0.02
    wall_y_len: ClassVar[float] = pos_gap - 0.02
    obstacle_wall_height: ClassVar[float] = 0.02
    # wall_x_len: ClassVar[float] = pos_gap - 0.03
    # wall_y_len: ClassVar[float] = pos_gap - 0.03
    # obstacle_wall_height: ClassVar[float] = 0.01
    wall_rot: ClassVar[float] = 0.0  # can be np.py/2
    wall_mass: ClassVar[float] = 0.0
    wall_friction: ClassVar[float] = 0.0
    wall_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.5, 0.5, 0.5, 1.0)

    # Boundary walls around grid. The walls must clear the ball's
    # equator (center sits ball_radius above the table): with lower
    # walls the ball leans on the wall's TOP EDGE while traveling along
    # a wall-adjacent row, and the slanted edge contact carries part of
    # its weight like a rail - measured 2.3x the free-rolling wind speed
    # (5.35 vs 2.28 mm/step), which breaks the constant-speed process
    # model and the GT simulator. At 0.06 the contact is a plain side
    # touch at the equator and travel speed matches free rolling.
    boundary_wall_height: ClassVar[float] = 0.06
    boundary_wall_thickness: ClassVar[float] = 0.002
    boundary_wall_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.9, 0.9, 0.9, 1)

    # -------------------------------------------------------------------------
    # Target Properties
    # -------------------------------------------------------------------------
    target_thickness: ClassVar[float] = 0.00001
    target_mass: ClassVar[float] = 0.0
    # Match the table's lateral friction: the pad covers a full grid
    # cell that every final approach rolls across, and a slick pad
    # (0.04, vs the table's 0.5) let the ball slide over it at ~2.2x
    # the free-rolling wind speed (5.0 vs 2.28 mm/step), breaking the
    # constant-speed process model and making the ball ping-pong across
    # the target instead of resting on it.
    target_friction: ClassVar[float] = 0.5
    target_color: ClassVar[Tuple[float, float, float, float]] = (0, 1, 0, 1.0)

    # =========================================================================
    # SIMULATION & DEBUG CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Visual/Debug Parameters
    # -------------------------------------------------------------------------
    debug_line_height: ClassVar[float] = 0.2
    debug_line_lifetime: ClassVar[float] = 0.2

    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"],
                       angular_features=["roll", "tilt", "wrist"])
    _fan_type = Type(
        "fan",
        [
            "x",  # fan base x
            "y",  # fan base y
            "z",  # fan base z
            "rot",  # base orientation (Z euler)
            "facing_side",  # 0=left,1=right,2=back,3=front
            "is_on",  # whether the controlling switch is on
        ],
        sim_features=["id", "side_idx", "fan_ids", "joint_ids"],
        angular_features=["rot"])
    # New separate switch type:
    _switch_type = Type(
        "switch",
        [
            "x",
            "y",
            "z",
            "rot",  # switch orientation
            "controls_fan",  # matches fan side
            "is_on",  # is this switch on
        ],
        sim_features=["id", "joint_id", "side_idx"],
        angular_features=["rot"])
    # Blockers. ``x_len``/``y_len``/``z_len`` are the side lengths of the
    # body's WORLD-AXIS-ALIGNED bounding box (not the body frame), so they
    # always pair with the ``x``/``y``/``z`` pose features: a collision rule
    # can write ``abs(bx - wx) < w.x_len / 2 + reach`` without consulting
    # ``rot``. For a box at rot = 0 or +/-pi/2 (the only rotations this env
    # uses; see ``wall_rot``) the AABB is exact.
    #
    # ``wall`` is the task's obstacle walls; ``boundary`` is the four slabs
    # enclosing the grid. They are deliberately DISTINCT types rather than
    # one type or a hierarchy: the dynamics treat them identically (one
    # contact rule iterating both), while predicates and NSRTs quantify over
    # ``wall`` alone, so the four always-present, never-manipulable boundary
    # slabs never enter symbolic grounding.
    _wall_type = Type("wall",
                      ["x", "y", "z", "rot", "x_len", "y_len", "z_len"],
                      angular_features=["rot"])
    # The boundary extents are task-dependent (they track the grid size), so
    # unlike the obstacle walls they cannot come from a class constant. They
    # are cached in sim_data by _reposition_boundary_walls, which is the same
    # code that writes them into PyBullet - so _get_state reads back exactly
    # the geometry the ball is colliding with.
    _boundary_type = Type("boundary",
                          ["x", "y", "z", "rot", "x_len", "y_len", "z_len"],
                          sim_features=["id", "x_len", "y_len", "z_len"],
                          angular_features=["rot"])
    # ``radius`` completes the contact geometry: with it and the blocker
    # extents above, the ball's stop distance is pure geometry over
    # observable features instead of a fitted constant.
    _ball_type = Type("ball", ["x", "y", "z", "radius"])
    _target_type = Type("target", ["x", "y", "z", "rot", "is_hit"],
                        angular_features=["rot"])

    @classmethod
    def get_configuration_dict(cls) -> Dict[str, Any]:
        """Return all configuration parameters as a dictionary."""
        config = {}

        # Get all ClassVar attributes
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and hasattr(cls, attr_name):
                attr_value = getattr(cls, attr_name)
                if isinstance(attr_value, (int, float, str, tuple, list)):
                    config[attr_name] = attr_value

        return config

    # -------------------------------------------------------------------------
    # Environment initialization
    # -------------------------------------------------------------------------
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._robot = Object("robot", self._robot_type)

        # Fans and switches - one object per side (left=0, right=1,
        # down=2, up=3).
        self._switch_sides = ["left", "right", "down", "up"]
        self._fans: List[Object] = [
            Object(f"fan_{i}", self._fan_type)
            for i in range(len(self._switch_sides))
        ]
        self._switches: List[Object] = [
            Object(f"switch_{i}", self._switch_type)
            for i in range(len(self._switch_sides))
        ]

        # Maze walls - create enough for the maximum walls per task
        max_walls_per_task = max(max(CFG.fan_train_num_walls_per_task),
                                 max(CFG.fan_test_num_walls_per_task))
        self._walls = [
            Object(f"wall{i}", self._wall_type)
            for i in range(max_walls_per_task)
        ]

        # Boundary slabs enclosing the grid. Unlike the obstacle walls these
        # are always present, one per side, named after the same directions
        # the fans/switches use (left/right/down/up).
        self._boundary_sides = ["left", "right", "down", "up"]
        self._boundaries = [
            Object(f"boundary_{side}", self._boundary_type)
            for side in self._boundary_sides
        ]

        # Ball
        self._ball = Object("ball", self._ball_type)

        # Target
        self._target = Object("target", self._target_type)

        super().__init__(use_gui=use_gui, **kwargs)

    @property
    def types(self) -> Set[Type]:
        # Physical-only types (agent runs grid-free). The grid helper types
        # (loc / side) are provided by PyBulletFanGroundTruthTypeFactory and
        # injected only for the oracle / process-planning approaches.
        return {
            self._robot_type, self._fan_type, self._switch_type,
            self._wall_type, self._boundary_type, self._ball_type,
            self._target_type
        }

    # -------------------------------------------------------------------------
    # PyBullet Initialization
    # -------------------------------------------------------------------------
    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        # Two tables side by side for extra workspace (mirrors
        # pybullet_domino). The second table is offset by +table_width/2 in y.
        table_id = create_object(
            asset_path="urdf/table.urdf",
            position=cls.table_pos,
            orientation=cls.table_orn,
            scale=cls.table_scale,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id"] = table_id
        table_id2 = create_object(
            asset_path="urdf/table.urdf",
            position=(cls.table_pos[0], cls.table_pos[1] + cls.table_width / 2,
                      cls.table_pos[2]),
            orientation=cls.table_orn,
            scale=cls.table_scale,
            use_fixed_base=True,
            physics_client_id=physics_client_id,
        )
        bodies["table_id2"] = table_id2

        # ---------------------------------------------------------------------
        # Create fans in four groups: left, right, back, front
        # We'll store them in the dictionary as fan_ids_left, fan_ids_right, ...
        # ---------------------------------------------------------------------
        fan_urdf = "urdf/partnet_mobility/fan/101450/mobility.urdf"

        left_fan_ids = []
        for _ in range(cls.num_left_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            left_fan_ids.append(fid)

        right_fan_ids = []
        for _ in range(cls.num_right_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            right_fan_ids.append(fid)

        back_fan_ids = []
        for _ in range(cls.num_back_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            back_fan_ids.append(fid)

        front_fan_ids = []
        for _ in range(cls.num_front_fans):
            fid = create_object(asset_path=fan_urdf,
                                scale=cls.fan_scale,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            front_fan_ids.append(fid)

        bodies["fan_ids_left"] = left_fan_ids
        bodies["fan_ids_right"] = right_fan_ids
        bodies["fan_ids_back"] = back_fan_ids
        bodies["fan_ids_front"] = front_fan_ids

        # ---------------------------------------------------------------------
        # Create 4 switches at the requested positions
        #   order: left=0, right=1, back=2, front=3
        # ---------------------------------------------------------------------
        switch_urdf = "urdf/partnet_mobility/switch/102812/switch.urdf"
        switch_ids = []
        for _ in range(4):
            sid = create_object(
                asset_path=switch_urdf,
                # position=(sx, sy, cls.table_height),
                # orientation=p.getQuaternionFromEuler(
                #     [0, 0, srot]),
                scale=cls.switch_scale,
                use_fixed_base=True,
                physics_client_id=physics_client_id)
            switch_ids.append(sid)
        bodies["switch_ids"] = switch_ids

        # ---------------------------------------------------------------------
        # Maze walls
        # ---------------------------------------------------------------------
        max_walls_per_task = max(max(CFG.fan_train_num_walls_per_task),
                                 max(CFG.fan_test_num_walls_per_task))
        wall_ids = []
        for _ in range(max_walls_per_task):
            wall_id = create_pybullet_block(
                color=cls.wall_color,
                half_extents=(cls.wall_x_len / 2, cls.wall_y_len / 2,
                              cls.obstacle_wall_height / 2),
                mass=cls.wall_mass,
                friction=cls.wall_friction,
                position=(0.75, 1.28,
                          cls.table_height + cls.obstacle_wall_height / 2),
                orientation=p.getQuaternionFromEuler([0, 0, 0]),
                physics_client_id=physics_client_id)
            wall_ids.append(wall_id)
        bodies["wall_ids"] = wall_ids

        # ---------------------------------------------------------------------
        # Create the ball
        # ---------------------------------------------------------------------
        ball_id = create_pybullet_sphere(
            color=cls.ball_color,
            radius=cls.ball_radius,
            mass=cls.ball_mass,
            friction=cls.ball_friction,
            # Match lateral with spinning so the ball resists rotating around
            # the contact normal — necessary for it to "stick" where the fan
            # parks it instead of pinwheeling.
            spinning_friction=cls.ball_friction,
            position=(0.75, 1.35, cls.table_height + cls.ball_height_offset),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        p.changeDynamics(ball_id,
                         -1,
                         linearDamping=cls.ball_linear_damping,
                         angularDamping=cls.ball_angular_damping,
                         physicsClientId=physics_client_id)
        bodies["ball_id"] = ball_id

        # ---------------------------------------------------------------------
        # Create the target
        # ---------------------------------------------------------------------
        target_id = create_pybullet_block(
            color=(0, 1, 0, 1.0),
            half_extents=(cls.pos_gap / 2, cls.pos_gap / 2,
                          cls.target_thickness),
            mass=cls.target_mass,
            friction=cls.target_friction,
            position=(0, 0, cls.table_height),
            orientation=p.getQuaternionFromEuler([0, 0, 0]),
            physics_client_id=physics_client_id)
        # Match the table's rolling friction (create_pybullet_block only
        # sets lateral). The pad covers the full target cell and the
        # ball ROLLS ON TOP of it; with zero rolling resistance its
        # steady-state wind speed triples there (6.7 vs 2.28 mm/step),
        # so it shoots across the target instead of resting on it.
        p.changeDynamics(target_id,
                         -1,
                         rollingFriction=0.001,
                         physicsClientId=physics_client_id)
        bodies["target_id"] = target_id

        return physics_client_id, pybullet_robot, bodies

    @staticmethod
    def _get_joint_id(obj_id: int,
                      joint_name: str,
                      physics_client_id: int = 0) -> int:
        num_joints = p.getNumJoints(obj_id, physicsClientId=physics_client_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j, physicsClientId=physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store references to all PyBullet object IDs and their joints."""
        self._table_ids = [
            pybullet_bodies["table_id"], pybullet_bodies["table_id2"]
        ]
        # 0 = left, 1 = right, 2 = back, 3 = front

        # Store all fan IDs grouped by side
        fan_ids_by_side = [
            pybullet_bodies["fan_ids_left"],  # side 0
            pybullet_bodies["fan_ids_right"],  # side 1
            pybullet_bodies["fan_ids_back"],  # side 2
            pybullet_bodies["fan_ids_front"]  # side 3
        ]

        # Update each fan object with its side's fan IDs and joint IDs
        for side_idx, fan_obj in enumerate(self._fans):
            fan_obj.side_idx = side_idx
            fan_obj.fan_ids = fan_ids_by_side[side_idx]
            fan_obj.joint_ids = [
                self._get_joint_id(fid, "joint_0", self._physics_client_id)
                for fid in fan_obj.fan_ids
            ]
            # Assign an arbitrary ID from the fans on this side (use the first
            # one)
            fan_obj.id = fan_obj.fan_ids[0] if fan_obj.fan_ids else -1

        # Switches
        for i, switch_obj in enumerate(self._switches):
            switch_obj.id = pybullet_bodies["switch_ids"][i]
            switch_obj.joint_id = self._get_joint_id(switch_obj.id, "joint_0",
                                                     self._physics_client_id)
            cap_switch_joint_travel(switch_obj.id, switch_obj.joint_id,
                                    self.switch_joint_scale,
                                    self._physics_client_id)
            switch_obj.side_idx = i  # 0=left,1=right,2=back,3=front

        for wall, obj_id in zip(self._walls, pybullet_bodies["wall_ids"]):
            wall.id = obj_id
        self._ball.id = pybullet_bodies["ball_id"]
        self._target.id = pybullet_bodies["target_id"]

        # Boundary slab bodies, parallel to self._boundaries. They are
        # rebuilt from the state (a box collision shape cannot be resized
        # in place) by _reposition_boundary_walls, which also refreshes the
        # Objects' ids.
        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids: List[int] = []
        # The (pose, extents) spec the current bodies were built for; lets
        # _reposition_boundary_walls skip an identical rebuild.
        self._boundary_wall_spec: Optional[Tuple[Tuple[float, ...],
                                                 ...]] = None

    # -------------------------------------------------------------------------
    # Read state from PyBullet
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        return []

    def _set_domain_specific_state(self, state: State) -> None:
        for switch_obj in self._switches:
            want_on = bool(state.get(switch_obj, "is_on") > 0.5)
            # Only reconcile a lever whose on/off reading actually
            # disagrees with the requested one. `is_on` is a threshold
            # over a continuous joint (see _is_switch_on), while
            # _set_switch_on snaps the joint to its travel *limit* - so
            # re-imposing an already-matching value teleports the lever
            # out from under a gripper that is mid-push and discards the
            # contact. _set_state runs on every step of a combined
            # base+learned simulator rollout (the learned rules edit
            # features the engine also holds, so State.allclose misses),
            # which let a jammed SwitchOn converge in the belief sim
            # while it stalled for real.
            if self._is_switch_on(switch_obj.id) != want_on:
                self._set_switch_on(switch_obj.id, want_on)

        # Position all fans correctly based on their side
        self._position_fans_on_sides()

        # Rebuild the boundary slabs from their own state features.
        self._reposition_boundary_walls(state)

        oov_x, oov_y = self._out_of_view_xy
        # Move irrelavent walls oov
        wall_obj = state.get_objects(self._wall_type)
        for i in range(len(wall_obj), len(self._walls)):
            update_object(self._walls[i].id,
                          position=(oov_x, oov_y, 0.0),
                          physics_client_id=self._physics_client_id)

    def _reset_single_object(self, obj: Object, state: State) -> None:
        """Skip the boundary slabs; they are rebuilt, not teleported.

        A box collision shape cannot be resized in place, so
        _reposition_boundary_walls destroys and recreates the boundary
        bodies from the state (refreshing the Objects' ids). It runs
        from _set_domain_specific_state, i.e. *after* this generic pose
        reset - so teleporting them here would dereference an id
        belonging to a body the previous rebuild already removed.
        """
        if obj.type == self._boundary_type:
            return
        super()._reset_single_object(obj, state)

    def _remove_boundary_walls(self) -> None:
        """Tear down the current boundary slab bodies."""
        for wall_id in self._boundary_wall_ids:
            if wall_id >= 0:
                p.removeBody(wall_id, physicsClientId=self._physics_client_id)
        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids = []
        self._boundary_wall_spec = None
        for boundary_obj in self._boundaries:
            boundary_obj.id = None

    @staticmethod
    def _body_dims_from_aabb(x_len: float, y_len: float, z_len: float,
                             rot: float) -> Tuple[float, float, float]:
        """Body-frame side lengths of a box whose world AABB is the input.

        The blocker types publish world-axis-aligned extents (see
        ``_wall_type``), but PyBullet needs body-frame half-extents. The
        inversion is exact for the only rotations this env uses
        (multiples of pi/2): a quarter turn just swaps x and y.
        """
        if abs(np.sin(rot)) > 0.5:  # +/-pi/2
            return (y_len, x_len, z_len)
        return (x_len, y_len, z_len)

    @classmethod
    def _aabb_from_body_dims(cls, x_len: float, y_len: float, z_len: float,
                             rot: float) -> Tuple[float, float, float]:
        """World AABB side lengths of a box with the given body dims.

        Inverse of ``_body_dims_from_aabb`` (the map is an involution).
        """
        return cls._body_dims_from_aabb(x_len, y_len, z_len, rot)

    def _reposition_boundary_walls(self, state: State) -> None:
        """Rebuild the boundary slab bodies from their state features.

        The four ``boundary`` objects carry their own pose and extents, so
        this is a straight state -> PyBullet write with no grid inference:
        the arena geometry the ball collides with is exactly what the agent
        observes.

        No-op when the requested spec is unchanged. That matters because
        _set_state runs on every step of a combined base+learned simulator
        rollout, and a box collision shape cannot be resized in place - the
        rebuild removes bodies, discarding any contact they were part of.
        """
        present = [b for b in self._boundaries if b in state]
        if not present:
            # A state with no boundary objects describes an open arena.
            self._remove_boundary_walls()
            return

        spec = tuple(
            tuple(
                float(state.get(b, f))
                for f in ("x", "y", "z", "rot", "x_len", "y_len", "z_len"))
            for b in present)
        if self._boundary_wall_ids and spec == self._boundary_wall_spec:
            return
        self._remove_boundary_walls()

        wall_ids = []
        for boundary_obj, (bx, by, bz, brot, x_len, y_len,
                           z_len) in zip(present, spec):
            dims = self._body_dims_from_aabb(x_len, y_len, z_len, brot)
            wall_id = create_pybullet_block(
                color=self.boundary_wall_color,
                half_extents=(dims[0] / 2, dims[1] / 2, dims[2] / 2),
                mass=self.wall_mass,
                friction=self.wall_friction,
                position=(bx, by, bz),
                orientation=p.getQuaternionFromEuler([0.0, 0.0, brot]),
                physics_client_id=self._physics_client_id)
            boundary_obj.id = wall_id
            boundary_obj.x_len = x_len
            boundary_obj.y_len = y_len
            boundary_obj.z_len = z_len
            wall_ids.append(wall_id)

        # pylint: disable=attribute-defined-outside-init
        self._boundary_wall_ids = wall_ids
        self._boundary_wall_spec = spec

    def _position_fans_on_sides(self) -> None:
        """Position all PyBullet fan bodies correctly on their respective
        sides."""
        # Calculate positions for each side. Back/front fans span the arena's
        # x-extent (fan_x_lb..fan_x_ub); left/right fans span the y-extent
        # (fan_y_lb..fan_y_ub), i.e. corner-to-corner between the bottom and
        # top fan rows. With the deepened arena these bands are long enough for
        # 5 evenly-spaced, non-overlapping fans on every side.
        left_coords = np.linspace(self.fan_y_lb, self.fan_y_ub,
                                  self.num_left_fans)
        right_coords = np.linspace(self.fan_y_lb, self.fan_y_ub,
                                   self.num_right_fans)
        front_coords = np.linspace(self.fan_x_lb, self.fan_x_ub,
                                   self.num_front_fans)
        back_coords = np.linspace(self.fan_x_lb, self.fan_x_ub,
                                  self.num_back_fans)

        # Position fans for each side
        for fan_obj in self._fans:
            side_idx = fan_obj.side_idx
            fan_ids = fan_obj.fan_ids

            if side_idx == 0:  # left
                for i, fan_id in enumerate(fan_ids):
                    px = self.left_fan_x
                    py = left_coords[i] if i < len(
                        left_coords) else left_coords[-1]
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, 0.0]  # facing right
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 1:  # right
                for i, fan_id in enumerate(fan_ids):
                    px = self.right_fan_x
                    py = right_coords[i] if i < len(
                        right_coords) else right_coords[-1]
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, np.pi]  # facing left
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 2:  # back
                for i, fan_id in enumerate(fan_ids):
                    px = back_coords[i] if i < len(
                        back_coords) else back_coords[-1]
                    py = self.down_fan_y
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, np.pi / 2]  # facing forward
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

            elif side_idx == 3:  # front
                for i, fan_id in enumerate(fan_ids):
                    px = front_coords[i] if i < len(
                        front_coords) else front_coords[-1]
                    py = self.up_fan_y
                    pz = self.table_height + self.fan_z_len / 2
                    rot = [0.0, 0.0, -np.pi / 2]  # facing backward
                    update_object(fan_id,
                                  position=(px, py, pz),
                                  orientation=p.getQuaternionFromEuler(rot),
                                  physics_client_id=self._physics_client_id)

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Extract features for creating the State object."""
        if obj.type == self._ball_type:
            if feature == "radius":
                return self.ball_radius
        if obj.type == self._wall_type and feature in ("x_len", "y_len",
                                                       "z_len"):
            # Obstacle walls are all built from the same class constants
            # (see initialize_pybullet); only their yaw varies.
            rot = p.getEulerFromQuaternion(
                p.getBasePositionAndOrientation(
                    obj.id, physicsClientId=self._physics_client_id)[1])[2]
            dims = self._aabb_from_body_dims(self.wall_x_len, self.wall_y_len,
                                             self.obstacle_wall_height, rot)
            return dims[("x_len", "y_len", "z_len").index(feature)]
        if obj.type == self._boundary_type and feature in ("x_len", "y_len",
                                                           "z_len"):
            # Cached by _reposition_boundary_walls when it built the body.
            cached = getattr(obj, feature)
            if cached is None:
                raise ValueError(
                    f"Boundary {obj.name} has no body yet; "
                    f"_reposition_boundary_walls must run before _get_state.")
            return float(cached)
        if obj.type == self._fan_type:
            if feature == "facing_side":
                return float(obj.side_idx)
            if feature == "is_on":
                controlling_switch = self._switches[obj.side_idx]
                return float(self._is_switch_on(controlling_switch.id))
        if obj.type == self._switch_type:
            if feature == "controls_fan":
                return float(obj.side_idx)
            if feature == "is_on":
                return float(self._is_switch_on(obj.id))
        # target.is_hit is computed by the concrete subclass: its
        # proximity threshold is goal semantics, which this module's
        # visibility contract keeps out of the base sim.
        raise ValueError(f"Unknown feature {feature} for object {obj}")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _is_switch_on(self, switch_id: int) -> bool:
        """Check if a switch's joint is above the threshold."""
        joint_id = self._get_joint_id(switch_id, "joint_0",
                                      self._physics_client_id)
        if joint_id < 0:
            return False
        j_pos, _, _, _ = p.getJointState(
            switch_id, joint_id, physicsClientId=self._physics_client_id)
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        frac = (j_pos / self.switch_joint_scale - j_min) / (j_max - j_min)
        return bool(frac > self.switch_on_threshold)

    def _set_switch_on(self, switch_id: int, power_on: bool) -> None:
        """Programmatically toggle a switch on/off."""
        joint_id = self._get_joint_id(switch_id, "joint_0",
                                      self._physics_client_id)
        if joint_id < 0:
            return
        info = p.getJointInfo(switch_id,
                              joint_id,
                              physicsClientId=self._physics_client_id)
        j_min, j_max = info[8], info[9]
        target_val = j_max if power_on else j_min
        p.resetJointState(
            switch_id,
            joint_id,
            target_val * self.switch_joint_scale,
            physicsClientId=self._physics_client_id,
        )
