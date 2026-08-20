"""The fan environment: hidden dynamics, tasks, and predicates.

The observable simulation core (scene geometry, body construction,
state read/write, switch mechanics) lives in
:mod:`predicators.envs.pybullet_fan_base`, which may be surfaced to
learning agents as reference source. This module holds everything an
agent must LEARN or must not see:

* the wind residual dynamics (``_domain_specific_step`` and its
  constants) - the learning target of the sim-learning experiments;
* task generation (the train/test distribution);
* predicates and goal semantics (their thresholds are what predicate
  invention rediscovers).
"""
from collections import deque
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.code_sim_learning.commands import ApplyForce
from predicators.envs.pybullet_fan_base import PyBulletFanBaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class PyBulletFanEnv(PyBulletFanBaseEnv):
    """A PyBullet environment where a ball is blown around by fans in a maze.

    Subclass of the observable sim core (see
    :mod:`predicators.envs.pybullet_fan_base`); this class adds the
    hidden wind dynamics, task generation, and predicates.
    """

    # -------------------------------------------------------------------------
    # Fan Motor & Physics
    # -------------------------------------------------------------------------
    fan_spin_velocity: ClassVar[float] = 100.0  # Velocity for joint_0
    # Wind force on the ball (N): a continuous force held across every
    # physics substep of the action after emission (a held-mode
    # ApplyForce, see _simulate_fans_dynamic), like real wind. The
    # magnitude is calibrated jointly with the ball's linear damping
    # (see PyBulletFanBaseEnv.ball_linear_damping): it sits ~60% above
    # the ~0.036 N stiction/seam creep threshold, and the damping
    # brings its terminal speed to the constant ~0.00224 m/action
    # free-field rate the domain is tuned around.
    wind_force_magnitude: ClassVar[float] = 0.06
    joint_motor_force: ClassVar[float] = 20.0  # Motor control force

    # -------------------------------------------------------------------------
    # Kinematic Ball Movement
    # -------------------------------------------------------------------------
    kinematic_ball_speed: ClassVar[
        float] = 0.003  # Speed for kinematic movement (m/s per simulation step)

    # -------------------------------------------------------------------------
    # Task Generation Parameters
    # -------------------------------------------------------------------------
    # num_walls_per_task will be set dynamically based on train/test mode
    position_tolerance: ClassVar[float] = 0.01

    # =========================================================================
    # DERIVED/CALCULATED VALUES
    # =========================================================================
    # Grid bounds. Derived from base scene geometry, but that tasks
    # place cells on a pos_gap grid inside these bounds is part of the
    # hidden task distribution, so they live here, not in the base sim.
    loc_y_lb = PyBulletFanBaseEnv.down_fan_y + 0.05
    loc_y_ub = PyBulletFanBaseEnv.up_fan_y - 0.05
    loc_x_lb = PyBulletFanBaseEnv.left_fan_x + 0.05
    loc_x_ub = PyBulletFanBaseEnv.right_fan_x - 0.05
    loc_x_mid = (loc_x_lb + loc_x_ub) * 0.5
    loc_y_mid = (loc_y_lb + loc_y_ub) * 0.5

    # -------------------------------------------------------------------------
    # Oracle helper types (grid cells / directions, injected only for
    # the oracle by PyBulletFanGroundTruthTypeFactory - the agent-visible
    # state is grid-free)
    # -------------------------------------------------------------------------
    _location_type = Type("loc", ["xx", "yy"], sim_features=["id", "xx", "yy"])
    _side_type = Type("side", ["side_idx"], sim_features=["id", "side_idx"])

    # -------------------------------------------------------------------------
    # Environment initialization
    # -------------------------------------------------------------------------
    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # Side helper objects (left/right/down/up), injected only for
        # the oracle. Created before the base __init__ because
        # _store_pybullet_bodies (called from it) assigns their
        # side_idx sim-features.
        self._sides: List[Object] = [
            Object(side_str, self._side_type)
            for side_str in ["left", "right", "down", "up"]
        ]
        super().__init__(use_gui=use_gui, **kwargs)

        # Define new predicates if desired
        self._FanOn = Predicate(
            "FanOn", [self._fan_type],
            self._FanOn_holds,
            natural_language_assertion=lambda os: f"fan {os[0]} is on")
        self._FanOff = Predicate(
            "FanOff", [self._fan_type],
            lambda s, o: not self._FanOn_holds(s, o),
            natural_language_assertion=lambda os: f"fan {os[0]} is off")
        self._SwitchOn = Predicate("SwitchOn", [self._switch_type],
                                   self._FanOn_holds)
        self._SwitchOff = Predicate("SwitchOff", [self._switch_type],
                                    lambda s, o: not self._FanOn_holds(s, o))
        # Physical goal predicate: the ball has reached the physical target.
        # The grid helper predicates (BallAtLoc / ClearLoc / SideOf /
        # FanFacingSide / OppositeFan) now live in
        # ground_truth_models/fan/predicates.py and are injected only for the
        # oracle, so the agent runs grid-free. The target's actual coordinates
        # are surfaced to the agent through the per-task goal_nl (see
        # _make_tasks).
        self._BallAtTarget = Predicate(
            "BallAtTarget", [self._ball_type, self._target_type],
            self._BallAtTarget_holds,
            natural_language_assertion=lambda os:
            f"ball {os[0]} has reached the target {os[1]}")
        self._Controls = Predicate("Controls",
                                   [self._switch_type, self._fan_type],
                                   self._Controls_holds)

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        super()._store_pybullet_bodies(pybullet_bodies)
        # Sides (no PyBullet bodies, just assign the direction indices
        # the oracle helper predicates read)
        self._sides[0].side_idx = 1.0
        self._sides[1].side_idx = 0.0
        self._sides[2].side_idx = 3.0
        self._sides[3].side_idx = 2.0

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_fan"

    @property
    def predicates(self) -> Set[Predicate]:
        # Physical-only vocabulary (agent runs grid-free). The grid helper
        # predicates (BallAtLoc / ClearLoc / SideOf / FanFacingSide /
        # OppositeFan) are provided by
        # PyBulletFanGroundTruthPredicateFactory and injected only for the
        # oracle / process-planning approaches.
        predicates = {
            self._FanOn,
            self._FanOff,
            self._BallAtTarget,
            self._Controls,
        }
        if not CFG.fan_known_controls_relation:
            predicates |= {self._SwitchOn, self._SwitchOff}
        return predicates

    @property
    def target_predicates(self) -> Set[Predicate]:
        return {self._BallAtTarget}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._BallAtTarget}

    # -------------------------------------------------------------------------
    # Task grid geometry (hidden: the task distribution's structure)
    # -------------------------------------------------------------------------
    @classmethod
    def _boundary_specs_for_grid(
            cls, x_coords: List[float],
            y_coords: List[float]) -> Dict[str, Dict[str, float]]:
        """Pose + world extents of the four boundary slabs for a grid.

        The slabs sit half a grid gap outside the extreme cells and span
        the full arena width, reproducing the grid-tight enclosure.
        """
        grid_x_min, grid_x_max = min(x_coords), max(x_coords)
        grid_y_min, grid_y_max = min(y_coords), max(y_coords)
        mid_x = (grid_x_min + grid_x_max) / 2
        mid_y = (grid_y_min + grid_y_max) / 2
        span_x = grid_x_max - grid_x_min + cls.pos_gap
        span_y = grid_y_max - grid_y_min + cls.pos_gap
        thickness = cls.boundary_wall_thickness
        z = cls.table_height + cls.boundary_wall_height / 2

        def _spec(x: float, y: float, x_len: float,
                  y_len: float) -> Dict[str, float]:
            return {
                "x": x,
                "y": y,
                "z": z,
                "rot": 0.0,
                "x_len": x_len,
                "y_len": y_len,
                "z_len": cls.boundary_wall_height,
            }

        return {
            "left": _spec(grid_x_min - cls.pos_gap / 2, mid_y, thickness,
                          span_y),
            "right": _spec(grid_x_max + cls.pos_gap / 2, mid_y, thickness,
                           span_y),
            "down": _spec(mid_x, grid_y_min - cls.pos_gap / 2, span_x,
                          thickness),
            "up": _spec(mid_x, grid_y_max + cls.pos_gap / 2, span_x,
                        thickness),
        }

    @classmethod
    def _generate_grid_coordinates(
            cls, num_pos_x: int,
            num_pos_y: int) -> Tuple[List[float], List[float]]:
        """Generate grid coordinates for the maze with specified dimensions."""
        if num_pos_x % 2 == 1:
            x_start = cls.loc_x_mid - (num_pos_x - 1) * cls.pos_gap / 2
        else:
            x_start = (cls.loc_x_mid - num_pos_x * cls.pos_gap / 2 +
                       cls.pos_gap / 2)

        if num_pos_y % 2 == 1:
            y_start = (cls.loc_y_mid - (num_pos_y - 1) * cls.pos_gap / 2)
        else:
            y_start = (cls.loc_y_mid - num_pos_y * cls.pos_gap / 2 +
                       cls.pos_gap / 2)

        x_coords = [x_start + i * cls.pos_gap for i in range(num_pos_x)]
        y_coords = [y_start + i * cls.pos_gap for i in range(num_pos_y)]

        # Assertions to ensure coordinates don't go beyond bounds
        assert min(x_coords) >= cls.loc_x_lb, (
            f"Minimum x coordinate {min(x_coords)} "
            f"is below lower bound {cls.loc_x_lb}")
        assert max(x_coords) <= cls.loc_x_ub, (
            f"Maximum x coordinate {max(x_coords)} "
            f"is above upper bound {cls.loc_x_ub}")
        assert min(y_coords) >= cls.loc_y_lb, (
            f"Minimum y coordinate {min(y_coords)} "
            f"is below lower bound {cls.loc_y_lb}")
        assert max(y_coords) <= cls.loc_y_ub, (
            f"Maximum y coordinate {max(y_coords)} "
            f"is above upper bound {cls.loc_y_ub}")

        return x_coords, y_coords

    @classmethod
    def _grid_coords_for_point(
            cls, ref_x: float,
            ref_y: float) -> Tuple[List[float], List[float]]:
        """Reproduce the exact task grid, inferred from an on-grid reference.

        The grid is one of the fixed (train / test) sizes centered in the
        workspace; odd/even sizes land on half-gap-offset phases, so exactly
        one candidate has a cell coinciding with a point that sits on a real
        grid cell. This lets the oracle helper injection and the physical
        boundary walls recover the grid without any loc objects in the
        (grid-free) state.

        IMPORTANT: pass a STATIONARY on-grid reference (e.g. the target),
        NOT the moving ball. A ball mid-move sits between cells and can
        momentarily align to the *other* grid phase (train vs test), which
        would flip the whole injected grid between steps and desync any
        closed-loop policy tracking BallAtLoc/SideOf atoms.
        """
        candidates = [
            (CFG.fan_train_num_pos_x, CFG.fan_train_num_pos_y),
            (CFG.fan_test_num_pos_x, CFG.fan_test_num_pos_y),
        ]
        for num_x, num_y in candidates:
            x_coords, y_coords = cls._generate_grid_coordinates(num_x, num_y)
            if (any(abs(cx - ref_x) < cls.pos_gap / 2 for cx in x_coords)
                    and any(
                        abs(cy - ref_y) < cls.pos_gap / 2 for cy in y_coords)):
                return x_coords, y_coords
        # Reference off-grid (shouldn't happen); fall back to the test grid.
        return cls._generate_grid_coordinates(*candidates[-1])

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Handle the oracle helper objects and the goal-flavored
        ``target.is_hit`` sensor, then defer to the base sim for every physical
        feature."""
        # loc/side helper objects are injected only for the oracle (see
        # PyBulletFanGroundTruthTypeFactory) and carry no PyBullet body, so
        # their feature values are encoded in their names. Reconstruct them
        # from the name; this lets the _get_state round-trip succeed even
        # though the env itself is built grid-free.
        reconstructed = self._reconstruct_helper_feature_from_name(
            obj, feature)
        if reconstructed is not None:
            return reconstructed
        # target.is_hit lives here (not in the base sim) because its
        # proximity threshold is goal semantics.
        if obj.type == self._target_type and feature == "is_hit":
            ball_pos, _ = p.getBasePositionAndOrientation(
                self._ball.id, physicsClientId=self._physics_client_id)
            target_pos, _ = p.getBasePositionAndOrientation(
                self._target.id, physicsClientId=self._physics_client_id)
            bx, by = ball_pos[0], ball_pos[1]
            tx, ty = target_pos[0], target_pos[1]
            return 1.0 if self._is_ball_close_to_position(bx, by, tx,
                                                          ty) else 0.0
        return super()._get_domain_specific_feature(obj, feature)

    @staticmethod
    def _reconstruct_helper_feature_from_name(obj: Object,
                                              feature: str) -> Optional[float]:
        """Reconstruct an injected loc/side feature from its object name.

        loc names encode coordinates ("loc_<x>_<y>", e.g.
        "loc_0.4700_1.2800") and side names encode the direction
        ("left"/"right"/"down"/"up"). Returns None for anything else so
        the caller can raise its own error.
        """
        if obj.type.name == "loc" and feature in ("xx", "yy"):
            _, x_str, y_str = obj.name.split("_")
            return float(x_str) if feature == "xx" else float(y_str)
        if obj.type.name == "side" and feature == "side_idx":
            return {
                "left": 1.0,
                "right": 0.0,
                "down": 3.0,
                "up": 2.0
            }[obj.name]
        return None

    # -------------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------------
    def _domain_specific_step(self) -> None:
        """Spin fans & blow the ball."""
        self._simulate_fans()
        state = self._get_state()
        # Draw a debug line at the ball's position
        bx, by = state.get(self._ball, "x"), state.get(self._ball, "y")
        p.addUserDebugLine(
            [bx, by, self.table_height],
            [bx, by, self.table_height + self.debug_line_height],
            [0, 1, 0],
            lifeTime=self.
            debug_line_lifetime,  # short lifetime so each step refreshes
            physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Fan Simulation
    # -------------------------------------------------------------------------
    def _simulate_fans(self) -> None:
        """Spin any switched-on fans and blow the ball."""
        if CFG.fan_use_kinematic:
            self._simulate_fans_kinematic()
        else:
            self._simulate_fans_dynamic()

    def _simulate_fans_dynamic(self) -> None:
        """Spin any on-side's fans and queue its wind on the ball.

        The wind goes through the same machinery a learned simulator
        uses (``queue_residual_commands``): each on-side contributes a
        held-mode ``ApplyForce``, queued here (post-step) and re-applied
        on every physics substep of the next action - so a learned
        rule emitting the same command is bit-identical to the env's
        own wind.
        """
        # For each switch, if on => spin all fans with same side_idx
        wind_commands = []
        for ctrl_fan_idx, switch_obj in enumerate(self._switches):
            on = self._is_switch_on(switch_obj.id)
            fan_obj = self._fans[
                ctrl_fan_idx]  # Get the single fan object for this side

            # Check if fan_ids attribute exists and is populated
            if not hasattr(fan_obj, 'fan_ids') or not fan_obj.fan_ids:
                continue

            if on and fan_obj.fan_ids:  # Apply force
                # Control all physical fans for this side
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=self.fan_spin_velocity,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )
                # Wind force using the first fan in the group for direction
                wind_commands.append(self._fan_wind_command(
                    fan_obj.fan_ids[0]))
            else:
                # Turn off all physical fans for this side
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0.0,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )
        self.queue_residual_commands(wind_commands)

    def _simulate_fans_kinematic(self) -> None:
        """Kinematic fan simulation using position-based movement."""
        # Get current ball position
        ball_pos, ball_orn = p.getBasePositionAndOrientation(
            self._ball.id, physicsClientId=self._physics_client_id)
        ball_x, ball_y, ball_z = ball_pos

        # Calculate movement vector based on active fans
        movement_x = 0.0
        movement_y = 0.0

        # Check each fan and accumulate movement vectors
        for ctrl_fan_idx, switch_obj in enumerate(self._switches):
            on = self._is_switch_on(switch_obj.id)
            fan_obj = self._fans[ctrl_fan_idx]

            # Check if fan_ids attribute exists and is populated
            if not hasattr(fan_obj, 'fan_ids') or not fan_obj.fan_ids:
                continue

            if on and fan_obj.fan_ids:
                # Still spin the fans visually
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=self.fan_spin_velocity,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )

                # Add movement based on fan direction
                if ctrl_fan_idx == 0:  # left fan - push right
                    movement_x += self.kinematic_ball_speed
                elif ctrl_fan_idx == 1:  # right fan - push left
                    movement_x -= self.kinematic_ball_speed
                elif ctrl_fan_idx == 2:  # back fan - push forward (up in y)
                    movement_y += self.kinematic_ball_speed
                elif ctrl_fan_idx == 3:  # front fan - push backward (down in y)
                    movement_y -= self.kinematic_ball_speed
            else:
                # Turn off fans visually
                for i, fan_id in enumerate(fan_obj.fan_ids):
                    joint_id = fan_obj.joint_ids[i]
                    if joint_id >= 0:
                        p.setJointMotorControl2(
                            bodyUniqueId=fan_id,
                            jointIndex=joint_id,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=0.0,
                            force=self.joint_motor_force,
                            physicsClientId=self._physics_client_id,
                        )

        # Apply the accumulated movement by setting ball position
        if movement_x != 0.0 or movement_y != 0.0:
            new_x = ball_x + movement_x
            new_y = ball_y + movement_y

            # Keep the ball within workspace bounds
            new_x = max(self.x_lb, min(self.x_ub, new_x))
            new_y = max(self.y_lb, min(self.y_ub, new_y))

            # Set the new ball position directly
            p.resetBasePositionAndOrientation(
                self._ball.id,
                posObj=[new_x, new_y, ball_z],
                ornObj=ball_orn,
                physicsClientId=self._physics_client_id)

    def _fan_wind_command(self, fan_id: int) -> ApplyForce:
        """The wind an on-fan blows: a held-mode world-frame force on the ball
        along the fan's +X (local frame), for the residual-command executor."""
        _, orn_fan = p.getBasePositionAndOrientation(fan_id,
                                                     self._physics_client_id)

        if CFG.fan_fans_blow_opposite_direction:
            local_dir = np.array([-1.0, 0.0, 0.0])
        else:
            local_dir = np.array([1.0, 0.0, 0.0])  # +X is "forward"
        rmat = np.array(p.getMatrixFromQuaternion(orn_fan)).reshape((3, 3))
        world_dir = rmat.dot(local_dir)
        force_vec = self.wind_force_magnitude * world_dir
        return ApplyForce(
            self._ball.name,
            (float(force_vec[0]), float(force_vec[1]), float(force_vec[2])))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _is_ball_close_to_position(self, bx: float, by: float, tx: float,
                                   ty: float) -> bool:
        """Check if the ball is close to the target."""
        return np.abs(bx - tx) < self.pos_gap / 2 and \
                np.abs(by - ty) < self.pos_gap / 2

    # -------------------------------------------------------------------------
    # Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def _FanOn_holds(state: State, objects: Sequence[Object]) -> bool:
        """(FanOn fan).

        True if the controlling switch is on.
        """
        (fan, ) = objects
        return state.get(fan, "is_on") > 0.5

    def _BallAtTarget_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        """(BallAtTarget ball target).

        True when the ball has reached the physical target cell. This is
        the grid-free goal predicate the agent plans toward; the grid
        helper predicates (BallAtLoc / ClearLoc / SideOf / FanFacingSide
        / OppositeFan) live in ground_truth_models/fan/predicates.py and
        are injected only for the oracle.
        """
        ball, target = objects
        return self._is_ball_close_to_position(state.get(ball, "x"),
                                               state.get(ball, "y"),
                                               state.get(target, "x"),
                                               state.get(target, "y"))

    def _Controls_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """(Controls fan switch)."""
        # Note: this probably needs to be updated.
        switch, fan = objects
        return state.get(fan,
                         "facing_side") == state.get(switch, "controls_fan")

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=CFG.num_train_tasks,
            num_pos_x=CFG.fan_train_num_pos_x,
            num_pos_y=CFG.fan_train_num_pos_y,
            possible_num_walls_per_task=CFG.fan_train_num_walls_per_task,
            rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(
            num_tasks=CFG.num_test_tasks,
            num_pos_x=CFG.fan_test_num_pos_x,
            num_pos_y=CFG.fan_test_num_pos_y,
            possible_num_walls_per_task=CFG.fan_test_num_walls_per_task,
            rng=self._test_rng)

    def _make_tasks(  # pylint: disable=redefined-outer-name
            self, num_tasks: int, num_pos_x: int, num_pos_y: int,
            possible_num_walls_per_task: List[int],
            rng: np.random.Generator) -> List[EnvironmentTask]:
        # Generate grid coordinates for this specific configuration
        x_coords, y_coords = self._generate_grid_coordinates(
            num_pos_x, num_pos_y)
        grid_pos = [(x, y) for y in y_coords for x in x_coords]
        _positions = [
            Object(f"loc_y{i}_x{j}", self._location_type)
            for i in range(num_pos_y) for j in range(num_pos_x)
        ]

        # Create position dictionary for this task configuration
        pos_dict = {}
        pos_index = 0
        for i in range(num_pos_y):
            for j in range(num_pos_x):
                if pos_index < len(_positions):
                    pos_obj = _positions[pos_index]
                    pos_dict[pos_obj] = {"xx": x_coords[j], "yy": y_coords[i]}
                    pos_index += 1

        # Draw debug lines for positions if debug is enabled
        if CFG.pybullet_draw_debug:
            for pos_obj, pos in pos_dict.items():
                p.addUserDebugLine([pos["xx"], pos["yy"], self.table_height], [
                    pos["xx"], pos["yy"],
                    self.table_height + self.debug_line_height
                ], [1, 0, 0],
                                   parentObjectUniqueId=-1,
                                   parentLinkIndex=-1)

        tasks = []  # pylint: disable=redefined-outer-name
        for _ in range(num_tasks):
            # Try to generat a valid task with path validation
            max_attempts = 100  # Prevent infinite loop
            for attempt in range(max_attempts):
                # Sample the number of walls for this task
                num_walls_per_task = rng.choice(possible_num_walls_per_task)
                available_pos = grid_pos.copy()

                # Robot
                robot_dict = {
                    "x": self.robot_init_x,
                    "y": self.robot_init_y,
                    "z": self.robot_init_z,
                    "fingers": self.open_fingers,
                    "roll": self.robot_init_roll,
                    "tilt": self.robot_init_tilt,
                    "wrist": self.robot_init_wrist,
                }

                # Optional curated 3x3 generation: ball on an edge cell,
                # target axis-aligned two cells away, one blocking wall.
                if (CFG.fan_3x3_strategic_task_gen and num_pos_x == 3
                        and num_pos_y == 3):
                    # Edge positions in 3x3 grid: exclude center position
                    center_pos = (x_coords[1], y_coords[1])  # Center position
                    edge_positions = [
                        pos for pos in available_pos if pos != center_pos
                    ]

                    # Ball position: choose from edge positions only
                    ball_pos = tuple(rng.choice(edge_positions))
                    # Safely remove the ball position
                    available_pos.remove(ball_pos)

                    # Choose target to create alignment (same row or column as
                    # ball)
                    aligned_targets = []

                    # Same row targets (horizontal alignment) - 2 steps away
                    for x in x_coords:
                        candidate_pos = (x, ball_pos[1])
                        if (candidate_pos in available_pos
                                and candidate_pos != ball_pos
                                and abs(x - ball_pos[0]) > 1.5 * self.pos_gap):
                            aligned_targets.append(candidate_pos)

                    # Same column targets (vertical alignment) - 2 steps away
                    for y in y_coords:
                        candidate_pos = (ball_pos[0], y)
                        if (candidate_pos in available_pos
                                and candidate_pos != ball_pos
                                and abs(y - ball_pos[1]) > 1.5 * self.pos_gap):
                            aligned_targets.append(candidate_pos)

                    if not aligned_targets:
                        # Fallback to any available position
                        aligned_targets = [
                            pos for pos in available_pos if pos != ball_pos
                        ]

                    tar_pos = tuple(rng.choice(aligned_targets))
                    # Safely remove the target position
                    available_pos.remove(tar_pos)
                    target_dict = {
                        "x": tar_pos[0],
                        "y": tar_pos[1],
                        "z": self.table_height,
                        "rot": 0.0,
                        "is_hit": 0.0,
                    }

                    # Strategic wall placement to block direct path
                    wall_positions = []
                    if num_walls_per_task > 0:
                        # Place wall to block direct path between ball and
                        # target
                        blocking_pos = self._get_strategic_wall_position(
                            ball_pos, tar_pos, x_coords, y_coords,
                            available_pos, rng)
                        if blocking_pos is not None:
                            wall_positions.append(blocking_pos)
                            # Safely remove the blocking position
                else:
                    # Uniform random placement (the default for all grids)
                    # Target
                    tar_pos = tuple(rng.choice(available_pos))
                    available_pos.remove(tar_pos)
                    target_dict = {
                        "x": tar_pos[0],
                        "y": tar_pos[1],
                        "z": self.table_height,
                        "rot": 0.0,
                        "is_hit": 0.0,
                    }

                    # Place walls and collect their grid positions
                    wall_positions = []
                    for i in range(num_walls_per_task):
                        wall_pos = tuple(rng.choice(available_pos))
                        available_pos.remove(wall_pos)
                        wall_positions.append(wall_pos)

                    # Ball position
                    ball_pos = tuple(rng.choice(available_pos))
                    available_pos.remove(ball_pos)

                # Convert continuous positions to grid indices for path
                # validation
                tar_grid_idx = None
                ball_grid_idx = None
                wall_grid_indices = set()

                # Find grid indices for target
                for i, y in enumerate(y_coords):
                    for j, x in enumerate(x_coords):
                        if np.isclose(x, tar_pos[0],
                                atol=self.position_tolerance) and \
                           np.isclose(y, tar_pos[1],
                                atol=self.position_tolerance):
                            tar_grid_idx = (j, i)
                            break
                    if tar_grid_idx is not None:
                        break

                # Find grid indices for ball
                for i, y in enumerate(y_coords):
                    for j, x in enumerate(x_coords):
                        if np.isclose(x, ball_pos[0],
                                atol=self.position_tolerance) and \
                           np.isclose(y, ball_pos[1],
                                atol=self.position_tolerance):
                            ball_grid_idx = (j, i)
                            break
                    if ball_grid_idx is not None:
                        break

                # Find grid indices for walls
                for wall_pos in wall_positions:
                    for i, y in enumerate(y_coords):
                        for j, x in enumerate(x_coords):
                            if np.isclose(x, wall_pos[0],
                                    atol=self.position_tolerance) and \
                               np.isclose(y, wall_pos[1],
                                    atol=self.position_tolerance):
                                wall_grid_indices.add((j, i))
                                break

                # Check if we have a valid path from ball to target
                if tar_grid_idx is not None and ball_grid_idx is not None and \
                   self._has_valid_path(ball_grid_idx, tar_grid_idx,
                   wall_grid_indices, num_pos_x, num_pos_y):
                    # Valid path found, create the task

                    init_dict = {}
                    init_dict[self._robot] = robot_dict
                    init_dict[self._target] = target_dict

                    for fan_obj in self._fans:
                        # Each fan_obj now represents all fans on one side
                        side_idx = fan_obj.side_idx
                        # Set position based on the center or representative
                        # position for the side
                        if side_idx == 2:  # down
                            px = (self.fan_x_lb +
                                  self.fan_x_ub) / 2  # center of back fans
                            py = self.down_fan_y
                            rot = np.pi / 2
                        elif side_idx == 3:  # up
                            px = (self.fan_x_lb +
                                  self.fan_x_ub) / 2  # center of front fans
                            py = self.up_fan_y
                            rot = -np.pi / 2
                        elif side_idx == 0:  # left
                            px = self.left_fan_x
                            py = (self.fan_y_lb +
                                  self.fan_y_ub) / 2  # center of left fans
                            rot = 0.0
                        else:  # right (side_idx == 1)
                            px = self.right_fan_x
                            py = (self.fan_y_lb +
                                  self.fan_y_ub) / 2  # center of right fans
                            rot = np.pi
                        fan_dict = {
                            "x": px,
                            "y": py,
                            "z": self.table_height + self.fan_z_len / 2,
                            "rot": rot,
                            "facing_side": float(side_idx),
                            "is_on": 0.0
                        }
                        init_dict[fan_obj] = fan_dict

                    # Switches default off
                    for switch_obj in self._switches:
                        init_dict[switch_obj] = {
                            "x":
                            self.switch_base_x +
                            self.switch_x_spacing * switch_obj.side_idx,
                            "y":
                            self.switch_y,
                            "z":
                            self.table_height,
                            "rot":
                            np.pi / 2,
                            "controls_fan":
                            float(switch_obj.side_idx),
                            "is_on":
                            0.0,
                        }

                    # Note: the loc/side grid helper objects are NOT baked
                    # into the (agent-visible) state anymore. They are
                    # injected only for the oracle by
                    # PyBulletFanGroundTruthTypeFactory, so the agent runs
                    # grid-free.

                    # Walls
                    for i, wall_pos in enumerate(wall_positions):
                        wall_rot = rng.uniform(-self.wall_rot, self.wall_rot)
                        wall_dims = self._aabb_from_body_dims(
                            self.wall_x_len, self.wall_y_len,
                            self.obstacle_wall_height, wall_rot)
                        init_dict[self._walls[i]] = {
                            "x": wall_pos[0],
                            "y": wall_pos[1],
                            "z":
                            self.table_height + self.obstacle_wall_height / 2,
                            "rot": wall_rot,
                            "x_len": wall_dims[0],
                            "y_len": wall_dims[1],
                            "z_len": wall_dims[2],
                        }

                    # Boundary slabs enclosing the grid. Unlike the loc/side
                    # grid helpers these ARE part of the agent-visible state:
                    # they are real bodies the ball collides with, so the
                    # dynamics must be expressible without them being latent.
                    boundary_specs = self._boundary_specs_for_grid(
                        x_coords, y_coords)
                    for boundary_obj, side in zip(self._boundaries,
                                                  self._boundary_sides):
                        init_dict[boundary_obj] = boundary_specs[side]

                    # Ball
                    ball_dict = {
                        "x": ball_pos[0],
                        "y": ball_pos[1],
                        "z": self.table_height + self.ball_height_offset,
                        "radius": self.ball_radius,
                    }
                    init_dict[self._ball] = ball_dict
                    break
            else:
                # If we couldn't find a valid configuration after max attempts
                raise ValueError(
                    f"Could not generate a valid task configuration after "
                    f"{max_attempts} attempts")
            print(f"Found a valid task after {attempt} attempts")

            init_state = utils.create_state_from_dict(init_dict)

            # Grid-free goal: the ball must reach the physical target. Since
            # the agent no longer sees the loc grid, the target's actual
            # coordinates are surfaced through goal_nl. The oracle rewrites
            # this into the grid goal BallAtLoc(ball, target_loc) when it
            # injects the helper objects (PyBulletFanGroundTruthTypeFactory).
            tx, ty = init_state.get(self._target, "x"), \
                init_state.get(self._target, "y")
            goal_atoms = {
                GroundAtom(self._BallAtTarget, [self._ball, self._target]),
            }
            # all fans are off in the goal
            for fan_obj in self._fans:
                goal_atoms.add(GroundAtom(self._FanOff, [fan_obj]))
            goal_nl = (f"Blow the ball to the target at position "
                       f"(x={tx:.2f}, y={ty:.2f}); all fans must be off.")
            tasks.append(
                EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl))
        return self._add_pybullet_state_to_tasks(tasks)

    def _get_strategic_wall_position(  # pylint: disable=redefined-outer-name
            self, ball_pos: Tuple[float, float],
            target_pos: Tuple[float, float], _x_coords: List[float],
            _y_coords: List[float], available_pos: List[Tuple[float, float]],
            rng: np.random.Generator) -> Optional[Tuple[float, float]]:
        """Get a wall position that is between the ball and target."""
        # Find positions that are between ball and target
        between_positions = []

        for pos in available_pos:
            # Check if position is between ball and target (on same row or
            # column)
            if (pos[0] == ball_pos[0] == target_pos[0] and  # Same column
                    min(ball_pos[1], target_pos[1]) < pos[1] < max(
                        ball_pos[1], target_pos[1])):
                between_positions.append(pos)
            elif (pos[1] == ball_pos[1] == target_pos[1] and  # Same row
                  min(ball_pos[0], target_pos[0]) < pos[0] < max(
                      ball_pos[0], target_pos[0])):
                between_positions.append(pos)

        # Return a random position between ball and target, or random if none
        # found
        if between_positions:
            return rng.choice(between_positions)
        return tuple(rng.choice(available_pos)) if available_pos else None

    def _has_valid_path(self, start_pos: Tuple[int,
                                               int], target_pos: Tuple[int,
                                                                       int],
                        blocked_positions: Set[Tuple[int, int]],
                        num_pos_x: int, num_pos_y: int) -> bool:
        """Check if there's a valid path from start to target using only
        cardinal directions."""
        if start_pos == target_pos:
            return True

        # BFS to find path using only cardinal directions
        queue = deque([start_pos])
        visited = {start_pos}

        # Cardinal directions: up, down, left, right
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        while queue:
            current_x, current_y = queue.popleft()

            for dx, dy in directions:
                next_x, next_y = current_x + dx, current_y + dy

                # Check bounds
                if not (0 <= next_x < num_pos_x and 0 <= next_y < num_pos_y):
                    continue

                # Check if position is blocked or already visited
                if (next_x,
                        next_y) in blocked_positions or (next_x,
                                                         next_y) in visited:
                    continue

                # Check if we reached the target
                if (next_x, next_y) == target_pos:
                    return True

                visited.add((next_x, next_y))
                queue.append((next_x, next_y))

        return False


if __name__ == "__main__":
    import time  # pylint: disable=ungrouped-imports
    CFG.seed = 0
    CFG.env = "pybullet_fan"
    env = PyBulletFanEnv(use_gui=True)
    _rng = np.random.default_rng(CFG.seed)
    _tasks = env._make_tasks(  # pylint: disable=protected-access
        10, CFG.fan_train_num_pos_x, CFG.fan_train_num_pos_y,
        CFG.fan_train_num_walls_per_task, _rng)

    for _task in _tasks:
        env._set_state(_task.init)  # pylint: disable=protected-access
        for _ in range(5000):
            _action = Action(
                np.array(env._pybullet_robot  # pylint: disable=protected-access
                         .initial_joint_positions))
            env.step(_action)
            time.sleep(0.1)
