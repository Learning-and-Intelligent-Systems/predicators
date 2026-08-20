"""Core abstractions for reusable parameterized skills."""
# pylint: disable=wrong-import-position,ungrouped-imports

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, \
    Optional, Sequence, Tuple, cast

if TYPE_CHECKING:
    from predicators.envs.pybullet_env import PyBulletEnv

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.pybullet_helpers.controllers import \
    _build_action_from_joints, _robot_supports_base_action, \
    get_change_fingers_action, get_move_end_effector_to_pose_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.robots.single_arm import \
    SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type


class PhaseAction(Enum):
    """The type of action a phase executes."""
    MOVE_TO_POSE = auto()
    CHANGE_FINGERS = auto()


# Process-wide cache of the simulator envs that skills use for
# collision-aware motion planning, one per env class. A PyBullet DIRECT
# client is never freed (nothing in this codebase calls p.disconnect), so
# constructing a fresh env per SkillConfig — which happens on every
# get_gt_options() call — leaks the entire physics world (~145MB for the
# domino env; the min-block K* probes call get_gt_options per probe and
# drove a run past 12GB). Sharing one env per class is safe because
# _plan_with_simulator re-syncs the full state via _set_state before every
# query. The cache is cleared on config updates (see
# utils.update_config_with_parser): env construction reads CFG, so a config
# change is exactly when a cached simulator goes stale. Evicted envs are
# dropped, not disconnected — SkillConfigs built earlier may still use them.
_SHARED_SIMULATOR_CACHE: Dict[type, Any] = {}


def shared_skill_simulator(env_cls: type) -> Any:
    """Return the process-wide shared motion-planning env for ``env_cls``."""
    sim = _SHARED_SIMULATOR_CACHE.get(env_cls)
    if sim is None:
        sim = env_cls(use_gui=False)
        _SHARED_SIMULATOR_CACHE[env_cls] = sim
    return sim


# Same rationale as _SHARED_SIMULATOR_CACHE, for the robot handle that
# SkillConfig needs: option factories used to call
# env_cls.initialize_pybullet() per get_options() call, leaking a full
# physics world each time even with motion planning off.
_SHARED_ROBOT_CACHE: Dict[type, Any] = {}


def shared_skill_robot(env_cls: Any) -> Any:
    """Return a process-wide robot handle for ``env_cls`` SkillConfigs, backed
    by one cached ``initialize_pybullet`` world per env class."""
    robot = _SHARED_ROBOT_CACHE.get(env_cls)
    if robot is None:
        _, robot, _ = env_cls.initialize_pybullet(using_gui=False)
        _SHARED_ROBOT_CACHE[env_cls] = robot
    return robot


def clear_shared_simulator_cache() -> None:
    """Drop cached skill simulators and robots; called on config changes."""
    _SHARED_SIMULATOR_CACHE.clear()
    _SHARED_ROBOT_CACHE.clear()


@dataclass(frozen=True)
class SkillConfig:
    """Configuration shared across all skill factories for one environment.

    Every skill factory function (``create_pick_skill``, ``create_place_skill``,
    etc.) takes a ``SkillConfig`` as its fourth argument.  Each environment
    options file creates one ``SkillConfig`` and passes it to all its skill
    factory calls.

    Example::

        config = SkillConfig(
            robot=pybullet_robot,
            open_fingers_joint=pybullet_robot.open_fingers,
            closed_fingers_joint=pybullet_robot.closed_fingers,
            fingers_state_to_joint=MyEnv._fingers_state_to_joint,
            robot_init_tilt=MyEnv.robot_init_tilt,   # default 0.0
            robot_init_wrist=MyEnv.robot_init_wrist,  # default 0.0
        )

    Attributes:
        robot: The PyBullet robot instance.
        open_fingers_joint: Joint value for fully open fingers.
        closed_fingers_joint: Joint value for fully closed fingers.
        fingers_state_to_joint: Callable that maps the finger *state feature*
            value to the corresponding joint value.  Signature:
            ``(robot, finger_state_value) -> joint_value``.  Typically
            ``MyEnv._fingers_state_to_joint``.
        collision_bodies: PyBullet body IDs to treat as obstacles during
            BiRRT planning.  Defaults to empty (no collision checking).
        move_to_pose_tol: Squared-distance tolerance for move-to-pose
            terminal (used when BiRRT falls back to incremental IK).
        finger_action_nudge_magnitude: Nudge magnitude for finger drift
            resistance in the wait option and during move phases.
        max_vel_norm: Maximum velocity norm for incremental IK EE movement.
        grasp_tol: Squared-distance tolerance for CHANGE_FINGERS terminal.
        ik_validate: Whether to validate IK solutions.
        robot_init_tilt: Default EE tilt (pitch) angle — the second Euler
            angle in ``[roll=0, pitch, yaw]``.
        robot_init_wrist: Default EE wrist (yaw) angle — the third Euler
            angle.  Usually 0.0 or ``-pi``.
        robot_home_pos: ``(x, y, z)`` home position the robot retreats to
            after push skills.  Required by ``create_push_skill``.
        transport_z: Safe Z height for transit above obstacles during
            pick, place, push, and pour skills.  Default ``0.7``.
        base_standoff: For mobile-base robots, the forward (y) distance at
            which the base parks in front of a reach target (with its x aligned
            to the target x), so the arm reaches it straight forward at a
            comfortable distance instead of sideways over the burner/a jug or
            fully extended.  ``None`` (default) disables base positioning; only
            mobile robots use it.
        base_y_max: Upper bound on the base y while positioning, to keep the
            base clear of the table front.  Default ``inf`` (no clamp).
        extra: Arbitrary dict for environment-specific constants that
            callbacks may need.  Access via ``config.extra["key"]``.
    """
    robot: SingleArmPyBulletRobot
    open_fingers_joint: float
    closed_fingers_joint: float
    fingers_state_to_joint: Callable[[SingleArmPyBulletRobot, float], float]
    collision_bodies: Tuple[int, ...] = ()
    move_to_pose_tol: float = 1e-4
    finger_action_nudge_magnitude: float = 1e-3
    max_vel_norm: float = 0.05
    grasp_tol: float = 5e-4
    ik_validate: bool = True
    robot_init_tilt: float = 0.0
    robot_init_wrist: float = 0.0
    robot_home_pos: Optional[Tuple[float, float, float]] = None
    transport_z: float = 0.7
    base_standoff: Optional[float] = None
    base_y_max: float = float("inf")
    base_align_x: bool = True
    base_home_xy: Optional[Tuple[float, float]] = None
    simulator: Optional[PyBulletEnv] = None
    collision_skip_types: Tuple[str, ...] = ()
    sim_extra_collision_bodies: Tuple[int, ...] = ()
    # Wait-option quiescence termination: when set, Wait terminates once
    # every non-robot object's state features change by less than this
    # eps for ``wait_quiescence_steps`` consecutive steps (the scene has
    # settled), instead of always running to the option-rollout cap.
    # ``None`` (default) keeps the never-terminate behavior - correct for
    # domains that Wait for TIME on a static scene (e.g. glue curing),
    # wrong only in cost for domains that Wait for motion to stop (a
    # domino cascade settles in ~100-200 steps but paid the full
    # 1000-step cap on every probe rollout).
    wait_quiescence_eps: Optional[float] = None
    wait_quiescence_steps: int = 10
    # Per-env override for the held object's bystander clearance during
    # BiRRT (metres); see pybullet_birrt_held_bystander_clearance in
    # settings.py. ``None`` (default) uses the global setting. Envs whose
    # bystanders topple from a graze (dominoes) should raise this above
    # pybullet_birrt_bystander_clearance; envs with tight corridors
    # cannot afford it.
    held_bystander_clearance: Optional[float] = None
    # Grasp-relative release for place skills: at the drop pose the
    # gripper opens GRADUALLY just until the simulator drops the grasp
    # constraint (observed as is_held flipping in the state), opens
    # ``_RELEASE_CLEAR_SLACK`` further so the pads clear the released
    # object, retreats HOLDING that width, and only fully opens back at
    # transport height. The release width therefore derives from the
    # measured grasp width of whatever is held - no per-object constant -
    # and the side clearance a placement needs shrinks from the full
    # opening span (±4 cm) to roughly the object thickness plus a few
    # millimetres. Requires the env to expose holding via an ``is_held``
    # object feature (all current place-skill envs do). False restores
    # the legacy full open at the drop pose.
    release_until_ungrasped: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


def build_params_space(
    param_defs: Sequence[Tuple[str, float, float]],
) -> Tuple[Box, Tuple[str, ...]]:
    """Build a params_space and description from ``(name, low, high)`` tuples.

    Returns:
        ``(params_space, params_description)``
    """
    names = tuple(name for name, _, _ in param_defs)
    low = np.array([lo for _, lo, _ in param_defs], dtype=np.float64)
    high = np.array([hi for _, _, hi in param_defs], dtype=np.float64)
    return Box(low=low, high=high, dtype=np.float64), names


def _fmt_option_params(params: Array) -> str:
    """Render an option's continuous params for failure messages.

    Failure messages are the agent's only channel for learning which
    parameter values produced an infeasible target, so echo them
    compactly (``[0.05, 0.02]``; ``[]`` for parameter-free options).
    """
    return "[" + ", ".join(f"{float(v):.4g}" for v in params) + "]"


# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

# Callback signature shared by ALL skill factory ``get_target_pose_fn`` args.
# (state, objects, params, config) -> (x, y, z, yaw)
TargetPoseFn = Callable[[State, Sequence[Object], Array, SkillConfig],
                        Tuple[float, float, float, float]]

# ---------------------------------------------------------------------------
# Internal type aliases for Phase target functions
# ---------------------------------------------------------------------------

# For MOVE_TO_POSE: returns (current_pose, target_pose, finger_status)
MoveToPoseTargetFn = Callable[[State, Sequence[Object], Array, SkillConfig],
                              Tuple[Pose, Pose, str]]

# For CHANGE_FINGERS: returns (current_val, target_val)
ChangeFingersTargetFn = Callable[[State, Sequence[Object], Array, SkillConfig],
                                 Tuple[float, float]]

# Memory keys used per phase, keyed by phase object id.
_BIRRT_TRAJ_KEY = "birrt_traj_{}"  # stores List[JointPositions] or None
_BIRRT_STEP_KEY = "birrt_step_{}"  # stores int index into trajectory
_BIRRT_FINGER_KEY = "birrt_finger_{}"  # stores finger_status str
_BIRRT_HOLD_KEY = "birrt_hold_{}"  # consecutive re-commands of a waypoint
_FINGER_TARGET_KEY = "finger_target_{}"  # anchored CHANGE_FINGERS target

# Grasp-relative release (SkillConfig.release_until_ungrasped): the
# opening commanded at the drop pose (anchored at the measured grasp
# width) while waiting for the simulator to drop the grasp constraint,
# and how much further the gripper opens once the release is observed so
# the pads clear the released object before the hold-width retreat. The
# open step must exceed every env's _finger_action_tol or that env never
# classifies the action as "opening" and never releases the constraint -
# all envs now share the base 1e-4, but grow's since-removed 5e-3
# override once silently swallowed a 2mm step (caught by its jug tests),
# so the step stays comfortably large. The planner's release-clearance
# check budgets the worst-case width: grasp + open step + clear slack +
# margin.
_RELEASE_OPEN_STEP = 0.01
_RELEASE_CLEAR_SLACK = 0.004
_RELEASE_CHECK_BUFFER = _RELEASE_OPEN_STEP + _RELEASE_CLEAR_SLACK + 0.002
_IK_STALL_BEST_KEY = "ik_stall_best_{}"  # best EE-to-target distance seen
_IK_STALL_COUNT_KEY = "ik_stall_count_{}"  # steps since last improvement


@dataclass
class Phase:
    """A single phase in a multi-phase skill.

    Attributes:
        name: Human-readable phase name (for logging).
        action_type: Whether this phase moves the EE or changes fingers.
        target_fn: Callable that computes targets from state/objects/params.
            For MOVE_TO_POSE: returns (current_pose, target_pose, finger_status)
            For CHANGE_FINGERS: returns (current_val, target_val)
        terminal_fn: Optional custom terminal condition override.
            Signature: (state, objects, params, config) -> bool
        finger_tol: Tolerance for CHANGE_FINGERS terminal (defaults to
            config.grasp_tol if None).
        use_motion_planning: If True (default) and action_type is
            MOVE_TO_POSE, use BiRRT to plan a joint-space trajectory on the
            first call and cache it; subsequent calls pop waypoints from the
            cached plan. Falls back to incremental IK if planning fails.
            If False, always use incremental IK stepping.
    """
    name: str
    action_type: PhaseAction
    # Union[MoveToPoseTargetFn, ChangeFingersTargetFn]; typed as Any to
    # avoid Pylance issues when unpacking return tuples after runtime
    # dispatch on action_type.
    target_fn: Any
    terminal_fn: Optional[Callable[
        [State, Sequence[Object], Array, SkillConfig], bool]] = None
    finger_tol: Optional[float] = None
    # For CHANGE_FINGERS: "open" or "close". When set, the terminal uses
    # an asymmetric tolerance (must reach at least target − √tol when
    # opening, at most target + √tol when closing) instead of the
    # symmetric (target − current)² < tol — which can falsely accept a
    # state where fingers haven't moved off the opposite endpoint.
    finger_direction: Optional[str] = None
    use_motion_planning: bool = field(
        default_factory=lambda: CFG.skill_phase_use_motion_planning)
    expect_contact: bool = False
    allow_shallow_held_object_contacts: bool = False
    # Force validated (iterative) IK for this phase's BiRRT goal pose, even
    # when CFG.pybullet_ik_validate is False. Unvalidated IK can return a goal
    # config whose EE pose is numerically close but whose gripper slightly
    # penetrates the very object being approached (the grasp target), making
    # BiRRT reject an otherwise-reachable grasp. Validating only this phase's
    # goal fixes that without the cost/regressions of globally validating
    # every transport/retreat IK.
    validate_ik: bool = False
    # Additionally collision-check this phase's BiRRT goal config with the
    # fingers OPEN. Set on a place descent whose next phase opens the
    # gripper: the opening sweep itself is not planned, so a drop pose
    # whose opening fingers would clip a neighbor (e.g. a domino placed
    # closer to the previous one than the finger span) must be rejected
    # at planning time, not discovered by toppling the neighbor.
    check_release_clearance: bool = False
    # For CHANGE_FINGERS phases whose target depends on the CURRENT finger
    # value (e.g. a grasp-relative release width of "current + slack"):
    # freeze the target at its first evaluation for the rest of the phase.
    # Without anchoring, "current + slack" ratchets - each step the fingers
    # open, the target moves further out, and the phase never terminates
    # short of fully open.
    anchor_finger_target: bool = False


class PhaseSkill:
    """A multi-phase controller that builds a ParameterizedOption.

    Each phase is executed sequentially. The skill advances to the next
    phase when the current phase's terminal condition is met. The overall
    skill terminates when the last phase is terminal.

    For MOVE_TO_POSE phases with use_motion_planning=True (the default),
    BiRRT plans a collision-free joint-space trajectory on the first call
    and caches it in the option memory dict. Subsequent calls pop waypoints
    from the cached plan one at a time. If BiRRT fails, the phase falls back
    to incremental IK delta-stepping.

    Usage:
        option = PhaseSkill("Pick", types, params_space, config, phases).build()
    """

    def __init__(self,
                 name: str,
                 types: Sequence[Type],
                 params_space: Box,
                 config: SkillConfig,
                 phases: List[Phase],
                 params_description: Optional[Tuple[str, ...]] = None,
                 base_mode: Optional[str] = None) -> None:
        assert len(phases) > 0
        self._name = name
        self._types = types
        self._params_space = params_space
        self._config = config
        self._phases = phases
        self._params_description = params_description
        # Mobile-base positioning mode for this skill (None disables it):
        #   "home"        park at the robot's home base (good offset to press a
        #                 switch; diagonal fixed-base reach for far targets).
        #   "align_left"  slide base x toward the target but not right of home
        #                 (frees the over-the-burner reach), forward in y.
        #   "diag"        keep base x at home, move forward in y (diagonal carry
        #                 that clears an adjacent jug / the faucet body).
        self._base_mode = base_mode
        # Collision diagnostics from the most recent failed BiRRT plan,
        # attached to the OptionExecutionFailure so agents learn which
        # object blocked the motion plan.
        self._last_plan_diagnostics: List[str] = []

    def build(self) -> ParameterizedOption:
        """Build and return the ParameterizedOption."""
        return ParameterizedOption(
            self._name,
            types=self._types,
            params_space=self._params_space,
            policy=self._policy,
            initiable=self._initiable,
            terminal=self._terminal,
            params_description=self._params_description,
        )

    def _initiable(self, state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        del state, objects, params  # unused
        memory["phase_idx"] = 0
        return True

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        phase_idx = memory["phase_idx"]
        phase = self._phases[phase_idx]

        # Check if current phase is terminal → advance.
        if self._phase_is_terminal(phase, state, memory, objects, params):
            phase_idx += 1
            memory["phase_idx"] = phase_idx
            if phase_idx >= len(self._phases):
                # Should not be called after overall terminal, but guard.
                phase_idx = len(self._phases) - 1
                memory["phase_idx"] = phase_idx
            phase = self._phases[phase_idx]
            logging.debug(f"[{self._name}] Advanced to phase {phase_idx}: "
                          f"{phase.name}")

        if phase.action_type == PhaseAction.MOVE_TO_POSE:
            return self._execute_move(phase, state, memory, objects, params)
        assert phase.action_type == PhaseAction.CHANGE_FINGERS
        return self._execute_fingers(phase, state, memory, objects, params)

    def _finger_target(self, phase: Phase, state: State, memory: Dict,
                       objects: Sequence[Object],
                       params: Array) -> Tuple[float, float]:
        """(current, target) finger values for a CHANGE_FINGERS phase, freezing
        the target at its first evaluation when the phase asks for anchoring
        (grasp-relative targets would otherwise ratchet)."""
        current_val, target_val = phase.target_fn(state, objects, params,
                                                  self._config)
        if phase.anchor_finger_target:
            key = _FINGER_TARGET_KEY.format(id(phase))
            if key not in memory:
                memory[key] = target_val
            target_val = memory[key]
        return current_val, target_val

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        phase_idx = memory["phase_idx"]
        if phase_idx < len(self._phases) - 1:
            return False
        phase = self._phases[phase_idx]
        return self._phase_is_terminal(phase, state, memory, objects, params)

    # ------------------------------------------------------------------
    # Phase terminal conditions
    # ------------------------------------------------------------------

    def _phase_is_terminal(self, phase: Phase, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
        """Check if a phase has reached its terminal condition."""
        # Custom terminal override takes priority.
        if phase.terminal_fn is not None:
            return phase.terminal_fn(state, objects, params, self._config)

        if phase.action_type == PhaseAction.CHANGE_FINGERS:
            current_val, target_val = self._finger_target(
                phase, state, memory, objects, params)
            tol = phase.finger_tol if phase.finger_tol is not None \
                else self._config.grasp_tol
            tol_lin = float(np.sqrt(tol))
            if phase.finger_direction == "open":
                return bool(current_val >= target_val - tol_lin)
            if phase.finger_direction == "close":
                return bool(current_val <= target_val + tol_lin)
            return bool((target_val - current_val)**2 < tol)

        # MOVE_TO_POSE
        if phase.use_motion_planning:
            return self._birrt_phase_is_terminal(phase, state, memory, objects,
                                                 params)
        return self._ik_phase_is_terminal(phase, state, objects, params)

    def _birrt_phase_is_terminal(self, phase: Phase, state: State,
                                 memory: Dict, objects: Sequence[Object],
                                 params: Array) -> bool:
        """Terminal for a BiRRT-planned phase.

        Returns True when the cached trajectory is fully consumed, or
        when the fallback IK terminal is satisfied (BiRRT planning
        failed). Returns False if the trajectory hasn't been computed
        yet (first call).
        """
        pid = id(phase)
        traj_key = _BIRRT_TRAJ_KEY.format(pid)
        step_key = _BIRRT_STEP_KEY.format(pid)

        if traj_key not in memory:
            # Trajectory not yet computed — not terminal.
            return False

        traj = memory[traj_key]
        if traj is None:
            # BiRRT failed; use distance-based terminal (IK fallback mode).
            return self._ik_phase_is_terminal(phase, state, objects, params)

        # All waypoints consumed — fall back to position-based terminal so
        # the phase doesn't end until the robot has actually converged to the
        # target (position control may lag behind the commanded trajectory,
        # and IK inaccuracy means the final waypoint may not exactly match
        # the target Cartesian pose).
        if memory[step_key] >= len(traj):
            return self._ik_phase_is_terminal(phase, state, objects, params)
        return False

    def _ik_phase_is_terminal(self, phase: Phase, state: State,
                              objects: Sequence[Object],
                              params: Array) -> bool:
        """Distance-based terminal for incremental IK phases."""
        current_pose, target_pose, _ = phase.target_fn(state, objects, params,
                                                       self._config)
        squared_dist = np.sum(
            np.square(np.subtract(current_pose.position,
                                  target_pose.position)))
        return bool(squared_dist < self._config.move_to_pose_tol)

    def _check_ik_stall(self, phase: Phase, state: State, memory: Dict,
                        objects: Sequence[Object], params: Array) -> None:
        """Abort the option when incremental IK stops making progress.

        Tracks the best end-effector-to-target distance in the option's
        memory; ``_ik_stall_window`` consecutive steps without improving
        it by ``_ik_stall_min_progress`` raise
        ``OptionExecutionFailure`` (the incremental-IK distance terminal
        can otherwise never fire, leaving the arm thrashing until the
        episode horizon).
        """
        current_pose, target_pose, _ = phase.target_fn(state, objects, params,
                                                       self._config)
        dist = float(
            np.linalg.norm(
                np.subtract(current_pose.position, target_pose.position)))
        pid = id(phase)
        best_key = _IK_STALL_BEST_KEY.format(pid)
        count_key = _IK_STALL_COUNT_KEY.format(pid)
        best = memory.get(best_key)
        if best is None or dist < best - self._ik_stall_min_progress:
            memory[best_key] = dist
            memory[count_key] = 0
            return
        memory[count_key] = memory.get(count_key, 0) + 1
        if memory[count_key] >= self._ik_stall_window:
            tgt = target_pose.position
            contact_report = self._stall_contact_report(
                cast(utils.PyBulletState, state))
            raise utils.OptionExecutionFailure(
                f"[{self._name}/{phase.name}] incremental-IK stalled: no "
                f"end-effector progress in {self._ik_stall_window} steps "
                f"({dist:.3f} m from the target ({tgt[0]:.3f}, {tgt[1]:.3f}, "
                f"{tgt[2]:.3f}) commanded by params "
                f"{_fmt_option_params(params)}); aborting option."
                f"{contact_report}")

    # ------------------------------------------------------------------
    # Phase execution
    # ------------------------------------------------------------------

    def _execute_move(self, phase: Phase, state: State, memory: Dict,
                      objects: Sequence[Object], params: Array) -> Action:
        """Dispatch to BiRRT or incremental IK based on phase flag.

        For mobile-base robots, first drive the base to a pose that puts
        the reach target in comfortable arm range (the arm BiRRT/IK then
        plans from the repositioned base).
        """
        base_action = self._maybe_drive_base(phase, state, memory, objects,
                                             params)
        if base_action is not None:
            return base_action
        if phase.use_motion_planning:
            return self._execute_move_birrt(phase, state, memory, objects,
                                            params)
        return self._execute_move_ik(phase, state, objects, params)

    # Mobile-base positioning. Before the first reach of an option, drive the
    # (kinematic) base to park `base_standoff` in front of the reach target with
    # its x aligned to the target x (base y clamped to base_y_max to stay clear
    # of the table), so the arm reaches *straight forward at a comfortable
    # distance* rather than sideways/over the burner or fully extended. The base
    # pose is a deterministic function of the option params, so it is
    # reproducible across refinement samples (unlike a per-sample search) and
    # adds just one base-drive step per option. Enabled per-env by setting
    # base_standoff; only active for mobile robots (e.g. mobile_fetch), a no-op
    # for fixed bases.
    _base_pos_tol: ClassVar[float] = 0.02  # xy tol to call the base positioned
    _base_step: ClassVar[float] = 0.08  # max base xy move per step (smooth)

    # Incremental-IK stall abort: when a phase is running on incremental IK
    # (BiRRT-failed fallback, or converging after a consumed trajectory) and
    # the end effector gets no closer to the phase target than its best
    # distance so far (by at least ``_ik_stall_min_progress``) for
    # ``_ik_stall_window`` consecutive steps, the option aborts with an
    # ``OptionExecutionFailure`` instead of flailing until the episode
    # horizon (where the thrashing arm bulldozes the scene).
    _ik_stall_window: ClassVar[int] = 25
    # Random in-limit IK restarts for the BiRRT goal solve, tried after
    # the current-joints and home seeds (see _solve_goal_ik).
    _goal_ik_num_restarts: ClassVar[int] = 8
    _ik_stall_min_progress: ClassVar[float] = 2e-3  # meters

    def _maybe_drive_base(self, phase: Phase, state: State, memory: Dict,
                          objects: Sequence[Object],
                          params: Array) -> Optional[Action]:
        """Return a one-step base-drive Action that stands the base in front of
        this option's reach target; None once positioned (or for fixed-base
        robots / when base positioning is disabled)."""
        robot = self._config.robot
        if self._config.base_standoff is None \
                or self._base_mode is None \
                or not _robot_supports_base_action(robot):
            return None
        pb_state = cast(utils.PyBulletState, state)
        sim_state = pb_state.simulator_state
        if not isinstance(sim_state, dict) or "base_pose" not in sim_state:
            return None
        if memory.get("_base_pos_done", False):
            return None
        (cur_x, cur_y, _), _ = sim_state["base_pose"]
        home_xy = self._config.base_home_xy
        if self._base_mode == "home" and home_xy is not None:
            # Push: park at the robot's home base, which sits diagonally off the
            # switch (offset opposite the push direction and in front) so the
            # arm presses it naturally. Head-on (x-aligned) pins the arm near a
            # singularity and makes the push wander off target.
            target_bx, target_by = home_xy
        else:
            _, target_pose, _ = phase.target_fn(state, objects, params,
                                                self._config)
            home_x = home_xy[0] if home_xy is not None else (
                self._config.robot_home_pos[0]
                if self._config.robot_home_pos is not None else float(cur_x))
            stay_home = False
            if self._base_mode == "align_left":
                # Pick: slide x toward the target but never to the right of
                # home. The over-the-burner reach only happens for targets left
                # of home; right targets (front jug, jug under the faucet) keep
                # home's diagonal approach, which clears the faucet body.
                target_bx = min(float(target_pose.position[0]), home_x)
            elif self._base_mode == "approach":
                # Pick a jug that may sit beside another jug (the 2-jug boil
                # tasks). Reposition only when a second jug actually blocks the
                # reach -- one sitting close to the target in both x and y, so
                # reaching it from home would sweep the arm across it (the jug0-
                # vs-jug1 grasp/lift collision a fixed base cannot avoid). Then
                # stand to the target's far side from that jug, offset laterally
                # (NOT x-aligned, which pins this arm at a singularity -- see
                # the "home" push note). With no blocker, keep home's diagonal
                # approach: moving the base in would only risk that singularity
                # (e.g. re-picking a jug under the faucet, with no neighbor).
                tx = float(target_pose.position[0])
                ty = float(target_pose.position[1])
                blocker_x: Optional[float] = None
                for other in state:
                    if other.type.name != "jug" or other in objects:
                        continue
                    ox = float(state.get(other, "x"))
                    oy = float(state.get(other, "y"))
                    if abs(ox - tx) < 0.4 and abs(oy - ty) < 0.4:
                        blocker_x = ox
                        break
                if blocker_x is None:
                    target_bx = home_x
                    stay_home = True
                else:
                    side = 1.0 if tx >= blocker_x else -1.0
                    target_bx = tx + side * 0.15
            else:
                # Place ("diag"): keep base x at home and only move forward in
                # y, so the carry stays diagonal (clearing an adjacent jug or
                # the faucet body) yet close enough for a comfortable reach.
                target_bx = home_x
            if stay_home:
                # No reposition needed: return to (or stay at) the home base so
                # the reach keeps home's well-conditioned diagonal geometry.
                target_by = home_xy[1] if home_xy is not None else float(cur_y)
            else:
                target_by = min(
                    float(target_pose.position[1]) -
                    self._config.base_standoff, self._config.base_y_max)
        dx, dy = target_bx - cur_x, target_by - cur_y
        dist = float(np.hypot(dx, dy))
        if dist < self._base_pos_tol:
            memory["_base_pos_done"] = True
            return None
        # Move the base toward the target in small increments rather than one
        # teleport, so a held jug follows the grasp constraint smoothly instead
        # of being yanked across the jump (which destabilizes the carry).
        if dist > self._base_step:
            dx *= self._base_step / dist
            dy *= self._base_step / dist
        base_delta = np.array([dx, dy, 0.0], dtype=np.float32)
        return _build_action_from_joints(robot, pb_state.joint_positions,
                                         base_delta)

    def _execute_move_birrt(self, phase: Phase, state: State, memory: Dict,
                            objects: Sequence[Object],
                            params: Array) -> Action:
        """Execute a MOVE_TO_POSE phase using BiRRT with lazy plan caching.

        On the first call for this phase:
          1. Compute the target joint positions via IK.
          2. Run BiRRT from the current joint positions to the target.
          3. Cache the resulting trajectory (or None on failure).
          4. Cache the finger_status for nudging during trajectory replay.

        On subsequent calls, pop the next waypoint from the cached trajectory
        and return the corresponding joint-position action, applying a small
        finger nudge matching the phase's finger_status (same as incremental
        IK) to prevent drift and allow finger transitions during movement.

        Falls back to incremental IK if BiRRT planning fails.
        """
        pid = id(phase)
        traj_key = _BIRRT_TRAJ_KEY.format(pid)
        step_key = _BIRRT_STEP_KEY.format(pid)
        finger_key = _BIRRT_FINGER_KEY.format(pid)

        pb_state = cast(utils.PyBulletState, state)
        robot = self._config.robot

        if traj_key not in memory:
            # --- First call: plan the trajectory. ---
            _, target_pose, finger_status = phase.target_fn(
                state, objects, params, self._config)
            memory[finger_key] = finger_status

            self._last_plan_diagnostics = []
            if self._config.simulator is not None:
                traj = self._plan_with_simulator(pb_state, target_pose,
                                                 phase.name,
                                                 phase.expect_contact, objects,
                                                 phase)
            else:
                traj = self._plan_without_simulator(pb_state, target_pose,
                                                    phase.name)

            if traj is None:
                if phase.expect_contact:
                    logging.debug(
                        "[%s/%s] BiRRT failed; falling back to "
                        "incremental IK.", self._name, phase.name)
                    memory[traj_key] = None
                else:
                    detail = ""
                    if self._last_plan_diagnostics:
                        detail = (" Blocking contacts: " +
                                  "; ".join(self._last_plan_diagnostics) + ".")
                    # A GOAL-config contact means the pose commanded by the
                    # option's parameters is itself infeasible - no path
                    # could ever reach it, so say that instead of blaming
                    # path planning.
                    if any(
                            d.startswith("GOAL")
                            for d in self._last_plan_diagnostics):
                        headline = (
                            "target configuration in collision: the pose "
                            "commanded by this option's parameters "
                            f"{_fmt_option_params(params)} is itself in "
                            "contact - adjust the parameters, not the path")
                    elif any(
                            d.startswith("START")
                            for d in self._last_plan_diagnostics):
                        headline = (
                            "start configuration in collision: the robot "
                            "begins this phase already in contact")
                    else:
                        headline = ("motion planning failed (no "
                                    "collision-free path)")
                    raise utils.OptionExecutionFailure(
                        f"[{self._name}/{phase.name}] BiRRT collision: "
                        f"{headline}.{detail}")
            else:
                # Skip the first waypoint — BiRRT includes the start
                # position (current joints) as traj[0].  Commanding the
                # robot to stay at its current position is a no-op that
                # triggers the option-model "option got stuck" check
                # (option_model_terminate_on_repeat), aborting the option
                # after a single step.
                traj_list = list(traj)
                memory[traj_key] = traj_list[1:] if len(traj_list) > 1 \
                    else traj_list
            memory[step_key] = 0

            # Restore robot joints — run_motion_planning leaves them at an
            # arbitrary configuration used during collision checking.
            robot.set_joints(pb_state.joint_positions)

        traj = memory[traj_key]
        if traj is None:
            # BiRRT failed — fall back to incremental IK.
            self._check_ik_stall(phase, state, memory, objects, params)
            return self._execute_move_ik(phase, state, objects, params)

        # --- Pop next waypoint from cached trajectory. ---
        step = memory[step_key]

        if step >= len(traj):
            # Trajectory fully consumed — use incremental IK to converge
            # to the exact target pose (BiRRT's IK solution may be slightly
            # off from the target Cartesian pose).
            self._check_ik_stall(phase, state, memory, objects, params)
            return self._execute_move_ik(phase, state, objects, params)

        finger_idx_l = robot.left_finger_joint_idx
        finger_idx_r = robot.right_finger_joint_idx

        # Tracking gate: re-command the previous waypoint until the arm has
        # converged to it. Advancing one waypoint per control step regardless
        # of tracking error lets position control lag several waypoints
        # behind and cut corners off the collision-checked path — enough to
        # swing a held object centimetres past the planner's bystander
        # clearance. A hold cap keeps an unreachable waypoint from stalling
        # the phase forever.
        target_joints = traj[step]
        track_tol = CFG.pybullet_birrt_replay_track_tol
        hold_key = _BIRRT_HOLD_KEY.format(pid)
        if track_tol > 0 and step > 0:
            prev_cmd = traj[step - 1]
            arm_err = max(
                abs(cur - cmd) for idx, (
                    cur,
                    cmd) in enumerate(zip(pb_state.joint_positions, prev_cmd))
                if idx not in (finger_idx_l, finger_idx_r))
            if arm_err > track_tol and memory.get(
                    hold_key, 0) < CFG.pybullet_birrt_replay_max_hold_steps:
                memory[hold_key] = memory.get(hold_key, 0) + 1
                target_joints = prev_cmd
            else:
                memory[hold_key] = 0
                memory[step_key] = step + 1
        else:
            memory[step_key] = step + 1

        # Apply finger nudge matching the phase's finger_status, identical
        # to what incremental IK does in controllers.py.  This prevents
        # finger drift and allows finger transitions (e.g. open→closed)
        # to happen gradually during BiRRT trajectory replay.
        joint_action = list(target_joints)
        current_fingers = pb_state.joint_positions[finger_idx_l]
        finger_status = memory[finger_key]
        if finger_status == "open":
            finger_delta = self._config.finger_action_nudge_magnitude
        elif finger_status == "hold":
            finger_delta = 0.0
        else:
            finger_delta = -self._config.finger_action_nudge_magnitude
        f_action = current_fingers + finger_delta
        joint_action[finger_idx_l] = f_action
        joint_action[finger_idx_r] = f_action

        # _build_action_from_joints pads zero base deltas for mobile robots
        # (BiRRT replays a fixed-base arm trajectory) and is a no-op clip for
        # fixed-base robots, keeping the action shape matched to the robot's
        # action space.
        return _build_action_from_joints(robot, joint_action)

    # ------------------------------------------------------------------
    # BiRRT planning helpers
    # ------------------------------------------------------------------

    # Non-physical types that have no PyBullet body and should be skipped
    # when collecting collision bodies.
    _SKIP_TYPES = frozenset({
        "robot",
        "loc",
        "angle",
        "human",
        "side",
        "direction",
    })

    @staticmethod
    def _collect_sim_objects(sim: PyBulletEnv) -> Dict[str, Object]:
        """Collect all Objects with body IDs from a PyBulletEnv instance."""
        obj_map: Dict[str, Object] = {}
        # Scan instance attributes for Object instances with body IDs.
        for attr_val in sim.__dict__.values():
            if isinstance(attr_val, Object) and attr_val.id is not None:
                obj_map[attr_val.name] = attr_val
            elif isinstance(attr_val, (list, tuple)):
                for item in attr_val:
                    if isinstance(item, Object) and item.id is not None:
                        obj_map[item.name] = item
        # Composed envs: also enumerate component objects.
        for comp in getattr(sim, '_components', []):
            for obj in comp.get_objects():
                obj_map[obj.name] = obj
        # Always include the robot.
        obj_map[sim._robot.name] = sim._robot  # pylint: disable=protected-access
        return obj_map

    def _plan_without_simulator(
        self,
        pb_state: utils.PyBulletState,
        target_pose: Pose,
        phase_name: str,
    ) -> Optional[Sequence[JointPositions]]:
        """Plan using the config robot's physics client (no collision
        bodies)."""
        robot = self._config.robot
        robot.set_joints(pb_state.joint_positions)
        try:
            target_joints: JointPositions = robot.inverse_kinematics(
                target_pose,
                validate=self._config.ik_validate,
                set_joints=True)
        except InverseKinematicsError:
            pos = target_pose.position
            logging.warning(
                f"[{self._name}/{phase_name}] IK failed for BiRRT target "
                f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}); "
                "falling back to incremental IK.")
            return None

        return run_motion_planning(
            robot=robot,
            initial_positions=pb_state.joint_positions,
            target_positions=target_joints,
            collision_bodies=self._config.collision_bodies,
            seed=CFG.seed,
            physics_client_id=robot.physics_client_id,
        )

    def _sim_collision_context(
        self, pb_state: utils.PyBulletState
    ) -> Tuple[utils.PyBulletState, set, Dict[int, str], Optional[int]]:
        """Remap ``pb_state`` onto the planning simulator and collect its
        collision bodies.

        Resets the simulator to the remapped state as a side effect.
        Returns ``(remapped_state, collision_bodies, body_names,
        held_object)``. Requires ``self._config.simulator``.
        """
        sim = self._config.simulator
        assert sim is not None

        # 1. Build name -> simulator Object mapping
        sim_obj_map = self._collect_sim_objects(sim)

        # 2. Remap state: simulator Objects with original feature values
        new_state_data: Dict[Object, Any] = {}
        for orig_obj, features in pb_state.data.items():
            sim_obj = sim_obj_map.get(orig_obj.name)
            if sim_obj is not None:
                new_state_data[sim_obj] = features.copy()

        remapped_state = utils.PyBulletState(
            new_state_data, simulator_state=pb_state.simulator_state)

        # 3. Reset simulator to current state
        sim._set_state(remapped_state)  # pylint: disable=protected-access

        # 4. Collect collision body IDs (exclude held objects and
        #    non-physical types) and find the held object.
        collision_bodies: set = set()
        body_names: Dict[int, str] = {}
        held_object: Optional[int] = None
        for orig_obj in pb_state:
            if orig_obj.type.name in self._SKIP_TYPES or \
                    orig_obj.type.name in self._config.collision_skip_types:
                continue
            sim_obj = sim_obj_map.get(orig_obj.name)
            if sim_obj is None or sim_obj.id is None:
                continue
            body_names[sim_obj.id] = orig_obj.name
            if "is_held" in orig_obj.type.feature_names and \
                    pb_state.get(orig_obj, "is_held") > 0.5:
                held_object = sim_obj.id
                continue
            collision_bodies.add(sim_obj.id)

        # 4a. Exclude bodies weld-attached to the held object (e.g. a glued
        #     assembly transported as a rigid unit, see pybullet_bridge).
        #     They travel with the grasped body, so treating them as static
        #     obstacles would make every transport plan collide immediately.
        #     Conservative approximation: welded partners sweep unchecked.
        if held_object is not None:
            get_welded = getattr(sim, "get_welded_partner_ids", None)
            if get_welded is not None:
                collision_bodies -= set(get_welded(held_object))

        # 4b. Add tables if present.
        if hasattr(sim, '_table_ids'):
            for tid in sim._table_ids:  # pylint: disable=protected-access
                collision_bodies.add(tid)
        elif hasattr(sim, '_table') and sim._table.id is not None:  # pylint: disable=protected-access
            collision_bodies.add(sim._table.id)  # pylint: disable=protected-access

        # 4c. Add extra sim collision bodies (e.g. virtual buffer zones).
        collision_bodies.update(self._config.sim_extra_collision_bodies)

        # 4d. Add environment-specific extra collision bodies (e.g. liquid
        #     blocks in Grow that aren't tracked as state Objects).
        collision_bodies.update(sim.get_extra_collision_ids())

        return remapped_state, collision_bodies, body_names, held_object

    def _stall_contact_report(self, pb_state: utils.PyBulletState) -> str:
        """Name the bodies the robot is touching when incremental IK stalls.

        A stall usually means an obstacle sits between the end effector
        and the phase target, so the contacting bodies are the best
        available explanation. Runs on the planning simulator; returns
        ``""`` when no simulator is configured or nothing is in contact
        (or the report fails - this is a best-effort diagnostic on an
        error path).
        """
        if self._config.simulator is None:
            return ""
        try:
            sim = self._config.simulator
            _, collision_bodies, body_names, held_object = \
                self._sim_collision_context(pb_state)
            planning_robot = sim._pybullet_robot  # pylint: disable=protected-access
            planning_robot.set_joints(pb_state.joint_positions)
            client = sim._physics_client_id  # pylint: disable=protected-access
            p.performCollisionDetection(physicsClientId=client)
            # Report against the wider (positive) threshold, mirroring
            # _log_collision_diagnostics: pybullet_birrt_contact_margin is
            # the NEGATIVE penetration allowance (-1mm), and a stall
            # typically presses at ~0 separation - filtering by the
            # negative margin would report nothing exactly when the agent
            # needs the blocker named.
            margin = max(CFG.pybullet_birrt_contact_margin,
                         CFG.pybullet_birrt_bystander_clearance)
            touching = []
            for body in sorted(collision_bodies):
                label = body_names.get(body, f"body {body}")
                for probe, probe_label in ((planning_robot.robot_id, "robot"),
                                           (held_object, "held object")):
                    if probe is None:
                        continue
                    contacts = p.getContactPoints(probe,
                                                  body,
                                                  physicsClientId=client)
                    if any(c[8] < margin for c in contacts):
                        min_dist = min(c[8] for c in contacts)
                        touching.append(f"{probe_label} within "
                                        f"{min_dist:.4f} m of {label}")
        except Exception:  # pylint: disable=broad-except
            return ""
        if not touching:
            return ""
        return " In contact: " + "; ".join(touching) + "."

    def _plan_with_simulator(
        self,
        pb_state: utils.PyBulletState,
        target_pose: Pose,
        phase_name: str,
        expect_contact: bool = False,
        objects: Sequence[Object] = (),
        phase: Optional[Phase] = None,
    ) -> Optional[Sequence[JointPositions]]:
        """Plan using the simulator env for collision-aware motion planning.

        Remaps the current state onto the simulator's objects, resets
        the simulator, collects collision body IDs, and runs IK + BiRRT
        on the simulator's physics client.
        """
        del objects  # Unused; kept for a uniform planner signature.
        sim = self._config.simulator
        assert sim is not None
        remapped_state, collision_bodies, body_names, held_object = \
            self._sim_collision_context(pb_state)

        # 5. IK + motion planning on simulator's robot
        planning_robot = sim._pybullet_robot  # pylint: disable=protected-access
        planning_robot.set_joints(pb_state.joint_positions)

        # Compute base_link_to_held_obj if an object is held (needed both for
        # motion planning and the collision-aware IK below).
        base_link_to_held_obj = None
        if held_object is not None and sim._held_obj_to_base_link is not None:  # pylint: disable=protected-access
            base_link_to_held_obj = p.invertTransform(
                *sim._held_obj_to_base_link)  # pylint: disable=protected-access

        # Validate the goal IK when globally enabled, or when this phase
        # requests it (e.g. a grasp approach, where an imprecise goal config
        # clips the target object and BiRRT then rejects a reachable grasp).
        validate_goal_ik = self._config.ik_validate or (phase is not None
                                                        and phase.validate_ik)
        try:
            target_joints: JointPositions = self._solve_goal_ik(
                planning_robot, target_pose, pb_state.joint_positions,
                validate_goal_ik)
        except InverseKinematicsError:
            pos = target_pose.position
            logging.warning(
                f"[{self._name}/{phase_name}] IK failed for BiRRT target "
                f"({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}); "
                "falling back to incremental IK.")
            return None

        goal_finger_joint = None
        if phase is not None and phase.check_release_clearance:
            # Check the width the fingers actually reach at the drop pose:
            # with a grasp-relative release, the measured grasp width plus
            # the worst-case release travel; else the legacy full open.
            if self._config.release_until_ungrasped:
                grasp_width = pb_state.joint_positions[
                    planning_robot.left_finger_joint_idx]
                goal_finger_joint = min(self._config.open_fingers_joint,
                                        grasp_width + _RELEASE_CHECK_BUFFER)
            else:
                goal_finger_joint = self._config.open_fingers_joint

        traj = run_motion_planning(
            robot=planning_robot,
            initial_positions=pb_state.joint_positions,
            target_positions=target_joints,
            collision_bodies=collision_bodies,
            seed=CFG.seed,
            physics_client_id=sim._physics_client_id,  # pylint: disable=protected-access
            held_object=held_object,
            base_link_to_held_obj=base_link_to_held_obj,
            allow_shallow_held_object_contacts=(
                phase.allow_shallow_held_object_contacts
                if phase is not None else False),
            goal_finger_joint=goal_finger_joint,
            held_bystander_clearance=self._config.held_bystander_clearance,
        )

        if traj is None and not validate_goal_ik:
            # The unvalidated goal solve may have accepted a one-shot IK
            # branch whose carried object is in collision. Before declaring
            # the option infeasible, retry with the fully validated goal-IK
            # stack (same restart machinery), which can land a different
            # in-limit branch whose goal configuration is collision-free.
            sim._set_state(remapped_state)  # pylint: disable=protected-access
            planning_robot.set_joints(pb_state.joint_positions)
            validated_target_joints: Optional[JointPositions] = None
            try:
                validated_target_joints = self._solve_goal_ik(
                    planning_robot,
                    target_pose,
                    pb_state.joint_positions,
                    validate=True)
            except InverseKinematicsError:
                pass
            if validated_target_joints is not None and \
                    validated_target_joints != target_joints:
                traj = run_motion_planning(
                    robot=planning_robot,
                    initial_positions=pb_state.joint_positions,
                    target_positions=validated_target_joints,
                    collision_bodies=collision_bodies,
                    seed=CFG.seed,
                    physics_client_id=sim._physics_client_id,  # pylint: disable=protected-access
                    held_object=held_object,
                    base_link_to_held_obj=base_link_to_held_obj,
                    allow_shallow_held_object_contacts=(
                        phase.allow_shallow_held_object_contacts
                        if phase is not None else False),
                    goal_finger_joint=goal_finger_joint,
                    held_bystander_clearance=(
                        self._config.held_bystander_clearance),
                )
                if traj is not None:
                    target_joints = validated_target_joints

        if traj is None and not expect_contact:
            self._last_plan_diagnostics = self._log_collision_diagnostics(
                planning_robot,
                sim._physics_client_id,  # pylint: disable=protected-access
                pb_state.joint_positions,
                target_joints,
                collision_bodies,
                held_object,
                base_link_to_held_obj,
                phase_name,
                body_names=body_names,
                goal_finger_joint=goal_finger_joint)

        return traj

    def _solve_goal_ik(self, planning_robot: SingleArmPyBulletRobot,
                       target_pose: Pose, current_joints: JointPositions,
                       validate: bool) -> JointPositions:
        """Goal-config IK that is accurate AFTER joint-limit clamping.

        PyBullet IK is a one-shot approximation with no accuracy
        guarantee (a far seed can miss by centimeters) and it ignores
        joint limits, following the branch of its seed configuration.
        Position control clamps to the limits at execution, so BiRRT
        would plan to a configuration whose executed end effector misses
        the target pose: the goal-config collision check then tests the
        wrong pose (a carried object can falsely "collide" with the
        table it was meant to hover over), and the distance-based phase
        terminal never fires. Every candidate is therefore accepted only
        when its limit-clamped version hits the pose within
        ``move_to_pose_tol`` under forward kinematics. When ``validate``
        is False, the cheap unvalidated one-shot is tried first and the
        SAME seed escalates to validated (iterated) IK if it misses.
        Seeds: the current joints, the home configuration, then
        deterministic random in-limit restarts. Raise
        ``InverseKinematicsError`` when no attempt produces an
        acceptable config.
        """
        limits = list(
            zip(planning_robot.joint_lower_limits,
                planning_robot.joint_upper_limits))
        seeds: List[JointPositions] = [
            list(current_joints),
            list(planning_robot.initial_joint_positions),
        ]
        rng = np.random.default_rng(CFG.seed)
        for _ in range(self._goal_ik_num_restarts):
            seeds.append([
                float(rng.uniform(lo, hi))
                if np.isfinite(lo) and np.isfinite(hi) and lo <= hi else float(
                    rng.uniform(cur - np.pi, cur + np.pi))
                for (lo, hi), cur in zip(limits, current_joints)
            ])
        best_err = float("inf")
        for seed in seeds:
            for attempt_validate in ((True, ) if validate else (False, True)):
                planning_robot.set_joints(seed)
                try:
                    candidate = planning_robot.inverse_kinematics(
                        target_pose,
                        validate=attempt_validate,
                        set_joints=True)
                except InverseKinematicsError:
                    continue
                clamped = [
                    float(np.clip(v, lo, hi)) if lo <= hi else float(v)
                    for v, (lo, hi) in zip(candidate, limits)
                ]
                ee_position = planning_robot.forward_kinematics(
                    clamped).position
                err = float(
                    np.sum(
                        np.square(
                            np.subtract(ee_position, target_pose.position))))
                if err < self._config.move_to_pose_tol:
                    return clamped
                best_err = min(best_err, err)
        raise InverseKinematicsError(
            f"Goal IK missed the target pose from all {len(seeds)} seeds "
            f"(best squared FK error after limit clamping {best_err:.6f}).")

    def _log_collision_diagnostics(
        self,
        planning_robot: SingleArmPyBulletRobot,
        physics_client_id: int,
        start_joints: JointPositions,
        goal_joints: JointPositions,
        collision_bodies: set,
        held_object: Optional[int],
        base_link_to_held_obj: Optional[Any],
        phase_name: str,
        body_names: Optional[Dict[int, str]] = None,
        goal_finger_joint: Optional[float] = None,
    ) -> List[str]:
        """Log which collision bodies cause start/goal collisions.

        Returns the diagnostic strings so callers can attach them to the
        ``OptionExecutionFailure`` - in the agent's sandbox that message
        is the only channel through which it learns WHICH object blocked
        the motion plan (and hence how to adjust its target pose).
        """
        from predicators.pybullet_helpers.link import \
            get_link_state  # pylint: disable=import-outside-toplevel
        diagnostics: List[str] = []

        def _body_label(body: int) -> str:
            if body_names and body in body_names:
                return body_names[body]
            body_name = ""
            try:
                body_name = p.getBodyInfo(
                    body, physicsClientId=physics_client_id)[1].decode()
            except Exception:  # pylint: disable=broad-except
                pass
            return f"body {body} ({body_name})"

        def _check(joints: JointPositions, label: str) -> None:
            planning_robot.set_joints(joints)
            if held_object is not None and base_link_to_held_obj is not None:
                wt_bl = get_link_state(
                    planning_robot.robot_id,
                    planning_robot.end_effector_id,
                    physics_client_id=physics_client_id).com_pose
                wt_ho = p.multiplyTransforms(wt_bl[0], wt_bl[1],
                                             base_link_to_held_obj[0],
                                             base_link_to_held_obj[1])
                p.resetBasePositionAndOrientation(
                    held_object,
                    wt_ho[0],
                    wt_ho[1],
                    physicsClientId=physics_client_id)
            p.performCollisionDetection(physicsClientId=physics_client_id)
            # Report against the wider of the two thresholds so that
            # bystander-clearance failures (positive separations) are
            # explained, not just hard penetrations.
            margin = max(CFG.pybullet_birrt_contact_margin,
                         CFG.pybullet_birrt_bystander_clearance)
            for body in collision_bodies:
                contacts = p.getContactPoints(
                    planning_robot.robot_id,
                    body,
                    physicsClientId=physics_client_id)
                if any(c[8] < margin for c in contacts):
                    min_dist = min(c[8] for c in contacts)
                    diagnostics.append(
                        f"{label}: robot within {min_dist:.4f} m of "
                        f"{_body_label(body)}")
                if held_object is not None:
                    contacts = p.getContactPoints(
                        held_object, body, physicsClientId=physics_client_id)
                    if any(c[8] < margin for c in contacts):
                        min_dist = min(c[8] for c in contacts)
                        diagnostics.append(
                            f"{label}: held object within {min_dist:.4f} m "
                            f"of {_body_label(body)}")

        _check(start_joints, "START")
        _check(goal_joints, "GOAL")
        if goal_finger_joint is not None:
            release_joints = list(goal_joints)
            release_joints[planning_robot.left_finger_joint_idx] = \
                goal_finger_joint
            release_joints[planning_robot.right_finger_joint_idx] = \
                goal_finger_joint
            _check(
                release_joints,
                "GOAL with fingers OPEN to release (the opening "
                "gripper needs side clearance at the drop pose)")
        for diag in diagnostics:
            logging.error(f"[{self._name}/{phase_name}] {diag}")
        return diagnostics

    def _execute_move_ik(self, phase: Phase, state: State,
                         objects: Sequence[Object], params: Array) -> Action:
        """Execute a MOVE_TO_POSE phase using incremental IK delta-stepping."""
        pb_state = cast(utils.PyBulletState, state)
        robot = self._config.robot
        robot.set_joints(pb_state.joint_positions)
        current_pose, target_pose, finger_status = phase.target_fn(
            state, objects, params, self._config)
        try:
            return get_move_end_effector_to_pose_action(
                robot=robot,
                current_joint_positions=pb_state.joint_positions,
                current_pose=current_pose,
                target_pose=target_pose,
                finger_status=finger_status,
                max_vel_norm=self._config.max_vel_norm,
                finger_action_nudge_magnitude=(
                    self._config.finger_action_nudge_magnitude),
                validate=self._config.ik_validate,
                # Base positioning is handled once per option by
                # _maybe_drive_base; keep incremental IK arm-only so the base
                # doesn't drift during contact phases (e.g. a switch push).
                move_base=False,
            )
        except utils.OptionExecutionFailure:
            cur = current_pose.position
            tgt = target_pose.position
            raise utils.OptionExecutionFailure(
                f"[{self._name}/{phase.name}] IK failed. "
                f"current=({cur[0]:.3f}, {cur[1]:.3f}, {cur[2]:.3f}), "
                f"target=({tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}), "
                f"params={params.tolist()}")

    def _execute_fingers(self, phase: Phase, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> Action:
        """Execute a CHANGE_FINGERS phase."""
        pb_state = cast(utils.PyBulletState, state)
        current_val, target_val = self._finger_target(phase, state, memory,
                                                      objects, params)
        return get_change_fingers_action(
            self._config.robot,
            pb_state.joint_positions,
            current_val,
            target_val,
            self._config.max_vel_norm,
        )
