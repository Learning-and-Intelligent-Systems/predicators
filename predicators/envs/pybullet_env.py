"""Base class for a PyBullet environment.

Provides common functionality for PyBullet-based robotic manipulation
environments including robot control, state synchronization, grasp detection,
and rendering.

For a comprehensive guide on creating new PyBullet environments, see:
    docs/pybullet_env_guide.md

Main public API:
    reset(train_or_test, task_idx) — reset env to a task, returns observation
    simulate(state, action) — forward-simulate without touching real env
    step(action) — _step_base (robot control, physics, grasps)
        → _domain_specific_step (water filling, heating, etc.)
        → get_observation. Domain dynamics are skipped when
        skip_residual_dynamics=True is passed to the constructor.
    get_observation() — read PyBullet state, optionally attach images/masks

State synchronization:
    _set_state(state) — write a State into PyBullet (robot pose, object
        poses, grasp constraints). Delegates domain-specific setup to
        _set_domain_specific_state().
    _get_state() — read PyBullet into a PyBulletState. Delegates
        domain-specific features to _get_domain_specific_feature().

Required overrides in subclasses:
    - get_name() -> str
    - initialize_pybullet(using_gui) -> (physics_id, robot, bodies_dict)
    - _store_pybullet_bodies(bodies_dict)
    - _get_object_ids_for_held_check() -> List[int]
    - _set_domain_specific_state(state)
    - _get_domain_specific_feature(obj, feature) -> float
    - _domain_specific_step() (optional, default no-op)
"""

import abc
import logging
from typing import Any, ClassVar, Dict, List, Optional, Protocol, Sequence, \
    Set, Tuple, Type, cast

import matplotlib
import numpy as np
import pybullet as p
from gym.spaces import Box
from PIL import Image

from predicators import utils
from predicators.code_sim_learning.commands import ApplyForce, ApplyTorque, \
    PhysicsCommand, SetVelocity
from predicators.envs import BaseEnv
from predicators.pybullet_helpers import retry_pybullet_call, studio_visuals
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.objects import update_object
from predicators.pybullet_helpers.real_robot_bridge import \
    GripperJointLayout, gripper_joint_layout_from_robot
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot, get_robot_home_ee_position
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, Mask, Object, \
    Observation, State, Video
from predicators.utils import PyBulletState

# The robot_init_{x,y,z} values env classes declare, saved per declaring
# class before a robot with a home configuration overwrites them (see
# _sync_robot_init_pos_with_home), so that they can be restored for robots
# that have none. Keyed per attribute so that a subclass inheriting a
# parent's already-synced value can still recover what the parent declared.
_DECLARED_ROBOT_INIT_POS: Dict[Tuple[Type["PyBulletEnv"], str], float] = {}


class ActionExecutor(Protocol):
    """Drives real hardware from a simulated env's rollout.

    The port an env exposes for "something outside is executing what I
    just simulated". Declared here, next to the four calls, and
    implemented elsewhere -- ``pybullet_helpers.real_robot_executor`` --
    so this module never imports anything that knows about a robot.

    An env with no executor attached is pure simulation.
    """

    def tasks_for(self, train_or_test: str) -> Optional[List[EnvironmentTask]]:
        """Tasks to use for this split, or None to keep the env's own.

        Asked *every* time the tasks are requested, because for a real
        environment "give me the train task" genuinely means "look at
        the world". An executor that has nothing new to say returns None
        and the env's own generated (and cached) tasks stand.
        """

    def after_reset(self, train_or_test: str, task_idx: int,
                    obs: Observation) -> None:
        """The env has been reset to a task's initial state."""

    def after_step(self, action: Action, obs: Observation) -> Observation:
        """The env has simulated ``action``.

        Returns the observation the caller should see, which may differ
        from ``obs``: an executor that looked at the real world and
        corrected the simulated twin returns the corrected reading.
        """

    def after_episode(self, completed: bool) -> None:
        """The episode is over; nothing more will be simulated.

        ``completed`` is False when the episode ended abnormally (an
        exception, or the step limit reached mid-option), which is the
        difference between "this plan ran to its end" and "this is
        however far it got". An executor holding deferred work decides
        on that flag whether to do it or drop it.

        Called exactly once per episode, on every exit path.
        """


class PyBulletEnv(BaseEnv):
    """Base class for a PyBullet environment."""
    # Parameters that aren't important enough to need to clog up settings.py

    # General robot parameters.
    # grasp_tol: value for which the objects with distance below to are
    # considered to be grasped, and also the value change finger option can be
    # terminated.
    grasp_tol: ClassVar[float] = 5e-2  # for large objects
    grasp_tol_small: ClassVar[float] = 5e-4  # for small objects
    _finger_action_tol: ClassVar[float] = 1e-4
    open_fingers: ClassVar[float] = 0.04
    closed_fingers: ClassVar[float] = 0.01
    robot_init_x: ClassVar[float]
    robot_init_y: ClassVar[float]
    robot_init_z: ClassVar[float]
    # Default initial EE orientation (Euler). Subclasses may override.
    # Used by per-env task-init dicts when populating the robot's
    # roll/tilt/wrist features.
    robot_init_roll: ClassVar[float] = 0.0
    robot_init_tilt: ClassVar[float] = 0.0
    robot_init_wrist: ClassVar[float] = 0.0
    y_lb: ClassVar[float]
    y_ub: ClassVar[float]
    robot_base_pos: ClassVar[Optional[Tuple[float, float, float]]] = None
    robot_base_orn: ClassVar[Optional[Tuple[float, float, float,
                                            float]]] = None

    # Object parameters.
    _obj_mass: ClassVar[float] = 0.5
    _obj_friction: ClassVar[float] = 1.2
    _obj_colors_main: ClassVar[List[Tuple[float, float, float,
                                          float]]] = [(0.95, 0.05, 0.1, 1.),
                                                      (0.05, 0.95, 0.1, 1.),
                                                      (0.1, 0.05, 0.95, 1.),
                                                      (0.4, 0.05, 0.6, 1.),
                                                      (0.6, 0.4, 0.05, 1.),
                                                      (0.05, 0.04, 0.6, 1.),
                                                      (0.95, 0.95, 0.1, 1.),
                                                      (0.95, 0.05, 0.95, 1.),
                                                      (0.05, 0.95, 0.95, 1.)]
    _obj_colors: ClassVar[List[Tuple[float, float, float, float]]] =\
        _obj_colors_main + [
        (0.941, 0.196, 0.196, 1.),  # Red
        (0.196, 0.941, 0.196, 1.),  # Green
        (0.196, 0.196, 0.941, 1.),  # Blue
        (0.941, 0.941, 0.196, 1.),  # Yellow
        (0.941, 0.196, 0.941, 1.),  # Magenta
        (0.196, 0.941, 0.941, 1.),  # Cyan
        (0.941, 0.588, 0.196, 1.),  # Orange
        (0.588, 0.196, 0.941, 1.),  # Purple
        (0.196, 0.941, 0.588, 1.),  # Teal
        (0.941, 0.196, 0.588, 1.),  # Pink
        (0.588, 0.941, 0.196, 1.),  # Lime
        (0.196, 0.588, 0.941, 1.),  # Sky Blue
    ]
    _out_of_view_xy: ClassVar[Sequence[float]] = [10.0, 10.0]
    _default_orn: ClassVar[Sequence[float]] = [0.0, 0.0, 0.0, 1.0]

    # Object types that have no PyBullet body — features managed
    # entirely by _get_domain_specific_feature().
    _VIRTUAL_OBJECT_TYPES: ClassVar[frozenset] = frozenset(
        {"loc", "angle", "human", "side", "direction"})

    # Features whose values are angles in radians; comparisons should
    # treat them modulo 2π so a State that carries wrist=4.68 (out of
    # the canonical range PyBullet reports) round-trips against
    # _get_state's wrist=-1.60 without firing the reconstruction warning.
    _ANGLE_FEATURES: ClassVar[frozenset] = frozenset(
        {"rot", "yaw", "roll", "pitch", "tilt", "wrist"})

    # Euler-angle features that jointly encode one full 3D orientation must
    # be compared as a *rotation*, not axis-by-axis. At gimbal lock (e.g. the
    # EE pointing straight down, tilt=±π/2) the individual angles are
    # numerically degenerate — only the rotation they jointly encode is
    # meaningful — so an axis-by-axis compare reports up to π of spurious
    # error on the *same* physical orientation (a different but equivalent
    # gimbal-lock branch). (roll, tilt, wrist) is the robot EE orientation,
    # built by _extract_robot_state via getQuaternionFromEuler([roll, tilt,
    # wrist]); it is the only free-SO(3) triple here (only the robot carries
    # tilt/wrist). _reconstruction_diff groups these and compares the
    # geodesic angle between the two rotations instead of each axis.
    _ORIENTATION_EULER_TRIPLES: ClassVar[Tuple[Tuple[str, str, str],
                                               ...]] = (("roll", "tilt",
                                                         "wrist"), )

    # _set_state round-trips the written state through _get_state and
    # compares, then reacts by mismatch *magnitude* — no per-env opt-in:
    #   * any feature off by more than _reconstruction_warn_atol → warn,
    #   * any feature off by more than _reconstruction_raise_atol → raise.
    # Valid States legitimately fail to round-trip exactly for two reasons:
    # the generic reset path reconstructs the robot via IK from the EE pose
    # (dropping wrist roll → benign ~0.02 rad noise), and some envs store a
    # feature symbolically while placing the body elsewhere (e.g. pybullet_fan
    # positions fans by their side, not their State x/y → up to ~0.8 m of
    # benign workspace-scale disagreement). The raise threshold sits well
    # above both (~2.5x the worst observed) yet far below an impossible or
    # corrupt requested feature (e.g. held=-10000, off by 1e4), so only the
    # latter aborts — for every env, with no per-env strictness flag.
    _reconstruction_warn_atol: ClassVar[float] = 1e-3
    _reconstruction_raise_atol: ClassVar[float] = 2.0

    # Camera parameters.
    _camera_distance: ClassVar[float] = 0.8
    _camera_yaw: ClassVar[float] = 90.0
    _camera_pitch: ClassVar[float] = -24
    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.42)
    _camera_fov: ClassVar[float] = 60
    _debug_text_position: ClassVar[Pose3D] = (1.65, 0.25, 0.75)

    # Offscreen-render lighting (used by render()). Shadows plus a directional
    # key light give saved frames depth instead of flat ambient shading.
    _render_shadow: ClassVar[int] = 1
    # Key-light direction. When None it is derived from the camera (a front
    # key from the camera's side, elevated) so it lights camera-facing
    # surfaces for any env; set a Pose3D to override.
    _render_light_direction: ClassVar[Optional[Pose3D]] = None
    _render_light_ambient: ClassVar[float] = 0.55
    _render_light_diffuse: ClassVar[float] = 0.6
    _render_light_specular: ClassVar[float] = 0.05

    # Studio visuals: shared cosmetic scene dressing applied automatically by
    # the base initialize_pybullet (neutral GUI background + key light +
    # shadows, recolored floor, backdrop walls; see the studio_visuals helper).
    # Visual-only -- walls carry no collision and none of this enters the
    # symbolic state. Set _use_studio_visuals = False on an env to opt out.
    _use_studio_visuals: ClassVar[bool] = True
    # Muted neutral floor (recolors the ground plane).
    floor_rgba: ClassVar[Optional[Tuple[float, float, float, float]]] = \
        (0.50, 0.51, 0.53, 1.0)
    # Light maple table texture, forwarded to create_object(texture_path=...)
    # by envs that texture their table (currently the domino envs).
    table_texture_path: ClassVar[Optional[str]] = "urdf/table.png"
    # Backdrop walls: wall_texture_path (warm matte paint) takes precedence
    # over wall_rgba. _wall_bounds (world frame) sets the enclosure; when None
    # it is derived from the camera so the room centers on the view with the
    # camera inside. Four walls, no ceiling (overhead views still see in).
    wall_rgba: ClassVar[Tuple[float, float, float, float]] = \
        (0.85, 0.83, 0.79, 1.0)
    wall_texture_path: ClassVar[Optional[str]] = "urdf/textures/wall.png"
    _wall_bounds: ClassVar[Optional[Dict[str, float]]] = None
    # Camera-derived room (used when _wall_bounds is None): half-extent and
    # height as multiples of the camera distance, plus wall thickness.
    _studio_room_half_factor: ClassVar[float] = 1.85
    _studio_room_height_factor: ClassVar[float] = 1.75
    _studio_room_thickness: ClassVar[float] = 0.05
    # Elevation (world z) of the camera-derived key-light direction.
    _studio_light_elevation: ClassVar[float] = 1.8
    # GUI window appearance (forwarded to create_gui_connection). A neutral
    # background reads far more like a real scene than PyBullet's lavender;
    # _gui_light_position is derived from the camera when None.
    _gui_background_rgb: ClassVar[Optional[Tuple[float, float, float]]] = \
        (0.82, 0.83, 0.85)
    _gui_light_position: ClassVar[Optional[Tuple[float, float, float]]] = None
    _gui_shadow_map_resolution: ClassVar[Optional[int]] = 8192
    _gui_shadow_map_world_size: ClassVar[Optional[int]] = 6

    def __init__(self,
                 use_gui: bool = False,
                 skip_residual_dynamics: bool = False) -> None:
        super().__init__(use_gui)

        # Forward declaration: subclasses must define _robot
        # before using methods that access it (like
        # _extract_robot_state, _get_robot_state_dict, etc.)
        self._robot: Object

        # When an object is held, a constraint is created to prevent slippage.
        self._held_constraint_id: Optional[int] = None
        self._held_obj_to_base_link: Optional[Any] = None
        self._held_obj_id: Optional[int] = None

        # When True, _domain_specific_step() is skipped in step().
        # Used by sim-learning to create base-sim-only envs.
        self._skip_domain_specific_dynamics: bool = skip_residual_dynamics

        # Drives real hardware from this env's rollouts; None means pure sim,
        # which is what every env built by the planner stays.
        self._executor: Optional[ActionExecutor] = None

        # Residual physics commands awaiting the next action's substeps;
        # see queue_residual_commands for the contract.
        self._pending_residual_commands: List[PhysicsCommand] = []

        # Set up all the static PyBullet content.
        self._physics_client_id, self._pybullet_robot, pybullet_bodies = \
            self.initialize_pybullet(self.using_gui)
        self._store_pybullet_bodies(pybullet_bodies)
        # Texture any table(s) the env registered (every env uses the
        # "table_id"/"table_id2" convention) with the studio wood texture.
        studio_visuals.apply_table_textures(type(self),
                                            self._physics_client_id,
                                            pybullet_bodies)

        # Populated by reset() / _set_state(); used by _get_state(),
        # _set_state(), and render_segmented_obj() for iteration.
        self._objects: List[Object] = []

        # Populated by _set_state(): (object, feature) pairs whose value the
        # reset could not reproduce — e.g. an observable derived from a
        # hidden sim-feature (bubbling_level from heat_level), which a State
        # carrying only observables cannot round-trip. Combined simulators
        # read this to restore the carried value after a backtracking reset,
        # so a learned rule that reads its own emitted feature still sees the
        # right input. Empty on sequential rollouts (no reset → nothing lost).
        self._last_unreconstructible_features: List[Tuple[Object, str]] = []

        # Opt-in contact recording (see start_contact_recording): while
        # the log is not None, every step() appends that step's contact
        # pairs. Off by default - recording every rollout would tax large
        # parameter sweeps for data nobody reads.
        self._contact_log: Optional[List[Tuple[int, int, int, int, int]]] = \
            None
        self._contact_step_count = 0
        self._contact_relevant_ids: Optional[Set[int]] = None

    # ── Setup & Initialization ──────────────────────────────────

    @classmethod
    def initialize_pybullet(
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Initialize the PyBullet environment.

        This method initializes the PyBullet physics
        simulation, loads the robot and shared object
        models, and returns the physics client ID, the
        robot instance, and a dictionary containing other
        object IDs and any additional information that
        needs to be tracked.

        Args:
            using_gui: If True, the PyBullet GUI is used.
                Otherwise, simulation runs headless.

        Returns:
            Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
                - int: The physics client ID.
                - SingleArmPyBulletRobot: The robot instance.
                - Dict[str, Any]: A dictionary containing object IDs and other
                                information from PyBullet that needs to be
                                tracked.

        Notes:
            - This is a public class method because it is also used by the
            oracle options.
            - This method loads object models that are shared across tasks.
            These objects can have different poses or colors, and the number of
            objects can vary across tasks (e.g., the number of blocks in the
            blocks domain). However, an object's size cannot be changed after
            loading.
            - Task-specific objects that need to be loaded with different sizes
            or other properties should be handled in the
            `_set_domain_specific_state` method, which is called during each
            task's reset.
            - Subclasses may override this method to load additional assets. In
            the subclass, register all object IDs here and move them out of view
            in the `_set_domain_specific_state` method.
        """
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if using_gui:  # pragma: no cover
            physics_client_id = studio_visuals.make_gui_connection(cls)
        else:
            physics_client_id = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=physics_client_id)

        # Load plane and apply the studio floor recolor.
        plane_id = p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"),
                              [0, 0, 0],
                              useFixedBase=True,
                              physicsClientId=physics_client_id)
        studio_visuals.apply_floor(cls, plane_id, physics_client_id)

        # Load robot.
        pybullet_robot = cls._create_pybullet_robot(physics_client_id)

        # Set gravity.
        p.setGravity(0., 0., -10., physicsClientId=physics_client_id)

        # Backdrop walls (visual only) to ground the scene like a room.
        studio_visuals.create_walls(cls, physics_client_id)

        return physics_client_id, pybullet_robot, {}

    @abc.abstractmethod
    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store any bodies created in cls.initialize_pybullet().

        This is separate from the initialization because the
        initialization is a class method (which is needed for options).
        Subclasses should decide what bodies to keep.
        """
        raise NotImplementedError("Override me!")

    @classmethod
    def _create_pybullet_robot(
            cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        """Instantiate the robot model.

        Called by initialize_pybullet().
        """
        if cls.robot_base_pos is None or cls.robot_base_orn is None:
            base_pose = None
        else:
            base_pose = Pose(cls.robot_base_pos, cls.robot_base_orn)

        home_position = get_robot_home_ee_position(CFG.pybullet_robot,
                                                   base_pose)
        cls._sync_robot_init_pos_with_home(home_position)
        ee_home = Pose((cls.robot_init_x, cls.robot_init_y, cls.robot_init_z),
                       cls.get_robot_ee_init_orn(home_position is not None))

        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home,
                                                base_pose)

    @classmethod
    def get_robot_ee_init_orn(cls, has_home_config: bool) -> Quaternion:
        """The end-effector orientation to home the robot to.

        A robot with a home configuration must home to the orientation its
        initial state describes. Otherwise it homes to one orientation and
        every state reset runs IK to another.

        Robots without a home configuration (the Fetch) keep homing to
        get_robot_ee_home_orn(). Their home orientation can likewise disagree
        with their initial state's, but they have no canonical branch to
        protect, and their initial_joint_positions seed the oracle's motion
        planning -- so moving their home perturbs plans that work today.

        Reads the class-level _robot_type. An env that only assigns
        _robot_type in __init__ (e.g. blocks, coffee) is invisible here
        and falls back to get_robot_ee_home_orn().
        """
        robot_type = getattr(cls, "_robot_type", None)
        if not has_home_config or robot_type is None:
            return cls.get_robot_ee_home_orn()
        names = robot_type.feature_names
        default_roll, default_tilt, default_wrist = p.getEulerFromQuaternion(
            cls.get_robot_ee_home_orn())
        roll = cls.robot_init_roll if "roll" in names else default_roll
        tilt = cls.robot_init_tilt if "tilt" in names else default_tilt
        wrist = cls.robot_init_wrist if "wrist" in names else default_wrist
        return p.getQuaternionFromEuler([roll, tilt, wrist])

    @classmethod
    def _sync_robot_init_pos_with_home(
            cls, home_position: Optional[Pose3D]) -> None:
        """Point robot_init_{x,y,z} at the robot's home configuration, for
        robots that have one (home_position is None for robots that do not).

        Envs specify robot_init_{x,y,z} for the Fetch, whose reach is
        much longer than the Panda's. A robot with a home configuration
        ignores them and starts where it rests instead. Robots without a
        home configuration (the Fetch) keep the env's positions.
        """
        declared = cls._declared_robot_init_pos()
        if home_position is None:
            home_position = declared
        attrs = ("robot_init_x", "robot_init_y", "robot_init_z")
        for attr, declared_value, value in zip(attrs, declared, home_position):
            _DECLARED_ROBOT_INIT_POS.setdefault((cls, attr), declared_value)
            setattr(cls, attr, value)

    @classmethod
    def _declared_robot_init_pos(cls) -> Pose3D:
        """The robot_init_{x,y,z} this env declares, seeing through any values
        a previous _sync_robot_init_pos_with_home wrote over them.

        Resolved per attribute on the class that provides it, so that a
        subclass does not mistake a parent's synced value for a declared
        one.
        """
        values = []
        for attr in ("robot_init_x", "robot_init_y", "robot_init_z"):
            for klass in cls.__mro__:
                if attr in vars(klass):
                    values.append(
                        _DECLARED_ROBOT_INIT_POS.get((klass, attr),
                                                     vars(klass)[attr]))
                    break
            else:
                raise AttributeError(
                    f"{cls.__name__} does not declare {attr}.")
        return (values[0], values[1], values[2])

    @classmethod
    def get_robot_ee_home_orn(cls) -> Quaternion:
        """Return the default end-effector orientation for this env.

        Used by initialize_pybullet() to set the robot's home pose, and
        by oracle options to compute motion-planning targets.
        """
        robot_ee_orns = CFG.pybullet_robot_ee_orns[cls.get_name()]
        return robot_ee_orns[CFG.pybullet_robot]

    # ── Public API & Properties ─────────────────────────────────

    @property
    def action_space(self) -> Box:
        return self._pybullet_robot.action_space

    def get_extra_collision_ids(self) -> Sequence[int]:
        """Return extra PyBullet body IDs to treat as collision obstacles.

        Called by the motion planner (skill factories) when computing
        collision-free paths.  Override in subclasses for bodies not
        tracked as state Objects (e.g. liquid blocks in Grow).
        """
        return ()

    def get_object_by_id(self, obj_id: int) -> Object:
        """Look up an Object by its PyBullet body ID.

        Used by agent tools and skill factories to map from a PyBullet
        collision/contact result back to the predicators Object.
        """
        for obj in self._objects:
            if obj.id == obj_id:
                return obj
        raise ValueError(f"Object with ID {obj_id} not found")

    # ── Contact Recording ───────────────────────────────────────

    def start_contact_recording(self) -> None:
        """Begin recording each step's contact pairs.

        Domain-general observability for agent tools: which robot links
        touched which objects, and which object pairs touched, at which
        low-level step. Recording stays cheap (raw body/link ids in the
        hot path; name resolution deferred to stop) but is opt-in - a
        parameter sweep should not pay for a log nobody reads.
        """
        self._contact_log = []
        self._contact_step_count = 0
        # Hot-path filter: only pairs among these bodies are worth
        # logging (stop_contact_recording would discard the rest anyway,
        # and every resting object touches the table every step).
        self._contact_relevant_ids = {self._pybullet_robot.robot_id} | {
            obj.id
            for obj in self._objects if obj.id is not None
        }

    @property
    def contact_steps_recorded(self) -> int:
        """Env steps seen by the current/last contact recording.

        Lets consumers that bucket events by an external action count
        (``_attach_step_contacts``) detect a count mismatch instead of
        silently mis-windowing.
        """
        return self._contact_step_count

    def stop_contact_recording(self) -> List[Dict[str, Any]]:
        """Stop recording and return name-resolved contact events.

        Each event is ``{"step": i, "a": name, "b": name}`` (``step``
        counts env steps since recording started, 1-based) where a name
        is ``robot:<link_name>`` for a robot link or the state Object's
        name. Pairs where either side is neither the robot nor a state
        Object (table, walls, floor) are dropped: every resting object
        touches static scenery every step, and that noise would bury the
        contacts that explain motion.
        """
        log = self._contact_log or []
        self._contact_log = None
        obj_names = {
            obj.id: obj.name
            for obj in self._objects if obj.id is not None
        }
        robot_id = self._pybullet_robot.robot_id
        link_names = {-1: "base"}
        for ji in self._pybullet_robot.joint_infos:
            link_names[ji.jointIndex] = ji.linkName
        events: List[Dict[str, Any]] = []
        for step, body_a, link_a, body_b, link_b in log:
            if body_a == robot_id and body_b == robot_id:
                continue
            names: List[Optional[str]] = []
            for body, link in ((body_a, link_a), (body_b, link_b)):
                if body == robot_id:
                    names.append(
                        f"robot:{link_names.get(link, f'link{link}')}")
                elif body in obj_names:
                    names.append(obj_names[body])
                else:
                    names.append(None)
            if names[0] is None or names[1] is None:
                continue
            events.append({"step": step, "a": names[0], "b": names[1]})
        return events

    def _record_step_contacts(self) -> None:
        """Append this step's deduplicated contact pairs to the log."""
        assert self._contact_log is not None
        self._contact_step_count += 1
        if CFG.pybullet_control_mode == "reset":
            # No stepSimulation ran, so the contact manifold is stale.
            p.performCollisionDetection(
                physicsClientId=self._physics_client_id)
        relevant = self._contact_relevant_ids
        seen: Set[Tuple[int, int, int, int]] = set()
        for c in p.getContactPoints(physicsClientId=self._physics_client_id):
            # c[1]/c[2] = body ids, c[3]/c[4] = link indices,
            # c[8] = contact distance (negative = penetration).
            if c[8] > 1e-4:
                continue
            if relevant is not None and (c[1] not in relevant
                                         or c[2] not in relevant):
                continue
            key = (c[1], c[3], c[2], c[4])
            if key in seen:
                continue
            seen.add(key)
            self._contact_log.append((self._contact_step_count, *key))

    # ── Core Loop (Reset / Simulate / Step) ─────────────────────

    def reset(self,
              train_or_test: str,
              task_idx: int,
              render: bool = False) -> Observation:
        state = super().reset(train_or_test, task_idx)
        # Episode boundary: any residual commands queued for a step that
        # never ran belong to the abandoned rollout.
        self._pending_residual_commands = []
        self._set_state(state)
        observation = self.get_observation(render=render)
        if self._executor is not None:
            self._executor.after_reset(train_or_test, task_idx, observation)
        return observation

    def simulate(self, state: State, action: Action) -> State:
        """Apply an action to a state using the PyBullet simulator.

        Called by the option model during bilevel planning to forward-
        simulate candidate action sequences without touching the real
        environment.

        The _set_state guard handles two cases:
        - Skipped (common): during a sequential rollout the option model
          calls simulate(s1, a1) -> s2, then simulate(s2, a2) -> s3, etc.
          After each call, _current_state already equals the next input
          state, so _set_state is unnecessary.
        - Taken: when the planner jumps to a different state (e.g. trying
          a new skeleton or backtracking), or on the very first call
          before any reset() (_current_observation is None).
        """
        if self._current_observation is None or \
            not state.allclose(self._current_state):
            self._set_state(state)
        else:
            # Sequential rollout: PyBullet already holds this state, so no
            # reset happens and no feature is lost to reconstruction.
            self._last_unreconstructible_features = []
        # _step_once, NOT step: planning must never reach an attached executor.
        return self._step_once(action)

    def step(self, action: Action, render_obs: bool = False) -> Observation:
        """Execute one environment step with the given action.

        Flow: base sim → domain-specific dynamics → observation, then
        the attached executor (if any) gets to drive real hardware from
        that rollout and hand back a corrected observation.
        Subclasses override ``_domain_specific_step`` (not this method)
        to add post-base-sim dynamics (water filling, heating, etc.).
        """
        observation = self._step_once(action, render_obs)
        if self._executor is not None:
            observation = self._executor.after_step(action, observation)
        return observation

    def finish_execution(self, completed: bool) -> None:
        """Tell an attached executor the episode is over.

        The counterpart to ``step``'s executor hook: an executor that
        deferred work until it knew the episode's shape has no other
        moment to act on it, because nothing calls it again.
        """
        if self._executor is not None:
            self._executor.after_episode(completed)

    def _step_once(self,
                   action: Action,
                   render_obs: bool = False) -> Observation:
        """Advance the simulation one action, with no executor involved."""
        self._step_base(action)
        if not self._skip_domain_specific_dynamics:
            self._domain_specific_step()
        observation = self.get_observation(
            render=CFG.rgb_observation or render_obs)
        self._current_observation = observation
        return observation

    def _step_base(self, action: Action) -> None:
        """Run robot control, physics stepping, and grasp management."""
        # Send the action to the robot.
        target_joint_positions, base_delta = self._split_action(action)
        # Only relocate the (kinematic) base when there is an actual move.
        # Calling set_base_pose (resetBasePositionAndOrientation) every step,
        # even for a zero delta, perturbs the arm's contact dynamics — it makes
        # the mobile_fetch switch-push wander off target — whereas fixed-base
        # robots never touch the base. A zero delta is a no-op, so skip it.
        base_moved = bool(
            base_delta.size) and not bool(np.allclose(base_delta, 0.0))
        if base_moved:
            self._apply_base_delta(base_delta)
        self._pybullet_robot.set_motors(target_joint_positions.tolist())

        # If we are setting the robot joints directly, and if there is a held
        # object, we need to reset the pose of the held object directly. This
        # is because the PyBullet constraints don't seem to play nicely with
        # resetJointState (the robot will sometimes drop the object).
        #
        # The same hand-off is needed whenever the kinematic base just
        # teleported with an object in hand (mobile robots): set_base_pose jumps
        # the gripper, and over the single physics step the grasp constraint
        # would yank the object across the jump -- the jug lags, tips, or slides
        # in the gripper and then collides at the subsequent place/retreat. Pre-
        # placing it at the gripper (tracks the constant grasp offset, so this
        # is exact for a rigid grasp) makes the carry follow the base smoothly.
        if self._held_obj_id is not None and (CFG.pybullet_control_mode
                                              == "reset" or base_moved):
            world_to_base_link = get_link_state(
                self._pybullet_robot.robot_id,
                self._pybullet_robot.end_effector_id,
                physics_client_id=self._physics_client_id).com_pose
            base_link_to_held_obj = p.invertTransform(
                *self._held_obj_to_base_link)
            world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                     world_to_base_link[1],
                                                     base_link_to_held_obj[0],
                                                     base_link_to_held_obj[1])
            p.resetBasePositionAndOrientation(
                self._held_obj_id,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=self._physics_client_id)

        # Step the simulation here before adding or removing constraints
        # because detect_held_object() should use the updated state.
        if CFG.pybullet_control_mode != "reset":
            for _ in range(CFG.pybullet_sim_steps_per_action):
                # Residual physics commands act during this one action
                # (applyExternalForce is cleared by each stepSimulation,
                # so continuous actuation is re-applied per substep).
                self._apply_pending_residual_commands()
                p.stepSimulation(physicsClientId=self._physics_client_id)
        # Consumed: commands act for exactly one action and expire
        # unless the residual simulator re-queues them post-step.
        self._pending_residual_commands = []

        if self._contact_log is not None:
            self._record_step_contacts()

        # If not currently holding something, and fingers are closing, check
        # for a new grasp.
        if self._held_constraint_id is None and self._fingers_closing(action):
            self._held_obj_id = self._detect_held_object()
            if self._held_obj_id is not None:
                self._create_grasp_constraint()

        # If placing, remove the grasp constraint.
        if self._held_constraint_id is not None and \
            self._fingers_opening(action):
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None
            self._held_obj_id = None

    def _domain_specific_step(self) -> None:
        """Apply domain-specific dynamics after the base sim.

        Override in subclasses to add post-base-sim effects (water
        filling, heating, balance beam physics, etc.). Skipped when
        ``skip_residual_dynamics=True`` is passed to the constructor.
        """

    # ── Residual physics commands ───────────────────────────────

    def queue_residual_commands(self,
                                commands: Sequence[PhysicsCommand]) -> None:
        """Queue physics commands to act during the NEXT action's substeps.

        The shared executor for residual actuation (see
        :mod:`predicators.code_sim_learning.commands`), with two
        callers on the same post-step cadence: learned residual rules,
        and an env's own ``_domain_specific_step`` (the fan wind) - so
        a learned rule emitting the env's command is bit-identical to
        the env. Queued commands are executed during the physics
        substeps of the following ``step``/``simulate`` call - the
        engine, not the emitter, resolves the contacts the commanded
        motion runs into. They expire after that one action (re-queue
        to persist) and at episode reset. The caller owns
        jump-validity: queue only commands computed for the state the
        next step will run from (the combined simulators key their
        pending commands to that exact state and drop them when the
        planner backtracks elsewhere). Only meaningful on
        position/velocity control modes: the kinematic ``reset``
        control mode runs no physics substeps, so commands would have
        nothing to act on.
        """
        self._pending_residual_commands = list(commands)

    def _apply_pending_residual_commands(self) -> None:
        """Execute the queued commands against the live PyBullet world.

        Called before every physics substep, so force/torque commands
        act as continuous actuation across the whole action. Objects
        are resolved by NAME against ``self._objects`` (the set the
        current state carries), so command emitters built from another
        env instance's ``State`` still drive this env. Fails soft on
        unknown names or bodiless objects - agent-written rules may
        reference objects a probe state dropped - with a warning rather
        than a crashed rollout.
        """
        if not self._pending_residual_commands:
            return
        ids_by_name: Dict[str, int] = {}
        for obj in self._objects:
            obj_id = getattr(obj, "id", None)
            if obj_id is not None and obj_id >= 0:
                ids_by_name[obj.name] = obj_id
        for cmd in self._pending_residual_commands:
            body_id = ids_by_name.get(cmd.obj_name)
            if body_id is None:
                logging.warning(
                    "Residual command targets unknown or bodiless object "
                    "'%s'; skipping.", cmd.obj_name)
                continue
            if isinstance(cmd, ApplyForce):
                pos, _ = p.getBasePositionAndOrientation(
                    body_id, physicsClientId=self._physics_client_id)
                p.applyExternalForce(objectUniqueId=body_id,
                                     linkIndex=-1,
                                     forceObj=list(cmd.force),
                                     posObj=pos,
                                     flags=p.WORLD_FRAME,
                                     physicsClientId=self._physics_client_id)
            elif isinstance(cmd, ApplyTorque):
                p.applyExternalTorque(objectUniqueId=body_id,
                                      linkIndex=-1,
                                      torqueObj=list(cmd.torque),
                                      flags=p.WORLD_FRAME,
                                      physicsClientId=self._physics_client_id)
            elif isinstance(cmd, SetVelocity):
                kwargs: Dict[str, Any] = {}
                if cmd.linear is not None:
                    kwargs["linearVelocity"] = list(cmd.linear)
                if cmd.angular is not None:
                    kwargs["angularVelocity"] = list(cmd.angular)
                p.resetBaseVelocity(body_id,
                                    physicsClientId=self._physics_client_id,
                                    **kwargs)

    # ── Real execution ──────────────────────────────────────────

    def attach_executor(self, executor: ActionExecutor) -> None:
        """Let ``executor`` drive real hardware from this env's rollouts.

        The only way one is installed, and the default is none -- so an
        env is pure simulation unless somebody explicitly says
        otherwise. That matters because the planner builds envs of its
        own (``create_option_model``'s private simulator, the shared
        skill simulator) and those must never touch a robot.
        """
        self._executor = executor

    def get_train_tasks(self) -> List[EnvironmentTask]:
        """The train tasks, letting an executor supply them instead.

        ``BaseEnv`` caches these, which is right for a simulated env
        whose tasks are generated once. For a real one it is wrong:
        every episode faces a physically different scene, and the task
        has to be rebuilt from what is actually there. The executor is
        asked first; see ``ActionExecutor.tasks_for``.
        """
        self._maybe_replace_tasks("train")
        return super().get_train_tasks()

    def get_test_tasks(self) -> List[EnvironmentTask]:
        """The test tasks, letting an executor supply them instead."""
        self._maybe_replace_tasks("test")
        return super().get_test_tasks()

    def _maybe_replace_tasks(self, train_or_test: str) -> None:
        """Overwrite the cached tasks for a split if the executor has new
        ones."""
        if self._executor is None:
            return
        fresh = self._executor.tasks_for(train_or_test)
        if fresh is None:
            return
        cached = (self._train_tasks
                  if train_or_test == "train" else self._test_tasks)
        if cached and len(fresh) < len(cached):
            # Keep the list length: task *indices* have already been handed out
            # against the old length, so shrinking it would turn a live index
            # into an IndexError. A real environment perceives one world, so
            # every index legitimately names the same freshly-perceived scene.
            fresh = [fresh[0]] * len(cached)
        if train_or_test == "train":
            self._train_tasks = fresh
        else:
            self._test_tasks = fresh

    # ── State Write (State → PyBullet) ──────────────────────────

    def sync_to_state(self, state: State) -> None:
        """Overwrite this env's PyBullet world from ``state``, leaving no stale
        momentum behind.

        ``_set_state`` alone is not enough to adopt a state that came
        from *outside* this simulation (a perceived pose, say). It writes
        body poses through ``update_object``, which calls
        ``resetBasePositionAndOrientation`` and does **not** touch
        velocities, so every body keeps whatever momentum the previous
        rollout gave it and starts drifting on the next
        ``stepSimulation``. This zeroes those velocities explicitly, the
        same way the domino component already does after it places
        blocks.

        Robot joints need no equivalent: ``_set_state`` routes them
        through ``SingleArmPyBulletRobot.set_joints``, which already
        resets each joint with ``targetVelocity=0``.
        """
        self._set_state(state)
        self._zero_object_velocities()

    def _zero_object_velocities(self) -> None:
        """Zero the linear and angular velocity of every body in the state."""
        for obj in self._objects:
            obj_id = getattr(obj, "id", None)
            if obj_id is None:  # virtual object with no PyBullet body
                continue
            p.resetBaseVelocity(obj_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                                physicsClientId=self._physics_client_id)

    def gripper_joint_layout(self) -> GripperJointLayout:
        """Where the finger joints sit in an action array, and what open /
        closed finger values look like."""
        return gripper_joint_layout_from_robot(self._pybullet_robot)

    def _set_state(self, state: State) -> None:
        """State -> PyBullet: write the requested State into the simulator.

        Per-component diff: each piece of the State (robot pose, each
        object pose, held-object identity) is compared against the live
        PyBullet world and only re-written when it actually differs.
        This lets sequential rollouts (option model, learned process
        simulators) advance without snapping the arm or rebuilding the
        grasp constraint when only a subset of features changed — which
        is what eliminates the visible robot jitter during combined
        base+learned simulator calls. It also lets a learned rule move
        an *unheld* object without disturbing the arm or any other body.

        Call sites:
        - reset() / _add_pybullet_state_to_tasks(): initialization
        - simulate(): option-model / bilevel-planning rollouts
        - external callers (skill factories, agent tools, tests)
        """
        # Cohort change or the very first call forces a full reset:
        # per-component compares assume the same set of bodies.
        full_reset = (self._current_observation is None
                      or set(self._objects) != set(state.data))

        # Keep _current_observation in sync so step() can read it
        # (e.g. for finger-delta computation).
        self._current_observation = state
        self._objects = list(state.data)

        # Reset per-call; the reconstruction check below repopulates it with
        # any features this reset could not round-trip.
        self._last_unreconstructible_features = []

        # Mobile base: restore the base pose first, since every arm/object
        # world pose is expressed relative to it. _robot_matches_state also
        # checks the base, so a base move forces the joints + grasp constraint
        # to be rebuilt in the restored base frame below.
        self._restore_base_pose_from_state(state)

        wrote_anything = False

        # 1) Robot pose diff. Skipping this branch when the live joints
        # already match the requested pose is what eliminates arm
        # jitter: resetJointState would otherwise hard-snap the arm
        # on every simulate() call in a sequential rollout.
        robot_changed = full_reset or not self._robot_matches_state(state)

        # 2) Object pose diff. Identify which non-virtual object bodies
        # have moved relative to PyBullet.
        objects_to_reset: List[Object] = []
        for obj in self._objects:
            if obj.type.name == "robot" or \
                obj.type.name in self._VIRTUAL_OBJECT_TYPES or \
                obj.id is None:
                continue
            if full_reset or not self._object_pose_matches_state(obj, state):
                objects_to_reset.append(obj)

        # 3) Held-object identity diff. The grasp constraint must be
        # torn down and rebuilt whenever:
        #   - the held identity changes (including held → unheld and
        #     unheld → held),
        #   - the held object's recorded pose changes (the offset to
        #     the gripper moves), or
        #   - the gripper itself moves (resetJointState bypasses the
        #     constraint, so a kept constraint would leave the held
        #     body behind).
        new_held_id = self._held_obj_id_in_state(state)
        held_obj_moved = (self._held_obj_id is not None
                          and any(o.id == self._held_obj_id
                                  for o in objects_to_reset))
        rebuild_constraint = (full_reset or new_held_id != self._held_obj_id
                              or (self._held_obj_id is not None and
                                  (robot_changed or held_obj_moved)))

        # Tear down before robot/object resets so the held body is free
        # while we move things around.
        if rebuild_constraint:
            if self._held_constraint_id is not None:
                p.removeConstraint(self._held_constraint_id,
                                   physicsClientId=self._physics_client_id)
                wrote_anything = True
            self._held_constraint_id = None
            self._held_obj_to_base_link = None
            self._held_obj_id = None

        if robot_changed:
            # Prefer exact joint positions when the State carries them in
            # simulator_state — IK from (x, y, z, tilt, wrist) drops
            # wrist roll, which corrupts the held-object offset that
            # _create_grasp_constraint records below.
            joint_positions = self._extract_robot_joint_positions(state)
            # When simulator_state is a rich dict (produced exclusively by
            # _get_state), the joint hint is authoritative — skip
            # reset_state's roundtrip-vs-EE-pose guardrail, which can
            # spuriously fail on Euler->Quat float noise at the 1e-2
            # tolerance and force a lossy IK fallback. Raw-sequence and
            # missing simulator_state still go through the guardrail.
            sim_state = getattr(state, "simulator_state", None)
            trust_joints = (isinstance(sim_state, dict)
                            and "joint_positions" in sim_state)
            self._pybullet_robot.reset_state(self._extract_robot_state(state),
                                             joint_positions=joint_positions,
                                             trust_joints=trust_joints)
            # reset_state snaps the base back to the robot's fixed home pose;
            # for a mobile base, re-apply the requested base pose so the joints
            # (recorded for that base) place the arm in the right world frame
            # and the grasp constraint below is recorded in the correct frame.
            self._restore_base_pose_from_state(state)
            wrote_anything = True

        for obj in objects_to_reset:
            self._reset_single_object(obj, state)
            wrote_anything = True

        # Recreate the constraint after objects are repositioned so the
        # recorded base_link → object offset matches the new pose.
        if rebuild_constraint and new_held_id is not None:
            self._held_obj_id = new_held_id
            self._create_grasp_constraint()
            wrote_anything = True

        # 4) Subclass-specific state always runs (idempotent and cheap).
        self._set_domain_specific_state(state)

        # 5) Reconstruction check — only when we actually wrote something
        # kinematic. React by mismatch magnitude (see the threshold
        # ClassVars above): a large mismatch can't be benign IK noise, so
        # raise; a small one just warns since the IK reset path is lossy.
        if wrote_anything:
            reconstructed = self._get_state()
            warn_diff = self._reconstruction_diff(
                state, reconstructed, atol=self._reconstruction_warn_atol)
            if warn_diff:
                # raise_atol > warn_atol, so this is a subset of warn_diff;
                # only non-empty for mismatches too big to be IK noise.
                raise_diff = self._reconstruction_diff(
                    state, reconstructed, atol=self._reconstruction_raise_atol)
                if raise_diff:
                    raise ValueError(
                        f"Could not reconstruct state. Mismatched "
                        f"features:\n{raise_diff}")
                logging.warning(
                    "Could not reconstruct state exactly in reset. "
                    "Mismatched features:\n%s", warn_diff)
                # Structured view of the same mismatch, for combined
                # simulators to repair the carried value (see
                # _last_unreconstructible_features).
                self._last_unreconstructible_features = \
                    self._reconstruction_mismatch_features(
                        state, reconstructed,
                        atol=self._reconstruction_warn_atol)

    @classmethod
    def _reconstruction_mismatch_features(
            cls,
            requested: State,
            reconstructed: State,
            atol: float = 1e-3) -> List[Tuple[Object, str]]:
        """Structured counterpart of ``_reconstruction_diff``.

        Returns the ``(object, feature)`` pairs whose reconstructed
        value differs from the requested value by more than ``atol``.
        Combined simulators intersect this with their declared process
        features to repair exactly the learned-owned observables that a
        reset cannot round-trip (e.g. ``bubbling_level`` derived from a
        hidden ``heat_level``), leaving base-reconstructible features
        (kinematic ``x, y`` a robot can move) untouched. Angle features
        are compared modulo 2π; the orientation-triple geodesic handling
        in ``_reconstruction_diff`` is unnecessary here because
        orientation features are kinematic — never residual features —
        so they are filtered out by the caller's intersection
        regardless.
        """
        out: List[Tuple[Object, str]] = []
        for obj in set(requested.data) & set(reconstructed.data):
            req_vals = requested.data[obj]
            rec_vals = reconstructed.data[obj]
            if len(req_vals) != len(rec_vals):
                continue
            for i, feat in enumerate(obj.type.feature_names):
                req_v = float(req_vals[i])
                rec_v = float(rec_vals[i])
                if feat in cls._ANGLE_FEATURES:
                    delta = (rec_v - req_v + np.pi) % (2 * np.pi) - np.pi
                else:
                    delta = rec_v - req_v
                if abs(delta) > atol:
                    out.append((obj, feat))
        return out

    @classmethod
    def _reconstruction_diff(cls,
                             requested: State,
                             reconstructed: State,
                             atol: float = 1e-3,
                             max_lines: int = 10) -> str:
        """Format per-feature mismatches between two States for debugging.

        Returns a human-readable summary of which (object, feature)
        pairs differ by more than ``atol``, sorted by largest absolute
        delta. Truncates to ``max_lines`` rows so the warning stays
        scannable. Returns an empty string when no feature exceeds
        ``atol`` and the object set matches.

        Single angle features (see ``_ANGLE_FEATURES``) are compared modulo
        2π so a wrist value of 4.68 matches a reconstructed -1.60 (same
        physical orientation, different euler representation). Features that
        jointly form a full orientation (see ``_ORIENTATION_EULER_TRIPLES``)
        are instead compared as a rotation — the geodesic angle between the
        two — which is gimbal-lock safe: at tilt=±π/2 the per-axis split of
        roll/wrist is degenerate, so an axis-by-axis compare would report up
        to π of spurious error on the same physical orientation.
        """
        req_objs = set(requested.data)
        rec_objs = set(reconstructed.data)
        rows = []
        only_in_req = req_objs - rec_objs
        only_in_rec = rec_objs - req_objs
        if only_in_req:
            rows.append(f"  objects only in requested: "
                        f"{sorted(o.name for o in only_in_req)}")
        if only_in_rec:
            rows.append(f"  objects only in reconstructed: "
                        f"{sorted(o.name for o in only_in_rec)}")
        # (sort_key, formatted_row); orientation-triple and per-feature diffs
        # share one sorted, truncated list so the worst mismatch leads.
        feature_diffs: List[Tuple[float, str]] = []
        for obj in req_objs & rec_objs:
            req_vals = requested.data[obj]
            rec_vals = reconstructed.data[obj]
            if len(req_vals) != len(rec_vals):
                rows.append(f"  {obj.name}: feature-count mismatch "
                            f"requested={len(req_vals)} "
                            f"reconstructed={len(rec_vals)}")
                continue
            features = obj.type.feature_names
            # Compare any full Euler orientation triple as one rotation
            # (gimbal-lock safe); its constituent angles are then excluded
            # from the axis-by-axis pass below.
            handled: Set[str] = set()
            for triple in cls._ORIENTATION_EULER_TRIPLES:
                if not set(triple).issubset(features):
                    continue
                idx = [features.index(f) for f in triple]
                req_eul = [float(req_vals[j]) for j in idx]
                rec_eul = [float(rec_vals[j]) for j in idx]
                angle = cls._euler_orientation_angle(req_eul, rec_eul)
                handled.update(triple)
                if angle > atol:
                    axes = ", ".join(
                        f"{f}={r:.6f}->{c:.6f}"
                        for f, r, c in zip(triple, req_eul, rec_eul))
                    feature_diffs.append(
                        (angle, f"  {obj.name}.<orientation>: "
                         f"Δangle={angle:.6f} rad ({axes})"))
            for i, feat in enumerate(features):
                if feat in handled:
                    continue
                req_v = float(req_vals[i])
                rec_v = float(rec_vals[i])
                if feat in cls._ANGLE_FEATURES:
                    # Wrap the difference into [-π, π].
                    delta = (rec_v - req_v + np.pi) % (2 * np.pi) - np.pi
                else:
                    delta = rec_v - req_v
                if abs(delta) > atol:
                    feature_diffs.append((abs(delta), f"  {obj.name}.{feat}: "
                                          f"requested={req_v:.6f} "
                                          f"reconstructed={rec_v:.6f} "
                                          f"(Δ={rec_v - req_v:+.6f})"))
        feature_diffs.sort(key=lambda d: d[0], reverse=True)
        for _key, row in feature_diffs[:max_lines]:
            rows.append(row)
        if len(feature_diffs) > max_lines:
            rows.append(f"  ... and {len(feature_diffs) - max_lines} "
                        f"more features over the {atol:g} tolerance")
        return "\n".join(rows)

    @staticmethod
    def _euler_orientation_angle(euler_a: Sequence[float],
                                 euler_b: Sequence[float]) -> float:
        """Geodesic angle in radians (``[0, π]``) between the two rotations
        given as extrinsic-XYZ euler triples.

        Representation-invariant: two euler triples encoding the same
        rotation — including different gimbal-lock branches, e.g.
        (roll=2.42, tilt=π/2, wrist=-0.71) vs (roll=0, tilt=π/2,
        wrist=-3.13) — return ~0. Computed as the angle between the unit
        quaternions, taking the smaller of q and -q (double cover).
        """
        q_a = np.array(p.getQuaternionFromEuler(list(euler_a)))
        q_b = np.array(p.getQuaternionFromEuler(list(euler_b)))
        dot = float(np.clip(abs(float(np.dot(q_a, q_b))), 0.0, 1.0))
        return float(2.0 * np.arccos(dot))

    def _robot_matches_state(self, state: State, atol: float = 1e-3) -> bool:
        """True if PyBullet's live robot pose already equals state's.

        Compares at the joint level. The EE-quaternion path that
        ``_extract_robot_state`` builds always uses ``roll=0``, so any
        non-zero wrist roll in the live PyBullet pose would spuriously
        fail an EE-pose comparison and trigger a full robot reset on
        every simulate() call (visible jitter).

        ``atol`` matches ``State.allclose``'s feature tolerance: a looser
        check would let the fast-path skip a reset even when the live EE
        pose differs from the requested state by more than allclose
        accepts (e.g. when a caller hands us
        ``initial_joint_positions`` as a hint and the live joints are
        only 1e-2 close).

        Returns False when ``state`` has no joint_positions — the only
        live caller in that situation is
        ``_add_pybullet_state_to_tasks``, where forcing a reset is
        exactly the desired behavior.
        """
        jp = self._extract_robot_joint_positions(state)
        if jp is None:
            return False
        try:
            cur_jp = self._pybullet_robot.get_joints()
        except (KeyError, ValueError):
            return False
        if not bool(np.allclose(jp, cur_jp, atol=atol)):
            return False
        # Mobile base: a base move (with identical joints) still relocates the
        # whole arm, so it must count as a robot change.
        want_base = self._base_pose_from_state(state)
        if want_base is not None:
            cur_base = self._robot_base_pose_tuple()
            if cur_base is not None and not (
                    np.allclose(want_base[0], cur_base[0], atol=atol)
                    and np.allclose(want_base[1], cur_base[1], atol=atol)):
                return False
        return True

    @staticmethod
    def _base_pose_from_state(
        state: State
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float,
                                                          float]]]:
        """Pull a mobile base pose out of a State's simulator_state, if any."""
        sim_state = getattr(state, "simulator_state", None)
        if isinstance(sim_state, dict):
            return sim_state.get("base_pose", None)
        return None

    def _restore_base_pose_from_state(self, state: State) -> None:
        """Set the mobile base pose from the State's simulator_state, if it
        carries one (no-op for fixed-base robots / states without it)."""
        base_pose = self._base_pose_from_state(state)
        if base_pose is None:
            return
        robot = self._pybullet_robot
        if not hasattr(robot, "set_base_pose"):
            return
        pos, orn = base_pose
        robot.set_base_pose(  # type: ignore[attr-defined]
            Pose((pos[0], pos[1], pos[2]), (orn[0], orn[1], orn[2], orn[3])))

    def _object_pose_matches_state(self,
                                   obj: Object,
                                   state: State,
                                   atol: float = 1e-3) -> bool:
        """True if PyBullet's live pose for ``obj`` equals state[obj].

        ``atol`` matches ``_reconstruction_diff``'s tolerance so an
        object that the diff helper would complain about is also one the
        matches-check rejects — without this alignment, an object whose
        pose drifts within 1e-3..1e-2 sits stale in the planning sim
        (skipped by this check) while the diff still flags it, and the
        planning sim's plans get computed against the stale pose.
        """
        if obj.id is None:
            return True
        try:
            features = obj.type.feature_names
            (px, py, pz), orn = p.getBasePositionAndOrientation(
                obj.id, physicsClientId=self._physics_client_id)
            if "x" in features and \
                    not np.isclose(state.get(obj, "x"), px, atol=atol):
                return False
            if "y" in features and \
                    not np.isclose(state.get(obj, "y"), py, atol=atol):
                return False
            if "z" in features and \
                    not np.isclose(state.get(obj, "z"), pz, atol=atol):
                return False
            if {"rot", "yaw", "roll", "pitch"} & set(features):
                roll, pitch, yaw = p.getEulerFromQuaternion(orn)
                if "rot" in features and not np.isclose(
                        state.get(obj, "rot"), yaw, atol=atol):
                    return False
                if "yaw" in features and not np.isclose(
                        state.get(obj, "yaw"), yaw, atol=atol):
                    return False
                if "roll" in features and not np.isclose(
                        state.get(obj, "roll"), roll, atol=atol):
                    return False
                if "pitch" in features and not np.isclose(
                        state.get(obj, "pitch"), pitch, atol=atol):
                    return False
            return True
        except (KeyError, ValueError):
            return False

    def _held_obj_id_in_state(self, state: State) -> Optional[int]:
        """Which PyBullet body id is marked is_held > 0.5 in ``state``.

        Returns None if no object is held in ``state``. Mirrors the per-
        object logic in _reset_single_object before constraint
        management was hoisted out into _set_state.
        """
        for obj in state.data:
            if obj.id is None:
                continue
            if "is_held" not in obj.type.feature_names:
                continue
            try:
                if state.get(obj, "is_held") > 0.5:
                    return obj.id
            except (KeyError, ValueError):
                continue
        return None

    def _reset_single_object(self, obj: Object, state: State) -> None:
        """Teleport a single physical object to match the given State.

        Pose only — grasp-constraint management is centralized in
        _set_state so teardown/rebuild stays in one place.

        Called by _set_state() for every non-robot, non-virtual object
        whose pose differs from PyBullet (or for all such objects on a
        full reset).
        """
        # Skip objects without pybullet IDs (handled by subclass).
        if obj.id is None:
            return

        # 1) Position/orientation if those features exist
        features = obj.type.feature_names
        cur_x, cur_y, cur_z = p.getBasePositionAndOrientation(
            obj.id, physicsClientId=self._physics_client_id)[0]
        px = state.get(obj, "x") if "x" in obj.type.feature_names else cur_x
        py = state.get(obj, "y") if "y" in obj.type.feature_names else cur_y
        pz = state.get(obj, "z") if "z" in obj.type.feature_names else cur_z

        if "rot" in features:
            angle = state.get(obj, "rot")
            # Convert from 2D angle to a 3D quaternion (assuming rotation around
            # z)
            orn = p.getQuaternionFromEuler([0.0, 0.0, angle])
        elif {"yaw", "roll", "pitch"} & set(features):
            # Rebuild the full orientation from whichever Euler angles the type
            # carries (PyBullet's convention is [roll, pitch, yaw]). Dropping
            # roll/pitch here would make toppled objects — e.g. a fallen domino
            # with roll≈π — unreconstructible: _get_state reads the angle back,
            # the mismatch exceeds _reconstruction_raise_atol, and _set_state
            # raises instead of round-tripping. Missing angles default to 0.
            roll = state.get(obj, "roll") if "roll" in features else 0.0
            pitch = state.get(obj, "pitch") if "pitch" in features else 0.0
            yaw = state.get(obj, "yaw") if "yaw" in features else 0.0
            orn = p.getQuaternionFromEuler([roll, pitch, yaw])
        else:
            orn = self._default_orn  # e.g. (0,0,0,1)

        # 2) Update the object's position/orientation in PyBullet
        update_object(obj.id, (px, py, pz),
                      orn,
                      physics_client_id=self._physics_client_id)

    @abc.abstractmethod
    def _set_domain_specific_state(self, state: State) -> None:
        """Set simulator state for features that the base class doesn't handle.

        — e.g. switch on/off, liquid levels, button colors, balance beam
        positions.

        Called at the end of _set_state(), after the base class has
        already set robot joints, object poses, and grasp constraints.
        Subclasses must override.
        """
        raise NotImplementedError("Override me!")

    def _extract_robot_state(self, state: State) -> Array:
        """State -> robot array: extract robot features for PyBullet.

        Converts the robot's features in a State into the array format
        expected by self._pybullet_robot.reset_state()
        (same format as self._pybullet_robot.get_state()).

        Called by _set_state() to position the robot.
        """

        # EE Position
        def get_pos_feature(
                state: State,
                feature_name: str) -> float:  # type: ignore[no-untyped-def]
            if feature_name in self._robot.type.feature_names:
                return state.get(self._robot, feature_name)
            if f"pose_{feature_name}" in self._robot.type.feature_names:
                return state.get(self._robot, f"pose_{feature_name}")
            raise ValueError(f"Cannot find robot pos '{feature_name}'")

        rx = get_pos_feature(state, "x")
        ry = get_pos_feature(state, "y")
        rz = get_pos_feature(state, "z")

        # EE Orientation
        default_roll, default_tilt, default_wrist = p.getEulerFromQuaternion(
            self.get_robot_ee_home_orn())
        if "roll" in self._robot.type.feature_names:
            roll = state.get(self._robot, "roll")
        else:
            roll = default_roll
        if "tilt" in self._robot.type.feature_names:
            tilt = state.get(self._robot, "tilt")
        else:
            tilt = default_tilt
        if "wrist" in self._robot.type.feature_names:
            wrist = state.get(self._robot, "wrist")
        else:
            wrist = default_wrist
        qx, qy, qz, qw = p.getQuaternionFromEuler([roll, tilt, wrist])

        # Fingers
        f = state.get(self._robot, "fingers")
        f = self._fingers_state_to_joint(self._pybullet_robot, f)

        return np.array([rx, ry, rz, qx, qy, qz, qw, f], dtype=np.float32)

    def _extract_robot_joint_positions(
            self, state: State) -> Optional[JointPositions]:
        """Pull arm joint positions out of a State's simulator_state.

        Returns None when the State doesn't carry them (plain State, or
        a PyBulletState whose simulator_state has a different shape than
        this robot's arm). Callers fall back to IK in that case.
        """
        sim_state = getattr(state, "simulator_state", None)
        jp: Any
        if isinstance(sim_state, dict):
            jp = sim_state.get("joint_positions")
        elif sim_state is None:
            return None
        else:
            # PyBulletState also accepts simulator_state passed as a raw
            # joint-positions sequence (see PyBulletState.joint_positions
            # and tests/envs/test_pybullet_blocks.py:69-70).
            jp = sim_state
        if jp is None:
            return None
        try:
            jp_list = list(jp)
        except TypeError:
            return None
        if len(jp_list) != len(self._pybullet_robot.arm_joints):
            return None
        return cast(JointPositions, jp_list)

    @classmethod
    def _fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                                finger_state: float) -> float:
        """Map finger value in a State (e.g. open_fingers=0.04) to the
        corresponding PyBullet joint position.

        Linearly interpolates between the State-domain endpoints
        (cls.open_fingers / cls.closed_fingers) and the PyBullet-domain
        endpoints (pybullet_robot.open_fingers / .closed_fingers) so
        mid-transition finger values round-trip through _get_state /
        _set_state without being snapped to an endpoint.

        Called by _extract_robot_state() when writing State -> PyBullet.
        """
        s_open, s_closed = cls.open_fingers, cls.closed_fingers
        r_open, r_closed = (pybullet_robot.open_fingers,
                            pybullet_robot.closed_fingers)
        if s_open == s_closed:
            return r_open
        t = (finger_state - s_closed) / (s_open - s_closed)
        return r_closed + t * (r_open - r_closed)

    # ── State Read (PyBullet → State) ───────────────────────────

    # Features handled by _get_object_state_dict via PyBullet queries.
    _PYBULLET_FEATURES: ClassVar[frozenset] = frozenset({
        "x", "y", "z", "rot", "yaw", "roll", "pitch", "is_held", "r", "g", "b"
    })

    def _get_state(self, _render_obs: bool = False) -> State:
        """PyBullet -> State: read the simulator into a PyBulletState.

        Queries PyBullet for the current scene (joint positions, body
        poses, visual data, etc.) and packs the values into the
        agent-facing State representation.

        Handles common features (robot pose, object x/y/z/rot/is_held,
        color); subclass-specific features are delegated to
        `_get_domain_specific_feature`.

        Called by get_observation() (after reset/step) and by
        _set_state() to verify reconstruction fidelity.
        """
        state_dict: Dict[Object, Dict[str, float]] = {}
        state_dict[self._robot] = self._get_robot_state_dict()
        for obj in self._objects:
            if obj.type.name == "robot":
                continue
            state_dict[obj] = self._get_object_state_dict(obj)

        state = utils.create_state_from_dict(state_dict)
        joint_positions = self._pybullet_robot.get_joints()
        sim_state_dict: Dict[str, Any] = {
            "joint_positions": joint_positions,
            "physics_client_id": self._physics_client_id,
            "robot_id": self._pybullet_robot.robot_id,
        }
        # Mobile robots: carry the base pose so it round-trips through
        # _set_state (the base is not a State feature, so without this a
        # reconstruction would silently keep the live base pose, breaking
        # option-model / refinement rollouts that move the base).
        base_pose = self._robot_base_pose_tuple()
        if base_pose is not None:
            sim_state_dict["base_pose"] = base_pose
        pyb_state = PyBulletState(state.data, simulator_state=sim_state_dict)
        return pyb_state

    def _robot_base_pose_tuple(
        self
    ) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float,
                                                          float]]]:
        """Return the mobile base pose as (position, orientation) tuples, or
        None for fixed-base robots."""
        robot = self._pybullet_robot
        if int(getattr(robot, "base_action_dim", 0)) <= 0 or \
                not hasattr(robot, "get_base_pose"):
            return None
        base_pose = robot.get_base_pose()  # type: ignore[attr-defined]
        return (tuple(base_pose.position), tuple(base_pose.orientation))

    def _get_robot_state_dict(self) -> Dict[str, float]:
        """Build a feature dict for the robot from PyBullet state.

        Called by _get_state() to populate the robot entry in the State.
        Subclasses with non-standard robot features (e.g. cover's
        normalized hand, blocks' pose_x/y/z) should override this.
        """
        rx, ry, rz, qx, qy, qz, qw, rf = self._pybullet_robot.get_state()
        # Rescale the finger JOINT into the State domain. _extract_robot_state
        # applies _fingers_state_to_joint on the way out, so reading the raw
        # joint here makes the round-trip asymmetric: the skills' fingers
        # helper re-converts an already-joint value, deflating every reading.
        rf = self._fingers_joint_to_state(self._pybullet_robot, rf)
        r_dict: Dict[str, float] = {"x": rx, "y": ry, "z": rz, "fingers": rf}
        roll, tilt, wrist = p.getEulerFromQuaternion([qx, qy, qz, qw])
        r_features = self._robot.type.feature_names
        if "roll" in r_features:
            r_dict["roll"] = roll
        if "tilt" in r_features:
            r_dict["tilt"] = tilt
        if "wrist" in r_features:
            r_dict["wrist"] = wrist
        return r_dict

    def _get_object_state_dict(self, obj: Object) -> Dict[str, float]:
        """Build a feature dict for a single non-robot object.

        Virtual objects (loc, angle, etc.) delegate all features to
        _get_domain_specific_feature.  Physical objects get
        pose/color/is_held from PyBullet; the rest are delegated.
        """
        obj_features = obj.type.feature_names
        obj_dict: Dict[str, float] = {}

        if obj.type.name in self._VIRTUAL_OBJECT_TYPES:
            for feature in obj_features:
                obj_dict[feature] = \
                    self._get_domain_specific_feature(obj, feature)
            return obj_dict

        # Physical object — query PyBullet for pose
        try:
            (px, py, pz), orn = retry_pybullet_call(
                p.getBasePositionAndOrientation,
                obj.id,
                physicsClientId=self._physics_client_id)
        except Exception as e:
            raise RuntimeError(f"Failed to get pose for object {obj.name} "
                               f"(id={obj.id})") from e
        if "x" in obj_features:
            obj_dict["x"] = px
        if "y" in obj_features:
            obj_dict["y"] = py
        if "z" in obj_features:
            obj_dict["z"] = pz

        if {"rot", "yaw", "roll", "pitch"} & set(obj_features):
            roll, pitch, yaw = p.getEulerFromQuaternion(orn)
            if "rot" in obj_features:
                obj_dict["rot"] = yaw
            if "yaw" in obj_features:
                obj_dict["yaw"] = yaw
            if "roll" in obj_features:
                obj_dict["roll"] = roll
            if "pitch" in obj_features:
                obj_dict["pitch"] = pitch

        if "is_held" in obj_features:
            obj_dict["is_held"] = 1.0 if obj.id == self._held_obj_id else 0.0

        if {"r", "g", "b"} & set(obj_features):
            visual_data = retry_pybullet_call(
                p.getVisualShapeData,
                obj.id,
                physicsClientId=self._physics_client_id)[0]
            (r, g, b, _a) = visual_data[7]
            obj_dict["r"] = r
            obj_dict["g"] = g
            obj_dict["b"] = b

        # Remaining features delegated to subclass
        for feature in obj_features:
            if feature not in self._PYBULLET_FEATURES:
                obj_dict[feature] = \
                    self._get_domain_specific_feature(
                        obj, feature)

        return obj_dict

    @abc.abstractmethod
    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        """Return a single feature value for a non-robot object.

        Called by _get_object_state_dict() for:
        - All features of virtual objects (those in _VIRTUAL_OBJECT_TYPES)
        - Non-standard features of physical objects (anything not in
          _PYBULLET_FEATURES, e.g. is_on, growth, water_height)
        """
        raise NotImplementedError("Override me!")

    @classmethod
    def _fingers_joint_to_state(cls, pybullet_robot: SingleArmPyBulletRobot,
                                finger_joint: float) -> float:
        """Inverse of _fingers_state_to_joint().

        Linear interpolation (see _fingers_state_to_joint for rationale).

        Called by _get_robot_state_dict() when reading PyBullet ->
        State.
        """
        s_open, s_closed = cls.open_fingers, cls.closed_fingers
        r_open, r_closed = (pybullet_robot.open_fingers,
                            pybullet_robot.closed_fingers)
        if r_open == r_closed:
            return s_open
        t = (finger_joint - r_closed) / (r_open - r_closed)
        return s_closed + t * (s_open - s_closed)

    # ── Grasp Detection & Constraint Management ─────────────────

    @abc.abstractmethod
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return PyBullet body IDs of objects that can be grasped.

        Called by _detect_held_object() (inside step()) to decide which
        bodies to check for finger contact.  Subclasses return only the
        IDs of graspable objects (e.g. blocks, not tables).
        """
        raise NotImplementedError("Override me!")

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        """Compute the expected inward-facing normal for each finger.

        Called by _detect_held_object() to distinguish objects between
        the fingers (valid grasp) from objects touching the outside.
        """
        _rx, _ry, _rz, qx, qy, qz, qw, _rf = self._pybullet_robot.get_state()

        # Convert the quaternion to a rotation matrix
        rotation_matrix = p.getMatrixFromQuaternion([qx, qy, qz, qw])
        rotation_matrix = np.array(rotation_matrix).reshape(3, 3)

        # Define the initial normal vectors for the fingers
        if CFG.pybullet_robot == "panda":
            # Fingers close along EE-frame y.
            normal = np.array([0., 1., 0.], dtype=np.float32)
        elif CFG.pybullet_robot in {"fetch", "mobile_fetch"}:
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        # Transform the normal vectors using the rotation matrix
        transformed_normal = rotation_matrix.dot(normal)
        transformed_normal_neg = rotation_matrix.dot(-1 * normal)

        return {
            self._pybullet_robot.left_finger_id: transformed_normal,
            self._pybullet_robot.right_finger_id: transformed_normal_neg,
        }

    def _detect_held_object(self) -> Optional[int]:
        """Return the PyBullet body ID of the grasped object, or None.

        Called by step() when fingers are closing and no object is
        currently held.  Checks contact between each finger and every
        graspable body (from _get_object_ids_for_held_check()), using
        contact-normal alignment to reject touches on the outside of the
        gripper.  If multiple objects qualify, returns the closest.
        """
        expected_finger_normals = self._get_expected_finger_normals()
        closest_held_obj = None
        closest_held_obj_dist = float("inf")
        for obj_id in self._get_object_ids_for_held_check():
            for finger_id, expected_normal in expected_finger_normals.items():
                assert abs(np.linalg.norm(expected_normal) - 1.0) < 1e-5
                # Find points on the object that are within grasp_tol distance
                # of the finger. Note that we use getClosestPoints instead of
                # getContactPoints because we still want to consider the object
                # held even if there is a tiny distance between the fingers and
                # the object.
                closest_points = p.getClosestPoints(
                    bodyA=self._pybullet_robot.robot_id,
                    bodyB=obj_id,
                    distance=self.grasp_tol_small,
                    linkIndexA=finger_id,
                    physicsClientId=self._physics_client_id)
                for point in closest_points:
                    # If the contact normal is substantially different from
                    # the expected contact normal, this is probably an object
                    # on the outside of the fingers, rather than the inside.
                    # A perfect score here is 1.0 (normals are unit vectors).
                    contact_normal = point[7]
                    score = expected_normal.dot(contact_normal)
                    # logging.debug(f"With obj {obj_id}, score: {score}")
                    assert -1.01 <= score <= 1.01

                    # Take absolute as object/gripper could be rotated 180
                    # degrees in the given axis.
                    if np.abs(score) < 0.9:
                        continue
                    # Handle the case where multiple objects pass this check
                    # by taking the closest one. This should be rare, but it
                    # can happen when two objects are stacked and the robot is
                    # unstacking the top one.
                    contact_distance = point[8]
                    if contact_distance < closest_held_obj_dist:
                        closest_held_obj = obj_id
                        closest_held_obj_dist = contact_distance
        return closest_held_obj

    def _create_grasp_constraint(self) -> None:
        """Create a fixed PyBullet constraint between the end-effector and
        _held_obj_id so the object moves with the gripper.

        Called by step() after _detect_held_object() finds a grasp, and
        by _reset_single_object() when restoring a held state.
        """
        assert self._held_obj_id is not None
        base_link_to_world = np.r_[p.invertTransform(
            *p.getLinkState(self._pybullet_robot.robot_id,
                            self._pybullet_robot.end_effector_id,
                            physicsClientId=self._physics_client_id)[:2])]
        world_to_obj = np.r_[p.getBasePositionAndOrientation(
            self._held_obj_id, physicsClientId=self._physics_client_id)]
        self._held_obj_to_base_link = p.invertTransform(*p.multiplyTransforms(
            base_link_to_world[:3], base_link_to_world[3:], world_to_obj[:3],
            world_to_obj[3:]))
        self._held_constraint_id = p.createConstraint(
            parentBodyUniqueId=self._pybullet_robot.robot_id,
            parentLinkIndex=self._pybullet_robot.end_effector_id,
            childBodyUniqueId=self._held_obj_id,
            childLinkIndex=-1,  # -1 for the base
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self._held_obj_to_base_link[0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=self._held_obj_to_base_link[1],
            physicsClientId=self._physics_client_id)

    def _fingers_closing(self, action: Action) -> bool:
        """True if this action's finger target is below current position.

        Called by step() to decide whether to check for a new grasp.
        """
        f_delta = self._action_to_finger_delta(action)
        return f_delta < -self._finger_action_tol

    def _fingers_opening(self, action: Action) -> bool:
        """True if this action's finger target is above current position.

        Called by step() to decide whether to release a held object.
        """
        f_delta = self._action_to_finger_delta(action)
        return f_delta > self._finger_action_tol

    def _get_finger_position(self, state: State) -> float:
        """Return the current left-finger joint position from state.

        Called by _action_to_finger_delta() to compute the delta between
        current and target finger positions.
        """
        state = cast(utils.PyBulletState, state)
        finger_joint_idx = self._pybullet_robot.left_finger_joint_idx
        return state.joint_positions[finger_joint_idx]

    def _action_to_finger_delta(self, action: Action) -> float:
        """Compute (target - current) finger joint position.

        Called by _fingers_closing() and _fingers_opening().
        """
        assert isinstance(self._current_observation, State)
        finger_position = self._get_finger_position(self._current_observation)
        joint_positions, _ = self._split_action(action)
        target = joint_positions[self._pybullet_robot.left_finger_joint_idx]
        return target - finger_position

    # ── Action Helpers ──────────────────────────────────────────

    def _split_action(self, action: Action) -> Tuple[np.ndarray, np.ndarray]:
        """Split an action into (arm_joint_targets, base_delta).

        Called by step() and _action_to_finger_delta().  For robots
        without a mobile base, base_delta is an empty array.
        """
        action_arr = action.arr
        base_dim = int(getattr(self._pybullet_robot, "base_action_dim", 0))
        if base_dim > 0:
            expected = len(self._pybullet_robot.arm_joints) + base_dim
            if action_arr.shape[0] == expected:
                return action_arr[:-base_dim], action_arr[-base_dim:]
            if action_arr.shape[0] == len(self._pybullet_robot.arm_joints):
                zeros = np.zeros(base_dim, dtype=action_arr.dtype)
                return action_arr, zeros
            raise ValueError(
                f"Unexpected action dim {action_arr.shape[0]}, expected "
                f"{len(self._pybullet_robot.arm_joints)} or {expected}.")
        return action_arr, np.zeros(0, dtype=action_arr.dtype)

    def _apply_base_delta(self, base_delta: np.ndarray) -> None:
        """Apply a delta (dx, dy, dtheta) to the robot base.

        Called by step() for mobile robots (e.g. mobile_fetch).
        """
        robot = self._pybullet_robot
        assert hasattr(robot, 'get_base_pose'), \
            "Robot does not support base pose operations"
        base_pose = robot.get_base_pose()
        current_yaw = p.getEulerFromQuaternion(base_pose.orientation)[2]
        new_yaw = current_yaw + float(base_delta[2])
        new_pose = Pose(
            (base_pose.position[0] + float(base_delta[0]),
             base_pose.position[1] + float(base_delta[1]),
             base_pose.position[2]),
            p.getQuaternionFromEuler([0.0, 0.0, new_yaw]),
        )
        robot.set_base_pose(new_pose)  # type: ignore[attr-defined]

    # ── Rendering & Observation ─────────────────────────────────

    def _get_camera_matrices(self) -> Tuple[Any, Any, int, int]:
        """Return (view_matrix, proj_matrix, width, height) for rendering.

        Called by render() and render_segmented_obj().
        """
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._camera_target,
            distance=self._camera_distance,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id)
        width = CFG.pybullet_camera_width
        height = CFG.pybullet_camera_height
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=self._camera_fov,
            aspect=float(width / height),
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._physics_client_id)
        return view_matrix, proj_matrix, width, height

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:  # pragma: no cover
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        del action, caption  # unused
        view_matrix, proj_matrix, width, height = self._get_camera_matrices()
        (_, _, px, _, _) = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            shadow=self._render_shadow,
            lightDirection=studio_visuals.light_direction(type(self)),
            lightAmbientCoeff=self._render_light_ambient,
            lightDiffuseCoeff=self._render_light_diffuse,
            lightSpecularCoeff=self._render_light_specular,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self._physics_client_id)
        # Without dtype, np.array on the plain-list px (pybullet built
        # without numpy support) silently produces int64 - 8x the memory
        # of uint8. VideoMonitor retains one frame per step for a whole
        # episode, so the dtype and the .copy() (dropping the alpha
        # channel's backing buffer) bound episode video memory.
        rgb_array = np.array(px, dtype=np.uint8).reshape((height, width, 4))
        return [rgb_array[:, :, :3].copy()]

    def render_segmented_obj(
        self,
        action: Optional[Action] = None,
        caption: Optional[str] = None,
    ) -> Tuple[Image.Image, Dict[Object, Mask]]:
        """Render the scene and return per-object segmentation masks.

        Called by get_observation(render=True) to attach RGB images and
        masks to the observation (used for VLM predicate grounding).
        """
        del action, caption  # unused
        view_matrix, proj_matrix, width, height = self._get_camera_matrices()
        (_, _, rgbImg, _,
         segImg) = p.getCameraImage(width=width,
                                    height=height,
                                    viewMatrix=view_matrix,
                                    projectionMatrix=proj_matrix,
                                    renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                    physicsClientId=self._physics_client_id)
        original_image: np.ndarray = np.array(rgbImg, dtype=np.uint8).reshape(
            (height, width, 4))
        seg_image = np.array(segImg).reshape((height, width))
        state_img = Image.fromarray(  # type: ignore[no-untyped-call]
            original_image[:, :, :3])
        mask_dict: Dict[Object, Mask] = {}
        for obj in self._objects:
            mask_dict[obj] = (seg_image == obj.id)
        return state_img, mask_dict

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("A PyBullet environment cannot render "
                                  "arbitrary states.")

    def get_observation(self, render: bool = False) -> Observation:
        """Get the current observation of this environment.

        Reads the current state from pybullet, updates
        _current_observation (the backing field), and returns a copy
        optionally with rendered images.
        """
        state = self._get_state()
        assert isinstance(state, PyBulletState)
        self._current_observation = state
        obs = state.copy()

        if render:
            obs.add_images_and_masks(*self.render_segmented_obj())

        return obs

    def make_fresh_test_instance(self) -> Optional[BaseEnv]:
        """Fresh same-class instance sharing this env's generated tasks.

        A long-lived PyBullet world accumulates history that state-level
        resets do not clear (residual velocities the reconstruction diff
        skips, auxiliary joints no reset touches, contact-solver state
        that survives ``restoreState``), so a test episode's physics
        depends on everything executed before it. A fresh instance is
        the only proven-deterministic reset (see
        ``code_sim_learning.rollout_env.rollout_states``). Not available
        with the GUI (PyBullet allows one GUI client) or when episodes
        execute on real hardware (``CFG.real_robot_execute``: a fresh
        instance would re-open the robot bridge mid-run).
        """
        if self.using_gui or CFG.real_robot_execute:
            return None
        fresh = type(self)(use_gui=False)
        # Share the generated task lists: identical tasks, no re-
        # generation. Tasks are immutable data, so sharing is safe.
        # pylint: disable=protected-access
        fresh._train_tasks = self._train_tasks
        fresh._test_tasks = self._test_tasks
        return fresh

    def dispose(self) -> None:
        """Disconnect this instance's PyBullet client."""
        p.disconnect(self._physics_client_id)

    # ── Task Utilities ──────────────────────────────────────────

    def _add_pybullet_state_to_tasks(
            self, tasks: List[EnvironmentTask]) -> List[EnvironmentTask]:
        """Convert plain-State tasks into PyBulletState tasks.

        Called by _generate_train/test_tasks() in subclasses.  Sets up
        the simulator for each task's init state so that joint positions
        and (optionally) rendered images are captured into the task.
        """
        pybullet_tasks = []
        for task in tasks:
            # Reset the robot.
            init = task.init
            self._set_state(init)
            # Cast _current_observation from type State to PybulletState
            joint_positions = self._pybullet_robot.get_joints()
            self._current_observation = utils.PyBulletState(
                init.data.copy(), simulator_state=joint_positions)
            pybullet_init = self.get_observation(render=CFG.render_init_state)
            pybullet_init.option_history = []
            pybullet_task = EnvironmentTask(
                pybullet_init,
                task.goal,
                goal_nl=task.goal_nl,
                evaluator=task.evaluator,
                offline_task_metrics=task.offline_task_metrics,
                early_stop_min_reward=task.early_stop_min_reward)
            pybullet_tasks.append(pybullet_task)
        return pybullet_tasks
