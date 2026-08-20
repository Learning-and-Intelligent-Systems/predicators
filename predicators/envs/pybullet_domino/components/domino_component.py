"""Domino component for the domino environment.

This component handles:
- Domino blocks (start, intermediate, target, glued)
- Target objects (hinged targets)
- Pivot objects (for 180-degree turns)
- Related predicates (Toppled, Upright, Tilting, etc.)
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, \
    Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.components.base_component import \
    DominoEnvComponent
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, update_object
from predicators.settings import CFG
from predicators.structs import Object, Predicate, State, Type

if TYPE_CHECKING:
    from predicators.envs.pybullet_domino.env import PyBulletDominoComposedEnv


@dataclass
class PlacementResult:
    """Result of placing a domino, target, or pivot in the sequence."""
    success: bool
    x: float
    y: float
    rotation: float
    domino_count: int
    pivot_count: int = 0
    target_count: int = 0
    just_turned_90: bool = False
    just_placed_target: bool = False
    # Yaw to place the *next* block at. Tracks the smooth 45-deg-per-turn
    # increment, which after a turn differs from ``rotation`` (the travel
    # direction used to lay out positions) by 180 deg — same physical box,
    # but the increment representation keeps a straight run reading as one
    # constant yaw instead of flipping. ``None`` means "same as rotation"
    # (no turn has happened yet).
    block_yaw: Optional[float] = None


class DominoComponent(DominoEnvComponent):
    """Component for domino blocks, targets, and pivots.

    Manages the core domino mechanics including:
    - Domino blocks with different colors for roles
      (start, target, intermediate, glued)
    - Target objects that can be toppled
    - Pivot objects for 180-degree direction changes

    Note: domino_width, domino_depth, domino_height, domino_mass, and
    domino_friction are defined in PyBulletDominoComposedEnv.
    """

    # =========================================================================
    # DOMINO CONFIGURATION
    # =========================================================================

    # Domino shape properties - defined in PyBulletDominoComposedEnv
    # domino_width, domino_depth, domino_height, domino_mass, domino_friction

    # Domino thresholds
    domino_roll_threshold: ClassVar[float] = np.deg2rad(5)
    # A free-standing domino tips over past atan(depth/height) ~= 5.7 deg:
    # beyond that its center of mass is past the pivot edge and gravity
    # torque topples it, so an unheld lean past ~10 deg is either mid-fall
    # (committed) or propped on another body - both mean the domino was
    # genuinely knocked over. This counts propped "leaners" (e.g. a target
    # coming to rest at ~20 deg against a still-standing neighbor) that a
    # stricter criterion would miss. Recorded runs show unheld rolls are
    # bimodal (< 3 deg placement jitter or > 79 deg full topples), so the
    # 10 deg line sits in a wide empty band.
    fallen_threshold: ClassVar[float] = np.deg2rad(10)

    # Domino colors
    start_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (0.56, 0.93, 0.56, 1.)
    target_domino_color: ClassVar[Tuple[float, float, float,
                                        float]] = (0.85, 0.7, 0.85, 1.0)
    domino_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.6, 0.8, 1.0, 1.0)
    glued_domino_color: ClassVar[Tuple[float, float, float,
                                       float]] = (1.0, 0.0, 0.0, 1.0)
    glued_percentage: ClassVar[float] = 0.5
    # Heavy (immovable-obstacle) blocks: domino-shaped, gray. Their TRUE
    # mass makes them untopple-able/unmovable; planning sims can believe a
    # different (normal) mass via the ``block_mass`` physical-param
    # override, which is what the heavy-block tasks exploit.
    heavy_block_color: ClassVar[Tuple[float, float, float,
                                      float]] = (0.35, 0.35, 0.35, 1.0)
    heavy_block_true_mass: ClassVar[float] = 1000.0

    # Target and pivot dimensions
    target_height: ClassVar[float] = 0.2
    pivot_width: ClassVar[float] = 0.2

    # Grid configuration - references domino_width from
    # PyBulletDominoComposedEnv
    @staticmethod
    def _get_env_class() -> TypingType["PyBulletDominoComposedEnv"]:
        """Get PyBulletDominoComposedEnv class to access shared config."""
        from predicators.envs.pybullet_domino.env import \
            PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
        return PyBulletDominoComposedEnv

    @property
    def domino_width(self) -> float:
        """Domino width."""
        if self._dim_override["width"] is not None:
            return self._dim_override["width"]
        return self._get_env_class().domino_width

    @property
    def domino_depth(self) -> float:
        """Domino depth."""
        if self._dim_override["depth"] is not None:
            return self._dim_override["depth"]
        return self._get_env_class().domino_depth

    @property
    def domino_height(self) -> float:
        """Domino height."""
        if self._dim_override["height"] is not None:
            return self._dim_override["height"]
        return self._get_env_class().domino_height

    @property
    def domino_mass(self) -> float:
        """Domino mass."""
        return self._get_env_class().domino_mass

    @property
    def domino_friction(self) -> float:
        """Domino friction."""
        return self._get_env_class().domino_friction

    @property
    def pos_gap(self) -> float:
        """Pos gap."""
        return self._get_env_class().pos_gap

    turn_shift_frac: ClassVar[float] = 0.6
    turn_choices: ClassVar[List[str]] = ["straight", "turn90", "pivot180"]

    # Topple thresholds
    topple_angle_threshold: ClassVar[float] = 0.4

    def __init__(self,
                 num_dominos_max: int = 9,
                 num_targets_max: int = 3,
                 num_pivots_max: int = 3,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 domino_width: Optional[float] = None,
                 domino_depth: Optional[float] = None,
                 domino_height: Optional[float] = None) -> None:
        """Initialize the domino component.

        Args:
            num_dominos_max: Maximum number of domino blocks.
            num_targets_max: Maximum number of target objects.
            num_pivots_max: Maximum number of pivot objects.
            workspace_bounds: Dict with x/y/z lower/upper bounds.
            domino_width/depth/height: per-component dimension overrides (m).
                None (default) falls back to the shared
                PyBulletDominoComposedEnv ClassVars.
        """
        super().__init__()

        self.num_dominos_max = num_dominos_max
        self.num_targets_max = num_targets_max
        self.num_pivots_max = num_pivots_max
        self._dim_override = {
            "width": domino_width,
            "depth": domino_depth,
            "height": domino_height
        }

        # Workspace bounds (will be set by composed env if not provided)
        if workspace_bounds is None:
            workspace_bounds = {
                "x_lb": 0.4,
                "x_ub": 1.1,
                "y_lb": 1.1,
                "y_ub": 1.6,
                "z_lb": 0.4,  # table_height
                "z_ub": 0.95
            }
        self.x_lb = workspace_bounds["x_lb"]
        self.x_ub = workspace_bounds["x_ub"]
        self.y_lb = workspace_bounds["y_lb"]
        self.y_ub = workspace_bounds["y_ub"]
        self.z_lb = workspace_bounds["z_lb"]
        self.z_ub = workspace_bounds["z_ub"]

        # Domino-specific placement bounds (narrower than workspace) to avoid
        # placing dominoes too close to edges. The lower (robot-side) margin is
        # 1.5x the width: keeping the start block farther from the near edge
        # makes it reliably reachable for the push, which lifts the oracle
        # push-only solve rate from ~92% to ~99% (the misses were robot
        # reach/push failures, not cascade stalls) while keeping task diversity.
        # 1.1 + 1.5 * 0.07 = 1.205
        self.domino_y_lb = self.y_lb + 1.5 * self.domino_width
        # 1.6 - 0.21 = 1.39
        self.domino_y_ub = self.y_ub - 3 * self.domino_width
        self.domino_x_lb = self.x_lb
        self.domino_x_ub = self.x_ub

        # Create types. yaw/roll are radians: marking them angular lets
        # consumers that difference states (sysID residuals) wrap errors
        # to [-pi, pi] instead of scoring -pi vs +pi as a 2*pi mistake.
        self._domino_type = Type(
            "domino",
            ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"],
            angular_features=["yaw", "roll"],
        )
        # Separate agent-facing class for the gray blocks of heavy-block
        # tasks: same feature layout as a domino (so the shared body pool
        # and state assembly stay uniform), but a distinct type, so typed
        # options (Pick/Place/Push take dominoes) structurally exclude
        # them and the physical-param registry can expose a per-class
        # ``block_*`` parameter family. The name is deliberately neutral
        # ("block", not "heavy"): whether these bodies differ physically
        # from dominoes is exactly what a learning agent must discover.
        self._block_type = Type(
            "block",
            ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"],
            angular_features=["yaw", "roll"],
        )
        self._target_type = Type("target", ["x", "y", "z", "yaw"],
                                 sim_features=["id", "joint_id"],
                                 angular_features=["yaw"])
        self._pivot_type = Type("pivot", ["x", "y", "z", "yaw"],
                                sim_features=["id", "joint_id"],
                                angular_features=["yaw"])

        # Create objects
        use_domino_as_target = CFG.domino_use_domino_blocks_as_target
        if use_domino_as_target:
            num_dominos = self.num_dominos_max + self.num_targets_max
            num_targets = 0
        else:
            num_dominos = self.num_dominos_max
            num_targets = self.num_targets_max

        self.dominos: List[Object] = []
        for i in range(num_dominos):
            obj = Object(f"domino_{i}", self._domino_type)
            self.dominos.append(obj)
        # Heavy-block mode: the LAST slot of the shared body pool is the
        # gray block, minted as its own ``block``-typed object (the body
        # and all slot-indexed machinery are unchanged; only the object
        # identity differs).
        self.blocks: List[Object] = []
        if CFG.domino_heavy_block_tasks and num_dominos > 0:
            block_obj = Object("block_0", self._block_type)
            self.dominos[-1] = block_obj
            self.blocks.append(block_obj)

        self.targets: List[Object] = []
        for i in range(num_targets):
            obj = Object(f"target_{i}", self._target_type)
            self.targets.append(obj)

        self.pivots: List[Object] = []
        for i in range(self.num_pivots_max):
            obj = Object(f"pivot_{i}", self._pivot_type)
            self.pivots.append(obj)

        # Constraint tracking for connected dominoes
        self.block_constraints: List[int] = []
        self.fixed_domino_ids: List[int] = []
        # Bodies currently carrying heavy-block mass (gray blocks); like
        # fixed_domino_ids, rebuilt on every reset and shielded from the
        # generic ``mass`` override.
        self.heavy_domino_ids: List[int] = []

        # Optional per-instance override of PyBullet contact/inertial params
        # (mass, friction, restitution, ...). Empty by default, so the env
        # behaves exactly as its ClassVars dictate. Set via
        # ``set_physical_params`` to make one env instance's physics diverge
        # from another's (e.g. a miscalibrated planning sim vs. the "real"
        # env) *in the same process* without touching the shared ClassVars.
        # Re-applied at the end of ``reset_state`` because reset rewrites
        # domino mass on glue/unglue.
        self._physical_param_override: Dict[str, float] = {}

        # Create predicates
        self._create_predicates()

    def _create_predicates(self) -> None:
        """Create all predicates for this component."""
        if CFG.domino_use_domino_blocks_as_target:
            self._Toppled = Predicate("Toppled", [self._domino_type],
                                      self._Toppled_holds)
        else:
            self._Toppled = Predicate("Toppled", [self._target_type],
                                      self._Toppled_holds)

        self._Upright = Predicate("Upright", [self._domino_type],
                                  self._Upright_holds)
        self._Tilting = Predicate("Tilting", [self._domino_type],
                                  self._Tilting_holds)
        self._InitialBlock = Predicate("InitialBlock", [self._domino_type],
                                       self._StartBlock_holds)
        self._MovableBlock = Predicate("MovableBlock", [self._domino_type],
                                       self._MovableBlock_holds)
        self._DominoNotGlued = Predicate("DominoNotGlued", [self._domino_type],
                                         self._DominoNotGlued_holds)
        # Position-based InFront over continuous domino poses. When the grid is
        # in use, GridComponent's derived InFront replaces this one (helper
        # predicates take precedence on name collisions).
        self._InFront = Predicate(
            "InFront", [self._domino_type, self._domino_type],
            self._InFront_holds,
            natural_language_assertion=lambda os:
            ("the two dominoes are chain-adjacent: one sits one spacing-gap "
             "ahead of the other along that other's facing (toppling) "
             "direction -- straight or bent 45 degrees left/right for a turn, "
             "in both placement direction and yaw -- so that toppling the "
             "back domino knocks the front one over"))

    # -------------------------------------------------------------------------
    # DominoEnvComponent interface implementation
    # -------------------------------------------------------------------------

    def get_types(self) -> Set[Type]:
        types = {self._domino_type}
        if self.blocks:
            types.add(self._block_type)
        if self.targets:
            types.add(self._target_type)
        if self.pivots:
            types.add(self._pivot_type)
        return types

    def get_predicates(self) -> Set[Predicate]:
        preds = {
            self._Toppled,
            self._Upright,
            self._Tilting,
            self._InitialBlock,
            self._MovableBlock,
            self._InFront,
        }
        if CFG.domino_has_glued_dominos:
            preds.add(self._DominoNotGlued)
        return preds

    def get_goal_predicates(self) -> Set[Predicate]:
        return {self._Toppled}

    def get_objects(self) -> List[Object]:
        return self.dominos + self.targets + self.pivots

    def initialize_pybullet(self, physics_client_id: int) -> Dict[str, Any]:
        """Create PyBullet bodies for dominoes, targets, and pivots."""
        self._physics_client_id = physics_client_id
        bodies: Dict[str, Any] = {}

        # Create dominoes
        domino_ids = []
        num_dominos_to_create = len(self.dominos)
        for i in range(num_dominos_to_create):
            domino_id = create_domino_block(
                color=self.start_domino_color if i == 0 else self.domino_color,
                half_extents=(self.domino_width / 2, self.domino_depth / 2,
                              self.domino_height / 2),
                mass=self.domino_mass,
                friction=self.domino_friction,
                orientation=(0.0, 0.0, 0.0, 1.0),
                physics_client_id=physics_client_id,
                add_top_triangle=True,
            )
            domino_ids.append(domino_id)
        bodies["domino_ids"] = domino_ids

        # Create targets
        target_ids = []
        for _ in self.targets:
            tid = create_object("urdf/domino_target.urdf",
                                position=(self.x_lb, self.y_lb, self.z_lb),
                                orientation=p.getQuaternionFromEuler(
                                    [0.0, 0.0, 0.0]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            target_ids.append(tid)
        bodies["target_ids"] = target_ids

        # Create pivots
        pivot_ids = []
        for _ in self.pivots:
            pid = create_object("urdf/domino_pivot.urdf",
                                position=(self.x_lb, self.y_lb, self.z_lb),
                                orientation=p.getQuaternionFromEuler(
                                    [0.0, 0.0, 0.0]),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=physics_client_id)
            pivot_ids.append(pid)
        bodies["pivot_ids"] = pivot_ids

        return bodies

    def store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        """Store PyBullet body IDs on objects."""
        for domino, id_ in zip(self.dominos, pybullet_bodies["domino_ids"]):
            domino.id = id_

        for target, id_ in zip(self.targets, pybullet_bodies["target_ids"]):
            target.id = id_
            assert self._physics_client_id is not None
            target.joint_id = self._get_joint_id(id_, "flap_hinge_joint",
                                                 self._physics_client_id)

        for pivot, id_ in zip(self.pivots, pybullet_bodies["pivot_ids"]):
            pivot.id = id_
            assert self._physics_client_id is not None
            pivot.joint_id = self._get_joint_id(id_, "flap_hinge_joint",
                                                self._physics_client_id)
        # A freshly (re)created body has default ClassVar dynamics; re-assert
        # any standing override so it survives body recreation as well as
        # reset (see set_physical_params).
        self._apply_physical_param_override()

    # -------------------------------------------------------------------------
    # Per-instance physical-parameter override (system-ID / sim-vs-real)
    # -------------------------------------------------------------------------

    _PHYSICAL_PARAM_KEYS = frozenset({
        "mass", "lateral_friction", "restitution", "rolling_friction",
        "spinning_friction", "block_mass", "block_lateral_friction"
    })

    # Map override keys -> p.changeDynamics kwarg names. ``mass`` is handled
    # separately (skipped for glued/fixed dominoes carrying the 1e10 sentinel).
    _CHANGE_DYNAMICS_KW = {
        "lateral_friction": "lateralFriction",
        "restitution": "restitution",
        "rolling_friction": "rollingFriction",
        "spinning_friction": "spinningFriction",
    }

    def set_physical_params(self, **params: Optional[float]) -> None:
        """Override PyBullet contact/inertial params on the live domino bodies.

        Accepts any of ``mass``, ``lateral_friction`` (PyBullet's
        ``lateralFriction``, i.e. sliding friction), ``restitution``,
        ``rolling_friction``, ``spinning_friction``, ``block_mass``,
        ``block_lateral_friction`` (pass ``None`` to leave a param at its
        current value). The ``block_*`` variants apply only to the gray
        ``block``-typed bodies (and beat the global param for those
        bodies); ``block_mass`` is a planning sim's BELIEF about them
        (the true value is ``heavy_block_true_mass``, asserted at every
        reset). Applies
        ``p.changeDynamics`` to every
        domino body in *this* component's physics client, so one env
        instance's physics can diverge from another's without disturbing the
        shared ClassVars. The override is stored and re-applied after every
        ``reset_state`` (reset rewrites mass on glue/unglue) and after body
        recreation.

        Only affects dynamics-layer params (``changeDynamics``); domino
        *geometry* (width/height) is baked at body creation and is not
        changeable here.
        """
        provided = {k: v for k, v in params.items() if v is not None}
        unknown = set(provided) - self._PHYSICAL_PARAM_KEYS
        if unknown:
            raise ValueError(
                f"Unknown physical param(s) {sorted(unknown)}; "
                f"expected a subset of {sorted(self._PHYSICAL_PARAM_KEYS)}.")
        self._physical_param_override.update(provided)
        self._apply_physical_param_override()

    def clear_physical_params(self) -> None:
        """Drop the override (bodies keep their last-set values until
        reset)."""
        self._physical_param_override = {}

    @property
    def physical_param_override(self) -> Dict[str, float]:
        """Copy of the standing override (see ``set_physical_params``)."""
        return dict(self._physical_param_override)

    def _apply_physical_param_override(self) -> None:
        """Push the stored override onto the live domino bodies."""
        override = self._physical_param_override
        if not override or self._physics_client_id is None:
            return
        base_kwargs = {
            self._CHANGE_DYNAMICS_KW[k]: v
            for k, v in override.items() if k in self._CHANGE_DYNAMICS_KW
        }
        for domino in self.dominos:
            if domino.id is None:
                continue
            kwargs = dict(base_kwargs)
            # Don't clobber the 1e10 glue sentinel on fixed dominoes, nor
            # the heavy-block mass on gray blocks (which have their own
            # override key below).
            if "mass" in override and domino.id not in self.fixed_domino_ids \
                    and domino.id not in self.heavy_domino_ids:
                kwargs["mass"] = override["mass"]
            # ``block_*`` params override gray blocks only — a planning
            # sim believing gray blocks are ordinary dominoes sets
            # ``block_mass`` to the normal domino mass. A block-specific
            # value beats the global one for the same body.
            if "block_mass" in override \
                    and domino.id in self.heavy_domino_ids:
                kwargs["mass"] = override["block_mass"]
            if "block_lateral_friction" in override \
                    and domino.id in self.heavy_domino_ids:
                kwargs["lateralFriction"] = override["block_lateral_friction"]
            if kwargs:
                p.changeDynamics(domino.id,
                                 -1,
                                 physicsClientId=self._physics_client_id,
                                 **kwargs)

    def reset_state(self, state: State) -> None:
        """Reset dominoes, targets, and pivots to match state."""
        assert self._physics_client_id is not None
        # Gray blocks share the domino body pool but carry their own
        # type, so every pool sweep must cover both.
        domino_objs = (state.get_objects(self._domino_type) +
                       state.get_objects(self._block_type))

        # Remove old constraints
        for constraint in self.block_constraints:
            p.removeConstraint(constraint,
                               physicsClientId=self._physics_client_id)
        self.block_constraints = []

        # Restore normal dynamics to previously fixed/heavy dominoes
        for domino_id in self.fixed_domino_ids + self.heavy_domino_ids:
            p.changeDynamics(domino_id,
                             -1,
                             mass=self.domino_mass,
                             physicsClientId=self._physics_client_id)
        self.fixed_domino_ids = []
        self.heavy_domino_ids = []

        # Update domino colors to match state
        for domino in domino_objs:
            if domino.id is not None:
                r = state.get(domino, "r")
                g = state.get(domino, "g")
                b = state.get(domino, "b")
                update_object(domino.id,
                              color=(r, g, b, 1.0),
                              physics_client_id=self._physics_client_id)

        # Move dominoes absent from the state out of view (by identity,
        # not prefix count: heavy-block tasks use the LAST domino slot
        # for the gray block, so the used set need not be a prefix).
        used_dominos = set(domino_objs)
        oov_x, oov_y = self.out_of_view_xy
        for domino in self.dominos:
            if domino in used_dominos:
                continue
            oov_x += 0.1
            oov_y += 0.1
            update_object(domino.id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Reset targets
        target_objs = state.get_objects(self._target_type)
        for target_obj in target_objs:
            self._set_flat_rotation(target_obj, 0.0)
        for i in range(len(target_objs), len(self.targets)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.targets[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Reset pivots
        pivot_objs = state.get_objects(self._pivot_type)
        for pivot_obj in pivot_objs:
            self._set_flat_rotation(pivot_obj, 0.0)
        for i in range(len(pivot_objs), len(self.pivots)):
            oov_x += 0.1
            oov_y += 0.1
            update_object(self.pivots[i].id,
                          position=(oov_x, oov_y, self.domino_height / 2),
                          physics_client_id=self._physics_client_id)

        # Handle glued dominoes
        if CFG.domino_has_glued_dominos:
            for domino in domino_objs:
                if domino.id is not None:
                    if self._DominoGlued_holds(state, [domino]):
                        p.changeDynamics(
                            domino.id,
                            -1,
                            mass=1e10,
                            physicsClientId=self._physics_client_id)
                        self.fixed_domino_ids.append(domino.id)

        # Handle heavy (gray) blocks: true physics makes them untopple-able.
        # The believed mass, if any, is re-asserted by the override below.
        for domino in domino_objs:
            if domino.id is not None and self._HeavyBlock_holds(
                    state, [domino]):
                p.changeDynamics(domino.id,
                                 -1,
                                 mass=self.heavy_block_true_mass,
                                 physicsClientId=self._physics_client_id)
                self.heavy_domino_ids.append(domino.id)

        # Zero residual velocities on every domino body: pose resets go
        # through resetBasePositionAndOrientation, which does NOT clear
        # velocities — a body that was mid-fall when the previous rollout
        # ended would carry its momentum into this "static" scene. This
        # was the source of history-dependent probe/episode outcomes
        # (the same layout toppling or dying depending on what the sim
        # ran beforehand).
        for domino in self.dominos:
            if domino.id is not None:
                p.resetBaseVelocity(domino.id, [0, 0, 0], [0, 0, 0],
                                    physicsClientId=self._physics_client_id)

        # Re-assert any standing physical-param override: reset just rewrote
        # mass on the (un)glued dominoes above, so the override (if any) must
        # be re-applied to keep this instance's physics diverged.
        self._apply_physical_param_override()

    def extract_feature(self, obj: Object, feature: str) -> Optional[float]:
        """Extract feature for domino-related objects."""
        # Let the base environment handle position/orientation extraction
        return None

    def get_object_ids_for_held_check(self) -> List[int]:
        """Return domino and pivot IDs for held checking."""
        domino_ids = [d.id for d in self.dominos if d.id is not None]
        pivot_ids = [p.id for p in self.pivots if p.id is not None]
        return domino_ids + pivot_ids

    # -------------------------------------------------------------------------
    # Predicate hold functions
    # -------------------------------------------------------------------------

    def _Toppled_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if target/domino is toppled."""
        obj, = objects
        if CFG.domino_use_domino_blocks_as_target:
            roll_angle = abs(state.get(obj, "roll"))
            return roll_angle >= self.fallen_threshold
        rot_z = state.get(obj, "yaw")
        return abs(utils.wrap_angle(rot_z)) < 0.8

    def _Upright_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if domino is upright."""
        obj, = objects
        tilt_angle = state.get(obj, "roll")
        return abs(tilt_angle) < self.domino_roll_threshold

    def _Tilting_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if domino is tilting (in transition)."""
        obj, = objects
        roll_angle = abs(state.get(obj, "roll"))
        return self.domino_roll_threshold <= roll_angle < self.fallen_threshold

    @classmethod
    def _StartBlock_holds(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        """Check if domino is the start block (light green)."""
        domino, = objects
        eps = 1e-3
        return (
            abs(state.get(domino, "r") - cls.start_domino_color[0]) < eps
            and abs(state.get(domino, "g") - cls.start_domino_color[1]) < eps
            and abs(state.get(domino, "b") - cls.start_domino_color[2]) < eps)

    @classmethod
    def _MovableBlock_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        """Check if domino is a movable block (blue)."""
        domino, = objects
        eps = 1e-3
        return (abs(state.get(domino, "r") - cls.domino_color[0]) < eps
                and abs(state.get(domino, "g") - cls.domino_color[1]) < eps
                and abs(state.get(domino, "b") - cls.domino_color[2]) < eps)

    @classmethod
    def _HeavyBlock_holds(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        """Check if domino is a heavy (immovable, gray) block."""
        domino, = objects
        return cls.is_heavy_color(state.get(domino,
                                            "r"), state.get(domino, "g"),
                                  state.get(domino, "b"))

    @classmethod
    def is_heavy_color(cls, r: float, g: float, b: float) -> bool:
        """Whether an (r, g, b) triple is the heavy-block gray."""
        eps = 1e-3
        return (abs(r - cls.heavy_block_color[0]) < eps
                and abs(g - cls.heavy_block_color[1]) < eps
                and abs(b - cls.heavy_block_color[2]) < eps)

    @classmethod
    def _TargetDomino_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        """Check if domino is a target (pink or glued red)."""
        domino, = objects
        eps = 1e-3
        return (cls._DominoGlued_holds(state, objects)) or (
            abs(state.get(domino, "r") - cls.target_domino_color[0]) < eps
            and abs(state.get(domino, "g") - cls.target_domino_color[1]) < eps
            and abs(state.get(domino, "b") - cls.target_domino_color[2]) < eps)

    @classmethod
    def _DominoNotGlued_holds(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        """Check if domino is NOT glued."""
        return not cls._DominoGlued_holds(state, objects)

    def _InFront_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Position-based ``InFront`` classifier over continuous poses.

        ``InFront(d1, d2)`` holds when one domino sits roughly one
        ``pos_gap`` ahead of the other along that other's facing
        (toppling) direction, with a discrete turn offset between their
        yaws (straight / 45-left / 45-right). It reads the continuous
        domino poses directly, so it is available to grid-free agent
        approaches.
        """
        domino1, domino2 = objects
        if state.get(domino1, "is_held") or state.get(domino2, "is_held"):
            return False

        pos_gap = self.pos_gap
        pos_tol = pos_gap * 0.3
        ang_tol = np.radians(15)
        # Cardinal-facing slack for the reference (back) domino. A domino
        # the robot re-places settles ~1 deg off cardinal, so a 1e-3 rad
        # (~0.06 deg) gate makes InFront(front, placed_back) unsatisfiable
        # for chained placements; allow a few degrees of slack instead.
        card_thresh = float(np.sin(np.radians(10)))
        # Straight, 45-degree right turn, and 45-degree left turn.
        turn_offsets = (-np.pi / 4, 0.0, np.pi / 4)

        def _ahead(back: Object, front: Object) -> bool:
            x_b = state.get(back, "x")
            y_b = state.get(back, "y")
            rot_b = state.get(back, "yaw")
            # The relationship only holds for (roughly) cardinal back-facings.
            if not (abs(np.sin(rot_b)) < card_thresh
                    or abs(np.cos(rot_b)) < card_thresh):
                return False
            # The front domino's yaw differs from the back's by a discrete
            # turn offset (straight / +-45 deg).
            diff = utils.wrap_angle(state.get(front, "yaw") - rot_b)
            if not any(abs(diff - off) < ang_tol for off in turn_offsets):
                return False
            # The front domino sits one pos_gap from the back, along the
            # back's facing -- which may itself be rotated by a turn offset,
            # so the chain can bend through a turn (the next block then lies
            # diagonally off the back rather than straight ahead).
            fx = state.get(front, "x")
            fy = state.get(front, "y")
            # A domino is 180-degree symmetric, so its facing names a
            # bidirectional topple axis: the front may sit one gap along
            # either end of that (possibly turn-rotated) axis.
            #
            # A turn-completing block always carries a half-width lateral
            # ("side") offset, applied orthogonal to the reference's facing
            # by the task generator (see DominoTaskGenerator.
            # _place_turn90_domino) so the toppling chain stays overlapping
            # through the corner. A turn placement (dir_off != 0) therefore
            # sits at +-side_offset along the perpendicular -- NOT on the bare
            # axis. Excluding lateral 0 here is what lets the Place sampler
            # distinguish the cascade-enabling offset pose from the
            # symbolically-equivalent-but-physically-dead on-axis pose (an
            # on-axis turn block fails this edge, so scoring prefers the
            # offset). Straight placements (dir_off == 0) stay exactly on the
            # axis, so no spurious edges appear.
            side_offset = self.domino_width / 2
            perp_x = np.cos(rot_b)
            perp_y = -np.sin(rot_b)
            for dir_off in turn_offsets:
                ang = rot_b + dir_off
                laterals = ((0.0, ) if abs(dir_off) < 1e-9 else
                            (side_offset, -side_offset))
                for sgn in (1.0, -1.0):
                    base_x = x_b + sgn * pos_gap * np.sin(ang)
                    base_y = y_b + sgn * pos_gap * np.cos(ang)
                    for lat in laterals:
                        expected_x = base_x + lat * perp_x
                        expected_y = base_y + lat * perp_y
                        if (abs(fx - expected_x) < pos_tol
                                and abs(fy - expected_y) < pos_tol):
                            return True
            return False

        # InFront(d1, d2) := d1 is ahead of d2, or d2 is ahead of d1.
        return _ahead(domino2, domino1) or _ahead(domino1, domino2)

    @classmethod
    def _DominoGlued_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        """Check if domino is glued (red color)."""
        eps = 1e-3
        r_val = state.get(objects[0], "r")
        g_val = state.get(objects[0], "g")
        b_val = state.get(objects[0], "b")
        return (abs(r_val - cls.glued_domino_color[0]) < eps
                and abs(g_val - cls.glued_domino_color[1]) < eps
                and abs(b_val - cls.glued_domino_color[2]) < eps)

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_joint_id(obj_id: int,
                      joint_name: str,
                      physics_client_id: int = 0) -> int:
        """Get joint ID by name from PyBullet object."""
        num_joints = p.getNumJoints(obj_id, physicsClientId=physics_client_id)
        for j in range(num_joints):
            info = p.getJointInfo(obj_id, j, physicsClientId=physics_client_id)
            if info[1].decode("utf-8") == joint_name:
                return j
        return -1

    def _set_flat_rotation(self, flap_obj: Object, rot: float = 0.0) -> None:
        """Set rotation of a hinged object (target/pivot)."""
        p.resetJointState(flap_obj.id,
                          flap_obj.joint_id,
                          rot,
                          physicsClientId=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Sequence generation helpers
    # -------------------------------------------------------------------------

    def place_domino(self,
                     _domino_idx: int,
                     x: float,
                     y: float,
                     rot: float,
                     is_start_block: bool = False,
                     is_target_block: bool = False,
                     is_heavy_block: bool = False,
                     rng: Optional[np.random.Generator] = None,
                     task_idx: Optional[int] = None) -> Dict:
        """Create a dictionary with placement parameters for a domino."""
        if is_heavy_block:
            color = self.heavy_block_color
        elif is_start_block:
            color = self.start_domino_color
        elif is_target_block:
            should_be_glued = False
            if CFG.domino_has_glued_dominos:
                if task_idx == 0:
                    should_be_glued = True
                elif task_idx == 1:
                    should_be_glued = False
                else:
                    should_be_glued = (rng is not None and
                                       rng.random() < self.glued_percentage)
            color = (self.glued_domino_color
                     if should_be_glued else self.target_domino_color)
        else:
            color = self.domino_color

        return {
            "x": x,
            "y": y,
            "z": self.z_lb + self.domino_height / 2,
            "yaw": rot,
            "roll": 0.0,
            "r": color[0],
            "g": color[1],
            "b": color[2],
            "is_held": 0.0,
        }

    def place_pivot_or_target(self,
                              x: float,
                              y: float,
                              rot: float = 0.0) -> Dict:
        """Create a dictionary with placement parameters for a pivot/target."""
        return {
            "x": x,
            "y": y,
            "z": self.z_lb,
            "yaw": rot,
        }

    # -------------------------------------------------------------------------
    # Public properties for type access
    # -------------------------------------------------------------------------

    @property
    def domino_type(self) -> Type:
        """Domino type."""
        return self._domino_type

    @property
    def block_type(self) -> Type:
        """Block type (the gray blocks of heavy-block tasks)."""
        return self._block_type

    @property
    def target_type(self) -> Type:
        """Target type."""
        return self._target_type

    @property
    def pivot_type(self) -> Type:
        """Pivot type."""
        return self._pivot_type

    @property
    def Toppled(self) -> Predicate:
        """Toppled."""
        return self._Toppled


def create_domino_block(
    color: Tuple[float, float, float, float],
    half_extents: Tuple[float, float, float],
    mass: float,
    friction: float,
    position: Pose3D = (0.0, 0.0, 0.0),
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    physics_client_id: int = 0,
    add_top_triangle: bool = False,
    *,
    restitution: float = 0.02,
    rolling_friction: float = 0.006,
    spinning_friction: Optional[float] = None,
    linear_damping: float = 0.0,
    angular_damping: float = 0.03,
    friction_anchor: bool = True,
    ccd: bool = True,
    ccd_swept_radius: Optional[float] = None,
) -> int:
    """Create a domino-tuned block with appropriate physics settings."""
    block_id = create_pybullet_block(
        color=color,
        half_extents=half_extents,
        mass=mass,
        friction=friction,
        position=position,
        orientation=orientation,
        physics_client_id=physics_client_id,
        add_top_triangle=add_top_triangle,
    )

    if spinning_friction is None:
        spinning_friction = friction

    p.changeDynamics(
        block_id,
        linkIndex=-1,
        lateralFriction=friction,
        rollingFriction=rolling_friction,
        spinningFriction=spinning_friction,
        restitution=restitution,
        linearDamping=linear_damping,
        angularDamping=angular_damping,
        frictionAnchor=friction_anchor,
        physicsClientId=physics_client_id,
    )

    if ccd:
        m = min(half_extents)
        swept = ccd_swept_radius if ccd_swept_radius is not None else 0.5 * m
        p.changeDynamics(
            block_id,
            linkIndex=-1,
            ccdSweptSphereRadius=swept,
            physicsClientId=physics_client_id,
        )

    return block_id
