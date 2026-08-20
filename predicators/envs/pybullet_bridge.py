"""A PyBullet environment where the robot builds an "n"-shaped bridge by gluing
rectangular blocks with a pickable glue bottle.

Motivating the partial-observability + slow-process story of
``AgentPOSimPredicateInventionApproach`` with a hidden process whose
consequence is *kinematic* rather than a feature readout: once a glue
joint cures, the two blocks are welded into one rigid assembly (a
body-to-body ``JOINT_FIXED`` constraint), so picking any block of the
assembly transports the whole thing. No other domain in the suite makes
the hidden latent change what actions *do*.

Mechanics:

- Every block exposes three glue-able faces: ``top`` (+z) and the two
  long-axis ends ``end_a``/``end_b`` (local -x/+x).
- The robot picks up the glue ``bottle``, holds its tip near a face's
  dab point to wet that face, and puts the bottle back down.
- While a wet face is in aligned resting contact with another block
  (neither block held), that joint's hidden ``cure_*`` counter ticks;
  at ``cure_threshold`` the joint irreversibly latches: the wet glue is
  consumed, both blocks record the attachment (``attached_*`` = partner
  block index), and a physical weld constraint is created.
- Interrupting the contact resets the counter (wet glue persists).

Task ("n"-shaped bridge), two sizes via ``CFG.bridge_task_spec_*``:

- **simple** (4 blocks): stand one leg block at each marked site, glue
  their tops, glue two span blocks end-to-end on the table, then seat
  the cured span assembly across the legs. 3 joints.
- **full** (7 blocks): each leg is a glued 2-block stack and the span
  is 3 blocks glued end-to-end. 6 joints.

In partially-observable mode (``CFG.partially_observable``) the
``cure_*`` counters are dropped from the observation; the agent sees
only wet-glue flags, the discrete ``attached_*`` flips, and the
kinematic consequences, and must infer the hidden dwell threshold.

Example command (oracle demo via bilevel process planning)::

    python predicators/main.py --env pybullet_bridge \
        --approach oracle_process_planning --seed 0 \
        --num_train_tasks 0 --num_test_tasks 5 \
        --sesame_check_expected_atoms False
"""

from typing import Any, ClassVar, Dict, FrozenSet, List, Optional, Sequence, \
    Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block, update_object
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type

# Faces that can be wetted with glue (block-local frame).
GLUE_FACES = ("top", "end_a", "end_b")
# Attachment slots: a cured joint occupies one slot on each block.
# ``bottom`` exists because a stack/seat joint welds a wet ``top`` to
# the underside of the partner.
ATTACH_SLOTS = ("top", "bottom", "end_a", "end_b")


class PyBulletBridgeEnv(PyBulletEnv):
    """Build an n-shaped bridge by gluing blocks; cured joints physically weld
    blocks into rigid assemblies.

    Two block schemas, one logical type: the fully-observable
    ``_block_type`` carries the ``cure_*`` counters as observable
    features, while ``_block_type_po`` drops them. Both keep the
    counters as ``sim_features`` so the ``block.cure_top`` (etc.) Python
    attributes -- the internal source of truth -- always drive the
    dynamics. ``__init__`` swaps the type when
    ``CFG.partially_observable`` is set.
    """

    # -------------------------------------------------------------------------
    # Table / workspace config (mirrors pybullet_bond)
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
    # Geometry
    # -------------------------------------------------------------------------
    # Legs stand (long axis = local z); spans lie (long axis = local x).
    # Two collision shapes, one logical type: every block then lives at
    # yaw-only orientations forever, so the type carries just `rot` and
    # we never hit the Euler round-trip gimbal traps that a reoriented
    # single shape would.
    leg_half_extents: ClassVar[Tuple[float, float,
                                     float]] = (0.025, 0.025, 0.05)
    span_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.05, 0.025, 0.025)
    block_mass: ClassVar[float] = 0.1
    bottle_half_extents: ClassVar[Tuple[float, float,
                                        float]] = (0.012, 0.012, 0.03)
    site_half_extents: ClassVar[Tuple[float, float,
                                      float]] = (0.045, 0.045, 0.0001)
    # Sites (where the legs stand) sit in a front band, staging behind.
    site_y: ClassVar[float] = 1.16
    site_sep_simple: ClassVar[float] = 0.15
    site_sep_full: ClassVar[float] = 0.25
    site_x_jitter: ClassVar[float] = 0.05
    # A tiny nominal gap for lateral butt joints so a placed block does
    # not shove its neighbor.
    lateral_place_gap: ClassVar[float] = 0.002
    # EE transport height for carried blocks/assemblies. Must clear the
    # tallest structure (full-variant leg stacks top out at 0.60).
    transport_z: ClassVar[float] = 0.70

    # -------------------------------------------------------------------------
    # Domain-specific config
    # -------------------------------------------------------------------------
    # Consecutive aligned-contact steps for a wet joint to cure. Must
    # comfortably exceed the Place option's own duration (~14 steps):
    # if curing completes during the Place retreat, the subsequent Wait
    # starts with its target atom already true, its first action is a
    # no-op, and the option model's repeat-state check kills it.
    cure_threshold: ClassVar[int] = 25
    # Bottle-tip proximity to a face's dab point that wets the face.
    # Only the single nearest in-range face is wetted per step, so
    # neighboring dab points (>= 2.5 cm apart) don't double-wet.
    apply_glue_radius: ClassVar[float] = 0.02
    # Dab points hover this far off the face surface.
    dab_margin: ClassVar[float] = 0.005
    # Stacking tolerances for OnBlock(top, bottom) (leg-on-leg).
    stack_align_tol: ClassVar[float] = 0.025
    stack_z_tol: ClassVar[float] = 0.02
    # Lateral butt-joint window for NextToEnd(right, left): projection
    # of the center offset onto the end direction, minus the two half
    # lengths, must land in [-0.01, +0.012] (contact up to a ~1 cm gap).
    lateral_proj_tol_lo: ClassVar[float] = 0.01
    lateral_proj_tol_hi: ClassVar[float] = 0.012
    # Perp/y tolerances must exceed real place-execution accuracy: a
    # span dropped ~1.5 cm can land ~2 cm off in y, and a lateral
    # placement error is FROZEN into the weld, shifting where the far
    # span meets its leg. A 3 cm miss still leaves 2 cm of overlap on
    # the 5 cm leg top, so the joint is physically sound; a 2 cm gate
    # left built bridges with one seat joint that could never cure.
    lateral_perp_tol: ClassVar[float] = 0.03
    lateral_z_tol: ClassVar[float] = 0.015
    # Seat tolerances for SeatedOn(span, leg).
    seat_x_window: ClassVar[float] = 0.045
    seat_y_tol: ClassVar[float] = 0.035
    seat_z_tol: ClassVar[float] = 0.02
    # AtSite xy tolerance (plus a z check that the block rests on the
    # table, so a stacked upper leg is not "at" the site). The site pad
    # is 9 cm wide; 4 cm absorbs placement error plus the nudge the
    # seat landing gives the legs (a marginal AtSite flicking false
    # mid-episode sends the replanner after unreachable re-placements).
    at_site_tol: ClassVar[float] = 0.04
    at_site_z_tol: ClassVar[float] = 0.02
    # Weld constraint strength. PyBullet's default (500, the same the
    # grasp constraint uses) already holds in probes; the high value
    # removes sag under a cantilevered span.
    weld_max_force: ClassVar[float] = 10000.0
    # Staging slots (see _stage_objects). Random rejection sampling
    # cannot pack the full variant's 8 objects + assembly strip + site
    # keepouts into the reachable lens (it saturates around 5 objects),
    # so staging assigns objects to a jittered grid instead: 5 columns
    # x 2 back rows, plus whatever front-row (site-band) slots clear
    # both sites. The span row is assembled along the middle row.
    stage_cols: ClassVar[Tuple[float, ...]] = (0.52, 0.63, 0.74, 0.85, 0.96)
    stage_row_front: ClassVar[float] = 1.16  # shared with the sites
    stage_row_mid: ClassVar[float] = 1.26  # span row assembled here
    stage_row_back: ClassVar[float] = 1.36
    stage_jitter: ClassVar[float] = 0.008
    site_keepout: ClassVar[float] = 0.09
    reach_radius: ClassVar[float] = 0.78

    # Colors
    site_color: ClassVar[Tuple[float, float, float,
                               float]] = (0.6, 0.6, 0.6, 1.0)  # gray
    bottle_color: ClassVar[Tuple[float, float, float,
                                 float]] = (0.9, 0.9, 0.98, 1.0)  # off-white
    glue_wet_color: ClassVar[Tuple[float, float, float,
                                   float]] = (0.95, 0.85, 0.25, 0.9)  # yellow

    # -------------------------------------------------------------------------
    # Types
    # -------------------------------------------------------------------------
    _robot_type = Type("robot",
                       ["x", "y", "z", "fingers", "roll", "tilt", "wrist"],
                       angular_features=["roll", "tilt", "wrist"])
    # `glue_*` (wet-glue flags) and `attached_*` (partner block index,
    # -1 if none) are observable in both modes. `cure_*` are the hidden
    # per-joint dwell counters: observable in FO mode, dropped in the PO
    # variant. All stay sim_features so the Python attributes (the
    # internal source of truth) always drive the dynamics.
    _block_features_common = [
        "x", "y", "z", "rot", "is_held", "upright", "glue_top", "glue_end_a",
        "glue_end_b"
    ]
    _block_features_tail = [
        "attached_top", "attached_bottom", "attached_end_a", "attached_end_b",
        "r", "g", "b"
    ]
    _block_sim_features = [
        "id", "glue_top", "glue_end_a", "glue_end_b", "cure_top", "cure_end_a",
        "cure_end_b", "attached_top", "attached_bottom", "attached_end_a",
        "attached_end_b"
    ]
    _block_type = Type("block",
                       _block_features_common +
                       ["cure_top", "cure_end_a", "cure_end_b"] +
                       _block_features_tail,
                       sim_features=_block_sim_features,
                       angular_features=["rot"])
    _block_type_po = Type("block",
                          _block_features_common + _block_features_tail,
                          sim_features=_block_sim_features,
                          angular_features=["rot"])
    _bottle_type = Type("bottle", ["x", "y", "z", "rot", "is_held"],
                        sim_features=["id"],
                        angular_features=["rot"])
    _site_type = Type("site", ["x", "y", "z"], sim_features=["id"])

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        # In partial-observability mode, swap the block type to the
        # variant without `cure_*` *before* any blocks/predicates are
        # built, so the reduced schema propagates everywhere.
        if CFG.partially_observable:
            self._block_type = self._block_type_po

        # Robot
        self._robot = Object("robot", self._robot_type)

        # Blocks: 4 leg-shaped + 3 span-shaped, fixed names/roles.
        self._legs = [Object(f"leg{i}", self._block_type) for i in range(4)]
        self._spans = [Object(f"span{i}", self._block_type) for i in range(3)]
        self._blocks: List[Object] = self._legs + self._spans
        self._block_index: Dict[str, int] = {
            blk.name: i
            for i, blk in enumerate(self._blocks)
        }

        # Glue bottle and the two leg sites.
        self._bottle = Object("bottle", self._bottle_type)
        self._sites = [Object(f"site{i}", self._site_type) for i in range(2)]

        # Live weld constraints: frozenset({body_id_a, body_id_b}) ->
        # PyBullet constraint id. Must exist before super().__init__
        # (reset paths may call _set_domain_specific_state).
        self._weld_constraints: Dict[FrozenSet[int], int] = {}
        # Glue-patch visual bodies: block name -> face -> body id.
        self._glue_patch_ids: Dict[str, Dict[str, int]] = {}

        super().__init__(use_gui, **kwargs)

        # Predicates
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._block_type],
                                  self._Holding_holds)
        self._HoldingBottle = Predicate("HoldingBottle",
                                        [self._robot_type, self._bottle_type],
                                        self._HoldingBottle_holds)
        self._GlueTop = Predicate("GlueTop", [self._block_type],
                                  self._make_glue_holds("top"))
        self._GlueEndA = Predicate("GlueEndA", [self._block_type],
                                   self._make_glue_holds("end_a"))
        self._GlueEndB = Predicate("GlueEndB", [self._block_type],
                                   self._make_glue_holds("end_b"))
        self._OnBlock = Predicate("OnBlock",
                                  [self._block_type, self._block_type],
                                  self._OnBlock_holds)
        self._NextToEnd = Predicate("NextToEnd",
                                    [self._block_type, self._block_type],
                                    self._NextToEnd_holds)
        self._SeatedOn = Predicate("SeatedOn",
                                   [self._block_type, self._block_type],
                                   self._SeatedOn_holds)
        self._AtSite = Predicate("AtSite", [self._block_type, self._site_type],
                                 self._AtSite_holds)
        self._SiteFree = Predicate("SiteFree", [self._site_type],
                                   self._SiteFree_holds)
        self._Attached = Predicate("Attached",
                                   [self._block_type, self._block_type],
                                   self._Attached_holds)
        self._Connected = Predicate("Connected",
                                    [self._block_type, self._block_type],
                                    self._Connected_holds)
        # Static shape predicates (planning-time grounding pruners) and
        # Loose (no cured attachments -- a block welded into an
        # assembly cannot be individually re-placed).
        self._Standing = Predicate("Standing", [self._block_type],
                                   lambda s, o: s.get(o[0], "upright") > 0.5)
        self._Lying = Predicate("Lying", [self._block_type],
                                lambda s, o: s.get(o[0], "upright") <= 0.5)
        self._Loose = Predicate("Loose", [self._block_type], self._Loose_holds)
        # Resting = not held. Pick processes delete it and place
        # processes re-add it, so the cure processes can require it
        # throughout their delay: picking a block mid-cure abstractly
        # aborts the cure, exactly matching the env's counter reset.
        self._Resting = Predicate("Resting", [self._block_type],
                                  lambda s, o: s.get(o[0], "is_held") <= 0.5)
        # TopFree = nothing rests on the block's top face. ApplyGlueTop
        # requires it: without it, a replan after a failed seat cure
        # would try to glue a leg top BURIED under the seated span (the
        # dab goal is inside the span, so the option can never execute,
        # and the ensuing retries wreck the built structure).
        self._TopFree = Predicate("TopFree", [self._block_type],
                                  self._TopFree_holds)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_bridge"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._HandEmpty, self._Holding, self._HoldingBottle, self._GlueTop,
            self._GlueEndA, self._GlueEndB, self._OnBlock, self._NextToEnd,
            self._SeatedOn, self._AtSite, self._SiteFree, self._Attached,
            self._Connected, self._Standing, self._Lying, self._Loose,
            self._Resting, self._TopFree
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # Goals pin the full geometric layout (not just Attached): the
        # extra atoms force the planner's bindings to a physically
        # consistent left-to-right build (see processes.py docstring)
        # and all of them persist in the finished bridge.
        return {
            self._Attached, self._AtSite, self._OnBlock, self._NextToEnd,
            self._SeatedOn
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._block_type, self._bottle_type,
            self._site_type
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

        # Table
        table_id = create_object(asset_path="urdf/table.urdf",
                                 position=cls.table_pos,
                                 orientation=cls.table_orn,
                                 scale=1.0,
                                 use_fixed_base=True,
                                 physics_client_id=physics_client_id)
        bodies["table_id"] = table_id

        # Blocks: 4 legs (standing shape) + 3 spans (lying shape).
        block_ids = []
        for half_extents, count in ((cls.leg_half_extents, 4),
                                    (cls.span_half_extents, 3)):
            for _ in range(count):
                block_id = create_pybullet_block(
                    color=(0.5, 0.5, 0.9, 1.0),
                    half_extents=half_extents,
                    mass=cls.block_mass,
                    friction=1.0,
                    physics_client_id=physics_client_id)
                # Damp post-landing slide/twist (see pybullet_bond).
                p.changeDynamics(block_id,
                                 -1,
                                 spinningFriction=0.1,
                                 rollingFriction=0.01,
                                 physicsClientId=physics_client_id)
                block_ids.append(block_id)
        bodies["block_ids"] = block_ids

        # Glue bottle (slim box, top-graspable).
        bottle_id = create_pybullet_block(color=cls.bottle_color,
                                          half_extents=cls.bottle_half_extents,
                                          mass=0.05,
                                          friction=1.0,
                                          physics_client_id=physics_client_id)
        bodies["bottle_id"] = bottle_id

        # Two site pads (thin fixed plates marking the leg positions).
        site_ids = []
        for _ in range(2):
            site_id = create_pybullet_block(
                color=cls.site_color,
                half_extents=cls.site_half_extents,
                mass=0,
                friction=0.5,
                physics_client_id=physics_client_id)
            site_ids.append(site_id)
        bodies["site_ids"] = site_ids

        # Wet-glue visual patches: one per block face, collision-free
        # (baseCollisionShapeIndex=-1), parked out of view when dry.
        # PyBullet can't tint one face of a single-shape body, so these
        # carry the "this face is wet" rendering.
        patch_ids: List[List[int]] = []
        oov_x, oov_y = cls._out_of_view_xy
        for i, half_extents in enumerate([cls.leg_half_extents] * 4 +
                                         [cls.span_half_extents] * 3):
            hx, hy, hz = half_extents
            per_face = []
            for face in GLUE_FACES:
                if face == "top":
                    patch_half = (hx - 0.001, hy - 0.001, 0.0015)
                else:
                    patch_half = (0.0015, hy - 0.001, hz - 0.001)
                vis_id = p.createVisualShape(p.GEOM_BOX,
                                             halfExtents=patch_half,
                                             rgbaColor=cls.glue_wet_color,
                                             physicsClientId=physics_client_id)
                patch_id = p.createMultiBody(baseMass=0,
                                             baseCollisionShapeIndex=-1,
                                             baseVisualShapeIndex=vis_id,
                                             basePosition=(oov_x, oov_y,
                                                           -1.0 - 0.1 * i),
                                             physicsClientId=physics_client_id)
                per_face.append(patch_id)
            patch_ids.append(per_face)
        bodies["glue_patch_ids"] = patch_ids

        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._table_ids = [pybullet_bodies["table_id"]]
        self._robot.id = self._pybullet_robot.robot_id
        for i, blk in enumerate(self._blocks):
            blk.id = pybullet_bodies["block_ids"][i]
        self._bottle.id = pybullet_bodies["bottle_id"]
        for i, site in enumerate(self._sites):
            site.id = pybullet_bodies["site_ids"][i]
        self._glue_patch_ids = {
            blk.name:
            dict(zip(GLUE_FACES, pybullet_bodies["glue_patch_ids"][i]))
            for i, blk in enumerate(self._blocks)
        }

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _attr(blk: Object, name: str, default: float) -> float:
        """Read a sim-feature attribute with an explicit None default (0.0 is a
        meaningful value for attached_* -- block index 0)."""
        val = getattr(blk, name)
        return float(val) if val is not None else default

    @classmethod
    def _is_leg_shaped(cls, blk: Object) -> bool:
        return blk.name.startswith("leg")

    @classmethod
    def _block_half_extents(cls, blk: Object) -> Tuple[float, float, float]:
        return cls.leg_half_extents if cls._is_leg_shaped(blk) \
            else cls.span_half_extents

    @classmethod
    def _face_world_dir(cls, state: State, blk: Object,
                        face: str) -> Tuple[float, float, float]:
        """Outward unit normal of a face in world frame (yaw-only)."""
        if face == "top":
            return (0.0, 0.0, 1.0)
        yaw = state.get(blk, "rot")
        sign = -1.0 if face == "end_a" else 1.0
        return (sign * float(np.cos(yaw)), sign * float(np.sin(yaw)), 0.0)

    @classmethod
    def _face_dab_point(cls, state: State, blk: Object,
                        face: str) -> Tuple[float, float, float]:
        """Where the bottle tip must hover to wet this face.

        Top faces: just above the face center. End faces: just above the
        top edge of the (vertical) end face, so the dab always comes
        from above (no sideways IK). Classmethod so the ApplyGlue skill
        (options.py) can share the exact geometry.
        """
        x = state.get(blk, "x")
        y = state.get(blk, "y")
        z = state.get(blk, "z")
        hx, _, hz = cls._block_half_extents(blk)
        if face == "top":
            return (x, y, z + hz + cls.dab_margin)
        dx, dy, _ = cls._face_world_dir(state, blk, face)
        return (x + dx * hx, y + dy * hx, z + hz + cls.dab_margin)

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------
    def _get_object_ids_for_held_check(self) -> List[int]:
        ids = [blk.id for blk in self._blocks if blk.id is not None]
        if self._bottle.id is not None:
            ids.append(self._bottle.id)
        return ids

    def _get_domain_specific_feature(self, obj: Object, feature: str) -> float:
        if obj.type in (self._block_type, self._block_type_po):
            if feature == "upright":
                return 1.0 if self._is_leg_shaped(obj) else 0.0
            if feature.startswith("glue_") or feature.startswith("cure_"):
                return self._attr(obj, feature, 0.0)
            if feature.startswith("attached_"):
                return self._attr(obj, feature, -1.0)
        raise ValueError(f"Unknown feature {feature} for object {obj}.")

    def _get_state(self, _render_obs: bool = False) -> State:
        """PyBullet -> State, plus the privileged (hidden) cure block.

        In partially-observable mode the ``cure_*`` counters are not
        observable features, so snapshot each block's true internal
        counters into ``state.privileged`` -- the env-only channel the
        agent never sees -- so backtracking restores each search node's
        own counters.
        """
        state = super()._get_state(_render_obs)
        if CFG.partially_observable:
            state.privileged = {
                blk.name: {
                    f"cure_{face}": self._attr(blk, f"cure_{face}", 0.0)
                    for face in GLUE_FACES
                }
                for blk in state.get_objects(self._block_type)
            }
        return state

    def _set_domain_specific_state(self, state: State) -> None:
        # Restore each block's internal glue / cure / attachment state.
        blocks = state.get_objects(self._block_type)
        for blk in blocks:
            for face in GLUE_FACES:
                setattr(blk, f"glue_{face}", state.get(blk, f"glue_{face}"))
                if f"cure_{face}" in blk.type.feature_names:
                    setattr(blk, f"cure_{face}",
                            state.get(blk, f"cure_{face}"))
                else:
                    priv = state.privileged or {}
                    setattr(
                        blk, f"cure_{face}",
                        float(priv.get(blk.name, {}).get(f"cure_{face}", 0.0)))
            for slot in ATTACH_SLOTS:
                setattr(blk, f"attached_{slot}",
                        state.get(blk, f"attached_{slot}"))
            # Colors are task-assigned features; the base env never
            # writes them to PyBullet, so apply them here.
            if blk.id is not None:
                update_object(blk.id,
                              color=(state.get(blk, "r"), state.get(blk, "g"),
                                     state.get(blk, "b"), 1.0),
                              physics_client_id=self._physics_client_id)

        # Sync physical weld constraints to the restored attachment
        # features. Handles planner backtracking to pre-cure nodes and
        # cross-episode residuals (a fresh task has all attached = -1,
        # so every stale weld is removed).
        self._sync_welds_to_state(state)

        # Wet-glue patch visuals.
        self._update_glue_patches(state)

        # Move irrelevant blocks out of view.
        oov_x, oov_y = self._out_of_view_xy
        in_state = set(blocks)
        for i, blk in enumerate(self._blocks):
            if blk not in in_state and blk.id is not None:
                update_object(blk.id,
                              position=(oov_x + 0.3 * i, oov_y, 0.0),
                              physics_client_id=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Weld constraint lifecycle
    # -------------------------------------------------------------------------
    def _create_weld(self,
                     body_a: int,
                     body_b: int,
                     ideal_dz: Optional[float] = None) -> None:
        """Create a body-to-body JOINT_FIXED weld at the CURRENT relative pose
        (see pybullet_coffee's plugged-in constraint).

        The relative transform is SNAPPED before freezing: relative
        roll/pitch are zeroed, and when ``ideal_dz`` is given (the
        joint's nominal vertical offset, known from the attachment slot)
        the relative z is set to it. An unsnapped weld freezes a
        millimeter-level inconsistency against the resting plane, and
        the constraint solver then applies steady micro-forces that make
        the welded assembly CREEP across the table (~2 cm over a few
        hundred idle steps), drifting it out of the seat gates.
        """
        key = frozenset({body_a, body_b})
        if key in self._weld_constraints:
            return
        pos_a, orn_a = p.getBasePositionAndOrientation(
            body_a, physicsClientId=self._physics_client_id)
        pos_b, orn_b = p.getBasePositionAndOrientation(
            body_b, physicsClientId=self._physics_client_id)
        inv_pos, inv_orn = p.invertTransform(pos_a, orn_a)
        rel_pos, rel_orn = p.multiplyTransforms(inv_pos, inv_orn, pos_b, orn_b)
        # Snap: yaw-only relative orientation; ideal vertical offset.
        _, _, rel_yaw = p.getEulerFromQuaternion(rel_orn)
        rel_orn = p.getQuaternionFromEuler([0.0, 0.0, rel_yaw])
        if ideal_dz is not None:
            rel_pos = (rel_pos[0], rel_pos[1], ideal_dz)
        cid = p.createConstraint(parentBodyUniqueId=body_a,
                                 parentLinkIndex=-1,
                                 childBodyUniqueId=body_b,
                                 childLinkIndex=-1,
                                 jointType=p.JOINT_FIXED,
                                 jointAxis=[0, 0, 0],
                                 parentFramePosition=rel_pos,
                                 parentFrameOrientation=rel_orn,
                                 childFramePosition=[0, 0, 0],
                                 childFrameOrientation=[0, 0, 0, 1],
                                 physicsClientId=self._physics_client_id)
        # The default maxForce sags under a cantilevered span.
        p.changeConstraint(cid,
                           maxForce=self.weld_max_force,
                           physicsClientId=self._physics_client_id)
        self._weld_constraints[key] = cid

    def _desired_weld_pairs(
            self,
            state: State) -> Dict[FrozenSet[int], Tuple[int, int, float]]:
        """Weld pairs implied by the attachment features, as key ->
        (parent_body, child_body, ideal_dz).

        The joint's nominal vertical offset follows from the attachment
        slot: a ``top`` slot means the partner rests on this block,
        ``bottom`` the reverse, end slots are coplanar.
        """
        pairs: Dict[FrozenSet[int], Tuple[int, int, float]] = {}
        for blk in state.get_objects(self._block_type):
            for slot in ATTACH_SLOTS:
                idx = int(round(self._attr(blk, f"attached_{slot}", -1.0)))
                if idx < 0:
                    continue
                partner = self._blocks[idx]
                if blk.id is None or partner.id is None:
                    continue
                key = frozenset({blk.id, partner.id})
                if key in pairs:
                    continue
                if slot == "top":
                    dz = self._block_half_extents(blk)[2] + \
                        self._block_half_extents(partner)[2]
                elif slot == "bottom":
                    dz = -(self._block_half_extents(blk)[2] +
                           self._block_half_extents(partner)[2])
                else:
                    dz = 0.0
                pairs[key] = (blk.id, partner.id, dz)
        return pairs

    def _sync_welds_to_state(self, state: State) -> None:
        """Make the live constraint set match the attachment features.

        Object poses have already been restored by _set_state, so
        missing welds are created at the restored relative poses.
        Persisting welds keep their original constraint (the restored
        poses satisfy it by construction).
        """
        desired = self._desired_weld_pairs(state)
        for key in list(self._weld_constraints):
            if key not in desired:
                p.removeConstraint(self._weld_constraints[key],
                                   physicsClientId=self._physics_client_id)
                del self._weld_constraints[key]
        for key, (body_a, body_b, ideal_dz) in desired.items():
            if key not in self._weld_constraints:
                self._create_weld(body_a, body_b, ideal_dz=ideal_dz)

    def get_welded_partner_ids(self, body_id: int) -> Set[int]:
        """All body ids rigidly welded (transitively) to ``body_id``.

        Consumed by the skill-factory motion planner to exclude welded
        partners of the held object from the collision set.
        """
        partners: Set[int] = set()
        frontier = [body_id]
        while frontier:
            current = frontier.pop()
            for key in self._weld_constraints:
                if current in key:
                    (other, ) = key - {current}
                    if other != body_id and other not in partners:
                        partners.add(other)
                        frontier.append(other)
        return partners

    # -------------------------------------------------------------------------
    # Domain dynamics
    # -------------------------------------------------------------------------
    def _domain_specific_step(self) -> None:
        """Advance the glue process one step: wet faces near the held bottle's
        tip, tick cure counters on wet aligned joints, latch + weld at
        threshold, and refresh the patch visuals.

        NOTE: no prev-step handshake -- effects apply the moment the
        gate holds (a one-step delay makes the first step after a state
        jump a no-op, tripping the option model's repeat-state check).
        """
        state = self._get_state()
        blocks = state.get_objects(self._block_type)

        # 1. Glue application: wet the single nearest in-range face.
        if state.get(self._bottle, "is_held") > 0.5:
            tip = (state.get(self._bottle, "x"), state.get(self._bottle, "y"),
                   state.get(self._bottle, "z") - self.bottle_half_extents[2])
            best: Optional[Tuple[Object, str]] = None
            best_dist = self.apply_glue_radius
            for blk in blocks:
                for face in GLUE_FACES:
                    if self._attr(blk, f"glue_{face}", 0.0) > 0.5:
                        continue
                    if self._attr(blk, f"attached_{face}", -1.0) >= 0:
                        continue
                    dab = self._face_dab_point(state, blk, face)
                    dist = float(np.linalg.norm(np.array(tip) - np.array(dab)))
                    if dist < best_dist:
                        best = (blk, face)
                        best_dist = dist
            if best is not None:
                blk, face = best
                setattr(blk, f"glue_{face}", 1.0)

        # 2. Curing: wet faces in aligned resting contact tick; at the
        #    threshold the joint latches irreversibly and welds.
        for blk in blocks:
            for face in GLUE_FACES:
                if self._attr(blk, f"glue_{face}", 0.0) <= 0.5:
                    continue
                if self._attr(blk, f"attached_{face}", -1.0) >= 0:
                    continue
                mate = self._find_mate(state, blk, face)
                if mate is None:
                    setattr(blk, f"cure_{face}", 0.0)
                    continue
                cure = self._attr(blk, f"cure_{face}", 0.0) + 1.0
                setattr(blk, f"cure_{face}", cure)
                if cure >= self.cure_threshold:
                    self._latch_joint(state, blk, face, mate)

        # 3. Visuals.
        self._update_glue_patches(state)

    def _find_mate(self, state: State, blk: Object,
                   face: str) -> Optional[Object]:
        """The unique block currently in aligned resting contact with ``blk``'s
        ``face``, or None.

        Geometry mirrors the OnBlock / SeatedOn / NextToEnd classifiers.
        """
        for other in state.get_objects(self._block_type):
            if other == blk:
                continue
            if face == "top":
                if self._OnBlock_holds(state, [other, blk]) or \
                        self._SeatedOn_holds(state, [other, blk]):
                    return other
            else:
                if self._end_adjacent(state, blk, face, other):
                    return other
        return None

    def _end_adjacent(self, state: State, blk: Object, face: str,
                      other: Object) -> bool:
        """``other`` butts against ``blk``'s end face (neither held)."""
        if self._Holding_holds(state, [self._robot, blk]) or \
                self._Holding_holds(state, [self._robot, other]):
            return False
        dx_dir, dy_dir, _ = self._face_world_dir(state, blk, face)
        dx = state.get(other, "x") - state.get(blk, "x")
        dy = state.get(other, "y") - state.get(blk, "y")
        dz = state.get(other, "z") - state.get(blk, "z")
        proj = dx * dx_dir + dy * dy_dir
        perp = abs(-dx * dy_dir + dy * dx_dir)
        # Extent of each block along the joint direction (horizontal, so
        # a leg contributes its cross-section half width).
        ext = self._block_half_extents(blk)[0] + \
            self._block_half_extents(other)[0]
        if not ext - self.lateral_proj_tol_lo <= proj <= \
                ext + self.lateral_proj_tol_hi:
            return False
        return perp < self.lateral_perp_tol and \
            abs(dz) < self.lateral_z_tol

    def _latch_joint(self, state: State, blk: Object, face: str,
                     mate: Object) -> None:
        """Irreversibly attach ``blk.face`` to ``mate``: record the partnership
        on both blocks, consume the glue, create the weld."""
        if face == "top":
            mate_slot = "bottom"
        else:
            # The mate's end face that points back toward blk.
            dx_dir, dy_dir, _ = self._face_world_dir(state, blk, face)
            m_yaw = state.get(mate, "rot")
            # mate's end_b direction:
            mbx, mby = float(np.cos(m_yaw)), float(np.sin(m_yaw))
            # It faces blk if it opposes blk's outward face direction.
            mate_slot = "end_b" if mbx * dx_dir + mby * dy_dir < 0 \
                else "end_a"
        if self._attr(mate, f"attached_{mate_slot}", -1.0) >= 0:
            # The mate's slot is somehow taken; refuse to latch rather
            # than corrupt the attachment graph (cure stays at the
            # threshold, so this re-checks every step).
            return
        setattr(blk, f"attached_{face}", float(self._block_index[mate.name]))
        setattr(mate, f"attached_{mate_slot}",
                float(self._block_index[blk.name]))
        setattr(blk, f"glue_{face}", 0.0)
        assert blk.id is not None and mate.id is not None
        if face == "top":
            # The mate rests on blk's top face.
            ideal_dz = self._block_half_extents(blk)[2] + \
                self._block_half_extents(mate)[2]
        else:
            ideal_dz = 0.0
        self._create_weld(blk.id, mate.id, ideal_dz=ideal_dz)

    def _update_glue_patches(self, state: State) -> None:
        """Show a yellow patch on each wet face; park all other patches out of
        view.

        Patches are visual-only bodies.
        """
        oov_x, oov_y = self._out_of_view_xy
        in_state = set(state.get_objects(self._block_type))
        for i, blk in enumerate(self._blocks):
            for j, face in enumerate(GLUE_FACES):
                patch_id = self._glue_patch_ids[blk.name][face]
                wet = blk in in_state and \
                    self._attr(blk, f"glue_{face}", 0.0) > 0.5
                if not wet:
                    update_object(patch_id,
                                  position=(oov_x + 0.3 * i, oov_y + 0.3 * j,
                                            -1.0),
                                  physics_client_id=self._physics_client_id)
                    continue
                x = state.get(blk, "x")
                y = state.get(blk, "y")
                z = state.get(blk, "z")
                yaw = state.get(blk, "rot")
                hx, _, hz = self._block_half_extents(blk)
                dx_dir, dy_dir, dz_dir = self._face_world_dir(state, blk, face)
                offset = hz + 0.0015 if face == "top" else hx + 0.0015
                pos = (x + dx_dir * offset, y + dy_dir * offset,
                       z + dz_dir * offset)
                update_object(patch_id,
                              position=pos,
                              orientation=p.getQuaternionFromEuler(
                                  [0.0, 0.0, yaw]),
                              physics_client_id=self._physics_client_id)

    # -------------------------------------------------------------------------
    # Predicates
    # -------------------------------------------------------------------------
    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") > 0.02

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        _, blk = objects
        return state.get(blk, "is_held") > 0.5

    @staticmethod
    def _HoldingBottle_holds(state: State, objects: Sequence[Object]) -> bool:
        _, bottle = objects
        return state.get(bottle, "is_held") > 0.5

    def _make_glue_holds(self, face: str) -> Any:

        def _holds(state: State, objects: Sequence[Object]) -> bool:
            blk, = objects
            return state.get(blk, f"glue_{face}") > 0.5

        return _holds

    def _OnBlock_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """A leg block rests stacked on another leg block."""
        top, bottom = objects
        if top == bottom:
            return False
        if state.get(top, "upright") <= 0.5 or \
                state.get(bottom, "upright") <= 0.5:
            return False
        if self._Holding_holds(state, [self._robot, top]) or \
                self._Holding_holds(state, [self._robot, bottom]):
            return False
        dx = state.get(top, "x") - state.get(bottom, "x")
        dy = state.get(top, "y") - state.get(bottom, "y")
        if np.hypot(dx, dy) >= self.stack_align_tol:
            return False
        dz = state.get(
            top, "z") - (state.get(bottom, "z") + 2 * self.leg_half_extents[2])
        return bool(abs(dz) < self.stack_z_tol)

    def _NextToEnd_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        """True when ``right`` butts against ``left``'s end_b face.

        The row grows in the +local-x direction of the left block.
        """
        right, left = objects
        if right == left:
            return False
        if state.get(right, "upright") > 0.5 or \
                state.get(left, "upright") > 0.5:
            return False
        return self._end_adjacent(state, left, "end_b", right)

    def _SeatedOn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The span block rests on the leg block's top, with the leg under the
        span's footprint."""
        span, leg = objects
        if span == leg:
            return False
        if state.get(span, "upright") > 0.5 or \
                state.get(leg, "upright") <= 0.5:
            return False
        if self._Holding_holds(state, [self._robot, span]) or \
                self._Holding_holds(state, [self._robot, leg]):
            return False
        if abs(state.get(span, "y") - state.get(leg, "y")) >= \
                self.seat_y_tol:
            return False
        if abs(state.get(span, "x") - state.get(leg, "x")) >= \
                self.seat_x_window:
            return False
        dz = state.get(span, "z") - (state.get(
            leg, "z") + self.leg_half_extents[2] + self.span_half_extents[2])
        return bool(abs(dz) < self.seat_z_tol)

    def _AtSite_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The block stands on the table at the site (a stacked upper leg is
        NOT at the site -- the z check pins the base block)."""
        blk, site = objects
        if state.get(blk, "upright") <= 0.5:
            return False
        if self._Holding_holds(state, [self._robot, blk]):
            return False
        dist = np.hypot(
            state.get(blk, "x") - state.get(site, "x"),
            state.get(blk, "y") - state.get(site, "y"))
        if dist >= self.at_site_tol:
            return False
        dz = state.get(blk,
                       "z") - (self.table_height + self.leg_half_extents[2])
        return bool(abs(dz) < self.at_site_z_tol)

    def _SiteFree_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (site, ) = objects
        for blk in state.get_objects(self._block_type):
            if self._AtSite_holds(state, [blk, site]):
                return False
        return True

    def _Attached_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The two blocks share a cured glue joint (symmetric)."""
        a, b = objects
        if a == b:
            return False
        idx_a = self._block_index[a.name]
        idx_b = self._block_index[b.name]
        for slot in ATTACH_SLOTS:
            if int(round(state.get(a, f"attached_{slot}"))) == idx_b:
                return True
            if int(round(state.get(b, f"attached_{slot}"))) == idx_a:
                return True
        return False

    def _TopFree_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Nothing rests on the block's top face (so the face can be reached by
        the glue bottle's dab)."""
        (blk, ) = objects
        for other in state.get_objects(self._block_type):
            if other == blk:
                continue
            if self._OnBlock_holds(state, [other, blk]) or \
                    self._SeatedOn_holds(state, [other, blk]):
                return False
        return True

    def _Loose_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """The block has no cured attachments (it can be individually picked
        and re-placed without dragging an assembly along)."""
        (blk, ) = objects
        return all(
            int(round(state.get(blk, f"attached_{slot}"))) < 0
            for slot in ATTACH_SLOTS)

    def _Connected_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        """Transitive closure of Attached (same rigid assembly)."""
        a, b = objects
        if a == b:
            return False
        blocks = state.get_objects(self._block_type)
        frontier = [a]
        seen = {a}
        while frontier:
            current = frontier.pop()
            for other in blocks:
                if other in seen:
                    continue
                if self._Attached_holds(state, [current, other]):
                    if other == b:
                        return True
                    seen.add(other)
                    frontier.append(other)
        return False

    # -------------------------------------------------------------------------
    # Task Generation
    # -------------------------------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_train_tasks,
                                specs=CFG.bridge_task_spec_train,
                                rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._make_tasks(num_tasks=CFG.num_test_tasks,
                                specs=CFG.bridge_task_spec_test,
                                rng=self._test_rng)

    def _make_tasks(self, num_tasks: int, specs: List[str],
                    rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            spec = str(rng.choice(specs))
            assert spec in ("simple", "full"), f"Unknown bridge spec {spec}"
            if spec == "simple":
                legs = self._legs[:2]
                spans = self._spans[:2]
                site_sep = self.site_sep_simple
            else:
                legs = self._legs[:4]
                spans = self._spans[:3]
                site_sep = self.site_sep_full

            init_dict: Dict[Object, Dict[str, float]] = {}
            init_dict[self._robot] = {
                "x": self.robot_init_x,
                "y": self.robot_init_y,
                "z": self.robot_init_z,
                "fingers": self.open_fingers,
                "roll": self.robot_init_roll,
                "tilt": self.robot_init_tilt,
                "wrist": self.robot_init_wrist,
            }

            # Sites: a front band, x jittered for diversity.
            mid_x = float(
                rng.uniform(self.x_mid - self.site_x_jitter,
                            self.x_mid + self.site_x_jitter))
            site_xs = (mid_x - site_sep / 2, mid_x + site_sep / 2)
            for site, sx in zip(self._sites, site_xs):
                init_dict[site] = {
                    "x": sx,
                    "y": self.site_y,
                    "z": self.table_height,
                }

            # Staging: assign objects to jittered grid slots; the span
            # row is assembled in place at span0's slot, growing in +x
            # along the middle row (see _stage_objects).
            stage_xy = self._stage_objects(rng, legs, spans, site_xs)

            # Block init features.
            for blk in legs + spans:
                is_leg = self._is_leg_shaped(blk)
                bx, by = stage_xy[blk]
                color_idx = int(rng.integers(len(self._obj_colors_main)))
                r_col, g_col, b_col, _ = self._obj_colors_main[color_idx]
                feats: Dict[str, float] = {
                    "x":
                    bx,
                    "y":
                    by,
                    "z":
                    self.table_height + (self.leg_half_extents[2] if is_leg
                                         else self.span_half_extents[2]),
                    "rot":
                    0.0,
                    "is_held":
                    0.0,
                    "upright":
                    1.0 if is_leg else 0.0,
                    "r":
                    r_col,
                    "g":
                    g_col,
                    "b":
                    b_col,
                }
                for face in GLUE_FACES:
                    feats[f"glue_{face}"] = 0.0
                    if f"cure_{face}" in self._block_type.feature_names:
                        feats[f"cure_{face}"] = 0.0
                for slot in ATTACH_SLOTS:
                    feats[f"attached_{slot}"] = -1.0
                init_dict[blk] = feats

            bx, by = stage_xy[self._bottle]
            init_dict[self._bottle] = {
                "x": bx,
                "y": by,
                "z": self.table_height + self.bottle_half_extents[2],
                "rot": 0.0,
                "is_held": 0.0,
            }

            init_state = utils.create_state_from_dict(init_dict)
            if CFG.partially_observable:
                init_state.privileged = {
                    blk.name: {f"cure_{face}": 0.0
                               for face in GLUE_FACES}
                    for blk in legs + spans
                }

            # Goal: the n-bridge standing at the two sites. Roles are
            # task-assigned by block name, and the goal pins the full
            # geometric layout (AtSite / OnBlock / NextToEnd / SeatedOn
            # in addition to Attached): every atom persists in the
            # finished bridge, and the pinning forces the planner's
            # bindings to the physically consistent left-to-right build
            # (an unpinned symmetric-Attached goal admits abstract plans
            # that assemble the row in the wrong direction and then
            # cannot seat it).
            if spec == "simple":
                base_left, base_right = legs[0], legs[1]
                seat_left, seat_right = legs[0], legs[1]
            else:
                base_left, base_right = legs[0], legs[2]
                seat_left, seat_right = legs[1], legs[3]
            goal_atoms = {
                GroundAtom(self._AtSite, [base_left, self._sites[0]]),
                GroundAtom(self._AtSite, [base_right, self._sites[1]]),
                GroundAtom(self._SeatedOn, [spans[0], seat_left]),
                GroundAtom(self._SeatedOn, [spans[-1], seat_right]),
                GroundAtom(self._Attached, [seat_left, spans[0]]),
                GroundAtom(self._Attached, [seat_right, spans[-1]]),
            }
            for left_span, right_span in zip(spans, spans[1:]):
                goal_atoms.add(
                    GroundAtom(self._NextToEnd, [right_span, left_span]))
                goal_atoms.add(
                    GroundAtom(self._Attached, [left_span, right_span]))
            if spec == "full":
                goal_atoms.add(GroundAtom(self._OnBlock, [legs[1], legs[0]]))
                goal_atoms.add(GroundAtom(self._OnBlock, [legs[3], legs[2]]))
                goal_atoms.add(GroundAtom(self._Attached, [legs[0], legs[1]]))
                goal_atoms.add(GroundAtom(self._Attached, [legs[2], legs[3]]))
            goal_nl = (
                "Build an n-shaped bridge standing at the two marked "
                "sites: stand a leg at each site" +
                (" (each leg is two blocks glued into a stack)"
                 if spec == "full" else "") +
                ", glue the span blocks end-to-end on the table into one "
                "assembly, apply glue to the leg tops, wait for joints "
                "to cure, then seat the cured span across the legs. Use "
                "the glue bottle to wet a face before mating it.")

            tasks.append(
                EnvironmentTask(init_state, goal_atoms, goal_nl=goal_nl))

        return self._add_pybullet_state_to_tasks(tasks)

    def _stage_objects(
            self, rng: np.random.Generator, legs: List[Object],
            spans: List[Object],
            site_xs: Tuple[float, float]) -> Dict[Object, Tuple[float, float]]:
        """Assign every staged object (blocks + bottle) a jittered grid slot.

        span0 takes the leftmost feasible middle-row slot and the strip
        [span0_x, span0_x + row_len] on that row is reserved for
        assembling the span row; everything else fills the remaining
        slots in random order.
        """
        base_x, base_y = self.robot_base_pos[0], self.robot_base_pos[1]
        row_len = (len(spans) - 1) * (2 * self.span_half_extents[0] +
                                      self.lateral_place_gap)

        def _reachable(x: float, y: float) -> bool:
            # Radial reach cap (empirical fetch validated-IK frontier,
            # see pybullet_bond), with margin for the slot jitter.
            return np.hypot(x - base_x, y - base_y) <= \
                self.reach_radius - 1.5 * self.stage_jitter

        # Candidate slots. Front-row slots must clear both sites.
        slots: List[Tuple[float, float]] = []
        for col in self.stage_cols:
            for row in (self.stage_row_mid, self.stage_row_back):
                if _reachable(col, row):
                    slots.append((col, row))
            if _reachable(col, self.stage_row_front) and all(
                    abs(col - sx) > self.site_keepout for sx in site_xs):
                slots.append((col, self.stage_row_front))

        # span0 + assembly strip on the middle row.
        strip_starts = [
            col for col in self.stage_cols
            if _reachable(col + row_len, self.stage_row_mid) and (
                col, self.stage_row_mid) in slots
        ]
        assert strip_starts, "no feasible span-row start"
        span0_col = float(rng.choice(strip_starts))
        span0_xy = (span0_col, self.stage_row_mid)
        # Reserve every middle-row slot the strip sweeps over.
        slots = [(cx, cy) for cx, cy in slots if not (
            cy == self.stage_row_mid and span0_col -
            self.site_keepout <= cx <= span0_col + row_len + self.site_keepout)
                 ]

        rest = spans[1:] + legs + [self._bottle]
        assert len(slots) >= len(rest), \
            f"only {len(slots)} staging slots for {len(rest)} objects"
        # Spans are 10 cm long (lying along x), so two spans in
        # same-row adjacent columns (0.11 apart) would overlap; legs
        # (5 cm) and the bottle are fine. Re-draw the assignment until
        # no two staged spans are same-row column-neighbors.
        span_rest = set(spans[1:])
        for _ in range(50):
            chosen = rng.choice(len(slots), size=len(rest), replace=False)
            # Include span0 at the strip start: a slot in the column
            # just left of it survives the strip reservation but is
            # still too close for a 10 cm neighbor.
            span_slots = [span0_xy] + [
                slots[int(si)]
                for obj, si in zip(rest, chosen) if obj in span_rest
            ]
            if all(
                    abs(ay - by) > 0.01 or abs(ax - bx) > 0.115
                    for i, (ax, ay) in enumerate(span_slots)
                    for bx, by in span_slots[i + 1:]):
                break
        stage_xy: Dict[Object, Tuple[float, float]] = {spans[0]: span0_xy}
        for obj, slot_i in zip(rest, chosen):
            stage_xy[obj] = slots[int(slot_i)]
        # Jitter everything (including span0).
        return {
            obj:
            (x + float(rng.uniform(-self.stage_jitter, self.stage_jitter)),
             y + float(rng.uniform(-self.stage_jitter, self.stage_jitter)))
            for obj, (x, y) in stage_xy.items()
        }


if __name__ == "__main__":

    def _main() -> None:
        """Quick manual visualization."""
        import time  # pylint: disable=import-outside-toplevel
        CFG.seed = 0
        CFG.env = "pybullet_bridge"
        CFG.num_train_tasks = 1
        env = PyBulletBridgeEnv(use_gui=True)
        task = env._generate_train_tasks()[0]  # pylint: disable=protected-access
        env._set_state(task.init)  # pylint: disable=protected-access
        while True:
            env.step(
                Action(np.array(env._pybullet_robot.initial_joint_positions)))  # pylint: disable=protected-access
            time.sleep(0.01)

    _main()
