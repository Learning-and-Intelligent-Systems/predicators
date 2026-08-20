"""Real-world domino envs: the ``pybullet_domino`` env retargeted to the real
Franka robot. These are **pure simulation** environments. They hold no robot,
ship no motion, and have no real/dry mode.

"Real" is two independent things, and this module keeps them separable:

  * the robot SETUP -- the Franka standing on its short pedestal, the
    extended table tile (:class:`RealSceneGeometryMixin`);
  * the TASKS and the blocks -- a single task rebuilt from a perceived
    scene, with the real dominoes (:class:`PyBulletDominoRealEnv`).

``pybullet_domino_real`` takes both. ``pybullet_domino_real_geometry``
(:class:`PyBulletDominoRealGeometryEnv`) takes only the first.

Driving an arm with either is the real-robot executor's job
(``predicators/pybullet_helpers/real_robot_executor.py``), which attaches to
the env and calls the conversions below. Keeping the two apart is what lets
these envs be tested without hardware and the executor without PyBullet.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.env import DominoEvaluator, \
    PyBulletDominoComposedEnv, PyBulletDominoEnv
from predicators.envs.pybullet_domino.real_geometry import Pose6D, \
    domino_env_euler, domino_world_z_offset, pose_base_to_world
from predicators.envs.pybullet_domino.task_generators import goal_text
from predicators.pybullet_helpers.objects import create_object, \
    create_pybullet_block
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, State, \
    TaskEvaluator


def _base_pose(xyz: Sequence[float], quat_xyzw: Sequence[float]) -> Pose6D:
    """Base-frame ``Pose6D`` from loose sequences.

    Unpacking is the length check: a capture record or observation with
    the wrong number of components fails here rather than silently
    producing a malformed pose.
    """
    x, y, z = (float(v) for v in xyz)
    qx, qy, qz, qw = (float(v) for v in quat_xyzw)
    return Pose6D((x, y, z), (qx, qy, qz, qw))


def _scoring_nl() -> str:
    """The reward structure, in the words the generator already uses.

    Kept identical to ``DominoTaskGenerator._make_task`` for the reason
    recorded there: without it an agent reads a rejected goal-reaching
    attempt as a fatal per-blue penalty rather than a missing solve
    bonus, and spends its budget theorising about that instead.
    """
    cost = CFG.domino_block_cost
    return ((f" Scoring: a solve earns +1 reward, and each blue "
             f"domino the cascade consumes (toppled or shoved "
             f"out of place) costs {cost:g}, so a solve that "
             f"uses one blue scores +{1.0 - cost:g}. Using "
             f"blues never disqualifies a solve.") +
            goal_text.CASCADE_VERIFICATION_NL)


def _canonical_roll(roll: float) -> float:
    """Fold a perceived roll into ``[-pi/2, pi/2)``.

    A domino is a box, so turning it 180 degrees about its own width
    axis leaves it exactly where it was. Both orientations are equally
    correct descriptions of the same physical domino, and a marker-based
    pose estimate returns one or the other arbitrarily -- in a single
    capture, some dominoes come back at roll 0 and others at roll +-pi.

    Roll is therefore only meaningful modulo pi, and folding it is not
    cosmetic: ``Toppled`` is defined as ``|roll| >= 10 degrees``, so an
    unfolded ``roll = pi`` makes an upright domino read as knocked over
    before anything has moved. A task whose goal is ``Toppled(target)``
    is then already satisfied in its own initial state, and the planner
    returns an empty plan and reports success.

    Standing (0 or +-pi) folds to ~0. Knocked over (+-pi/2) keeps its
    magnitude, which is all ``Toppled`` and ``Upright`` read.
    """
    return (roll + math.pi / 2) % math.pi - math.pi / 2


@dataclass(frozen=True)
class _PerceivedDomino:
    """One domino as perceived, normalized from either source.

    A scene-JSON record and a live ``DominoObservation`` entry carry the
    same content under different field names, so both are converted to
    this before anything else happens. ``slot`` is the index into the
    env's domino component; ``pose_base`` is in the robot base frame.
    """
    slot: int
    capture_id: int
    pose_base: Pose6D
    role: str


class RealSceneGeometryMixin:
    """Stages a ``pybullet_domino`` env on the real scene's ROBOT geometry.

    This is the physical setup only -- where the arm stands and what it
    stands on. Mixing it
    into a domino env raises the robot base onto the real scene's short
    pedestal, homes the arm at the real tilt/wrist, and spawns the
    pedestal plus the extended table tile the real scene has.

    Split out from :class:`PyBulletDominoRealEnv` so the two axes compose
    independently: that env pairs this geometry with tasks rebuilt from a
    perceived scene, while :class:`PyBulletDominoRealGeometryEnv` pairs it
    with the ordinary generated tasks.

    Deliberately NOT a ``BaseEnv`` subclass. ``create_new_env`` resolves an
    env by scanning ``get_all_subclasses(BaseEnv)`` for a matching
    ``get_name()``, so an intermediate env class here would inherit
    ``PyBulletDominoEnv.get_name()`` and shadow the real
    ``pybullet_domino``. A plain mixin never enters that scan.
    """

    # Annotations only, no values: these ClassVars belong to the domino env
    # this mixin is combined with. Declaring them tells the type checker what
    # _apply_real_geometry configures without giving the mixin its own copies
    # (which would shadow the env's) and without inheriting BaseEnv.
    robot_base_pos: ClassVar[Optional[Tuple[float, float, float]]]
    robot_init_tilt: ClassVar[float]
    robot_init_wrist: ClassVar[float]

    @classmethod
    def _apply_real_geometry(cls) -> None:
        """Set THIS subclass's robot geometry ClassVars from CFG (raise the
        base onto the real scene's pedestal).

        Applied in ``initialize_pybullet`` so it takes effect on BOTH
        the normal env build AND the skill factory's direct
        ``initialize_pybullet`` call (which bypasses ``__init__``).
        Reads the base xy from the untouched shared class, so it is
        idempotent. Only this subclass is configured -- the shared base
        is never mutated.

        The home EE height is not set here: the Panda homes to its own
        configuration, which is reachable by construction and identical in
        every env (see PyBulletEnv._sync_robot_init_pos_with_home). It used
        to be lowered by hand here to keep the Fetch-tuned home within the
        Panda's reach.
        """
        z_off = domino_world_z_offset(CFG.domino_real_table_z)
        base = PyBulletDominoComposedEnv.robot_base_pos
        assert base is not None
        base_xy = base[:2]
        cls.robot_base_pos = (base_xy[0], base_xy[1], float(z_off))
        cls.robot_init_tilt = float(CFG.domino_real_robot_init_tilt)
        cls.robot_init_wrist = float(CFG.domino_real_robot_init_wrist)

    @classmethod
    def initialize_pybullet(cls, using_gui: bool) -> Tuple[Any, Any, Any]:
        """Apply the real-scene geometry, build the world, then decorate this
        instance's sim (extended-table tile + robot pedestal).

        Every pipeline env is an instance of this class, so each
        configures + decorates itself.
        """
        cls._apply_real_geometry()
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)  # type: ignore[misc]
        if CFG.domino_real_decorate:
            cls._decorate(physics_client_id,
                          domino_world_z_offset(CFG.domino_real_table_z))
        return physics_client_id, pybullet_robot, bodies

    @classmethod
    def _decorate(cls, pcid: int, z_off: float) -> None:
        """Add the extended-table tile + robot pedestal (ported from
        ``birrt._decorate_scene``) with predicators' own body helpers."""

        def yq(yaw: float) -> Tuple[float, float, float, float]:
            return tuple(p.getQuaternionFromEuler([0.0, 0.0, yaw]))

        # Extra table tile toward the robot (world y=0.85), table top z=0.4.
        tile_id = create_object("urdf/table.urdf",
                                position=(0.75, 0.85, 0.2),
                                orientation=yq(np.pi / 2),
                                scale=1.0,
                                use_fixed_base=True,
                                physics_client_id=pcid)
        # Match the env's studio wood texture if this env uses studio visuals.
        tex_path = getattr(cls, "table_texture_path", None)
        if getattr(cls, "_use_studio_visuals", False) and tex_path and \
                isinstance(tile_id, int):
            texid = p.loadTexture(utils.get_env_asset_path(tex_path),
                                  physicsClientId=pcid)
            p.changeVisualShape(tile_id,
                                -1,
                                textureUniqueId=texid,
                                rgbaColor=(1, 1, 1, 1),
                                physicsClientId=pcid)
        # Robot mount pedestal: fill the table top (0.4) up to the base (z_off).
        riser_h = z_off - 0.4
        if riser_h > 1e-3:
            create_pybullet_block(color=(0.3, 0.3, 0.3, 1.0),
                                  half_extents=(0.10, 0.10, riser_h / 2),
                                  mass=0.0,
                                  friction=0.5,
                                  position=(0.75, 0.72, 0.4 + riser_h / 2),
                                  orientation=yq(0.0),
                                  physics_client_id=pcid)


class PyBulletDominoRealGeometryEnv(RealSceneGeometryMixin, PyBulletDominoEnv):
    """``pybullet_domino``'s OWN generated tasks, staged on the real scene's
    robot geometry.

    The point of separating this from :class:`PyBulletDominoRealEnv` is
    which half of "real" you want. That env replaces the tasks: it rebuilds
    a single task from a perceived scene and sizes its domino component
    from that scene's roles, so the generated-task machinery never runs.
    This env keeps all of that machinery and changes only where the arm
    stands.

    Point ``pybullet_robot`` at ``panda`` in the
    env config to get the Franka; the pedestal and table tile come from
    the mixin.

    The DOMINOES are deliberately the simulated ones -- same dimensions,
    same mass, built by the inherited ``_make_domino_component``. Only the
    robot and the table change.
    """

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_real_geometry"


class PyBulletDominoRealEnv(RealSceneGeometryMixin, PyBulletDominoEnv):
    """``pybullet_domino`` on the real scene: the geometry above, plus a single
    task sized and built from a perceived scene JSON."""

    def __init__(self, use_gui: bool = False, **kwargs: Any) -> None:
        self._z_off = domino_world_z_offset(CFG.domino_real_table_z)
        # Dominoes are placed in scene order, so slot i <-> capture id
        # self._scene_ids[i]; that mapping is what lets a live observation,
        # which carries capture ids and nothing else, name component slots.
        with open(CFG.domino_real_scene, encoding="utf-8") as f:
            self._scene_ids = [int(d["id"]) for d in json.load(f)["dominoes"]]
        super().__init__(use_gui=use_gui, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_domino_real"

    # -- roles --------------------------------------------------------------
    @staticmethod
    def _role_for_capture_id(capture_id: int) -> str:
        """Role keyed by capture id alone.

        Raw capture JSONs and live observations both carry ids but no
        roles, so ``CFG.domino_real_{start,target}_id`` names the green
        start and the purple target; everything else is a movable blue.
        """
        if capture_id == CFG.domino_real_start_id:
            return "start"
        if capture_id == CFG.domino_real_target_id:
            return "target"
        return "movable"

    @classmethod
    def _domino_role(cls, d: Dict[str, Any]) -> str:
        """Role ('start' / 'target' / 'movable') for a scene domino.

        Prefers an explicit ``role`` field if the scene carries one,
        otherwise falls back to the id keying above.
        """
        if "role" in d:
            return str(d["role"])
        return cls._role_for_capture_id(int(d["id"]))

    # -- component sizing + dims --------------------------------------------
    @classmethod
    def _scene_role_counts(cls) -> Tuple[int, int]:
        """(num_target, num_nontarget) domino counts from the scene JSON."""
        with open(CFG.domino_real_scene, encoding="utf-8") as f:
            roles = [cls._domino_role(d) for d in json.load(f)["dominoes"]]
        n_target = sum(1 for r in roles if r == "target")
        return n_target, len(roles) - n_target

    @classmethod
    def _make_domino_component(
            cls, workspace_bounds: Dict[str, float]) -> DominoComponent:
        """Allocate the scene's counts and the real perceived dimensions,
        passing dims through the component ctor (not a base ClassVar mutation).

        ``domino_real_domino_dims`` is (L, W, H): a standing domino has
        body-x (L) vertical, so env height=L, width=W (broad face),
        depth=H (thickness).
        """
        n_target, n_nontarget = cls._scene_role_counts()
        length, width, thickness = (float(v)
                                    for v in CFG.domino_real_domino_dims)
        return DominoComponent(num_dominos_max=n_nontarget,
                               num_targets_max=n_target,
                               num_pivots_max=0,
                               workspace_bounds=workspace_bounds,
                               domino_width=width,
                               domino_depth=thickness,
                               domino_height=length)

    # -- task generation ----------------------------------------------------
    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return [self._build_task_from_scene()]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return [self._build_task_from_scene()]

    # -- perception -> State / Task -----------------------------------------
    # ONE conversion, three callers: the captured scene JSON, a live
    # observation building a fresh task, and a live observation correcting an
    # existing state. Both sources are first normalized to _PerceivedDomino.
    #
    # These are pure conversions: no hardware, no I/O beyond reading the scene
    # file, and no notion of a robot. The real-world wrapper calls them.

    def _slot_for_capture_id(self, capture_id: int) -> Optional[int]:
        """Component slot holding the domino with this capture id.

        Dominoes are placed in scene order, so slot i holds capture id
        ``self._scene_ids[i]``. An id the scene never had is dropped.
        """
        try:
            return self._scene_ids.index(int(capture_id))
        except ValueError:
            logging.warning(
                "pybullet_domino_real: ignoring domino id %s, which is not "
                "in the scene %s", capture_id, self._scene_ids)
            return None

    def _perceived_from_scene(
            self, records: List[Dict[str, Any]]) -> List[_PerceivedDomino]:
        """Normalize scene-JSON records."""
        perceived = []
        for d in records:
            slot = self._slot_for_capture_id(int(d["id"]))
            if slot is None:
                continue
            pose = _base_pose(d["center_base_m"], d["quat_base_xyzw"])
            perceived.append(
                _PerceivedDomino(slot=slot,
                                 capture_id=int(d["id"]),
                                 pose_base=pose,
                                 role=self._domino_role(d)))
        return perceived

    def _perceived_from_observation(self, obs: Any) -> List[_PerceivedDomino]:
        """Normalize a ``DominoObservation`` captured from the real scene.

        Duck-typed deliberately: this reads ``obs.dominoes`` and each
        entry's ``id`` / ``xyz`` / ``quat_xyzw``, so the env imports no
        babyrobot and the conversion stays testable against a plain
        stub.
        """
        perceived = []
        for d in obs.dominoes:
            slot = self._slot_for_capture_id(int(d.id))
            if slot is None:
                continue
            pose = _base_pose(d.xyz, d.quat_xyzw)
            perceived.append(
                _PerceivedDomino(slot=slot,
                                 capture_id=int(d.id),
                                 pose_base=pose,
                                 role=self._role_for_capture_id(int(d.id))))
        return perceived

    def _env_angles(self, world: Pose6D,
                    capture_id: int) -> Tuple[float, float]:
        """``(roll, yaw)`` for a perceived domino, standing or knocked over.

        A domino type carries ``yaw`` and ``roll`` and no ``pitch``, so a
        perceived orientation is representable exactly when its pitch is
        zero -- which covers standing dominoes and dominoes lying on
        either face, i.e. every pose a free domino reaches on a flat
        table. Anything else (propped diagonally on a neighbour, say)
        loses its pitch when the state is written into PyBullet, so say
        so rather than silently flattening it.

        The roll is folded modulo pi; see :func:`_canonical_roll`.
        """
        roll, pitch, yaw = domino_env_euler(world)
        if abs(pitch) >= DominoComponent.domino_roll_threshold:
            logging.warning(
                "pybullet_domino_real: domino %s is pitched %.1f deg, which "
                "the (yaw, roll) domino state cannot represent; dropping the "
                "pitch", capture_id, math.degrees(pitch))
        return _canonical_roll(roll), yaw

    @staticmethod
    def _canonical_start_yaw(
            yaw: float, world: Pose6D, push_dir_world: Optional[Tuple[float,
                                                                      float]],
            target_xy: Optional[Tuple[float, float]]) -> float:
        """Flip the START domino's yaw to face the direction it must topple.

        A domino is 180-degree symmetric, so perception's yaw branch is
        arbitrary; the single push has to go the intended way. The
        desired direction is an explicit ``start_push_dir_base`` when
        the scene gives one, else the default "toward the target".
        """
        pdir = push_dir_world
        if pdir is None and target_xy is not None:
            pdir = (target_xy[0] - world.xyz[0], target_xy[1] - world.xyz[1])
        if pdir is None:
            return yaw
        fx, fy = math.sin(yaw), math.cos(yaw)
        if fx * pdir[0] + fy * pdir[1] < 0.0:
            return math.atan2(-fx, -fy)  # flip 180 to face the push dir
        return yaw

    def _init_state_from_perceived(
            self,
            perceived: List[_PerceivedDomino],
            push_dir_base: Optional[Sequence[float]] = None) -> State:
        """Initial ``State`` for a task built from perceived dominoes.

        Places each domino at its transplanted world (x, y) with the
        upright heading, colored by role (green=start, purple=target,
        blue=movable) via the component's ``place_domino``.
        """
        comp = self._domino_component
        assert comp is not None, "env has no domino component"
        assert len(perceived) <= len(comp.dominos), \
            f"perceived {len(perceived)} dominoes but only " \
            f"{len(comp.dominos)} slots"

        worlds = {
            pd.slot: pose_base_to_world(pd.pose_base, self._z_off)
            for pd in perceived
        }
        target_xy = next(((worlds[pd.slot].xyz[0], worlds[pd.slot].xyz[1])
                          for pd in perceived if pd.role == "target"), None)

        # Optional per-scene override of the start domino's push direction,
        # given in the base frame as [dx, dy]; transplanted to world
        # (base->world is a +pi/2 z-rotation, so (dx, dy) -> (-dy, dx)).
        push_dir_world = None
        if push_dir_base is not None:
            push_dir_world = (-float(push_dir_base[1]),
                              float(push_dir_base[0]))

        init_dict: Dict[Any, Dict[str, float]] = {}
        # Robot: env home (bench geometry already applied to the ClassVars).
        init_dict[self._robot] = {
            "x": self.robot_init_x,
            "y": self.robot_init_y,
            "z": self.robot_init_z,
            "fingers": self.open_fingers,
            "roll": self.robot_init_roll,
            "tilt": self.robot_init_tilt,
            "wrist": self.robot_init_wrist,
        }

        for pd in perceived:
            world = worlds[pd.slot]
            roll, yaw = self._env_angles(world, pd.capture_id)
            # Canonicalize the START domino's heading only while it is still
            # standing. The flip picks which of two 180-degree-symmetric
            # headings faces the push; a domino already lying down has no push
            # to orient, and flipping it would misreport which way it fell.
            if pd.role == "start" and \
                    abs(roll) < DominoComponent.fallen_threshold:
                yaw = self._canonical_start_yaw(yaw, world, push_dir_world,
                                                target_xy)
            entry = comp.place_domino(pd.slot,
                                      world.xyz[0],
                                      world.xyz[1],
                                      yaw,
                                      is_start_block=(pd.role == "start"),
                                      is_target_block=(pd.role == "target"))
            # Perceived world (x, y, z) and orientation. Keep place_domino's
            # role color / is_held; override the pose it assumed.
            entry["x"], entry["y"], entry["z"] = world.xyz
            entry["yaw"] = yaw
            entry["roll"] = roll
            init_dict[comp.dominos[pd.slot]] = entry

        return utils.create_state_from_dict(init_dict)

    def _task_from_perceived(
            self,
            perceived: List[_PerceivedDomino],
            push_dir_base: Optional[Sequence[float]] = None
    ) -> EnvironmentTask:
        """Task (init state + goal) for a set of perceived dominoes."""
        init_state = self._init_state_from_perceived(perceived, push_dir_base)
        comp = self._domino_component
        assert comp is not None, "env has no domino component"

        # Goal: topple the purple (target) domino(s). With
        # domino_use_domino_blocks_as_target, Toppled is typed on domino_type
        # and _TargetDomino_holds identifies targets by color.
        goal_atoms = set()
        for dom in init_state.get_objects(comp.domino_type):
            if comp._TargetDomino_holds(  # pylint: disable=protected-access
                    init_state, [dom]):
                goal_atoms.add(GroundAtom(comp.Toppled, [dom]))
        assert len(goal_atoms) >= 1, "no purple target domino found"

        goal_nl = (
            "Move the blue dominoes such that when the green domino is pushed, "
            "the purple domino is toppled. Do NOT directly push or topple the "
            "purple domino yourself.")
        evaluator = self._evaluator_for(init_state, goal_atoms)
        if evaluator is not None:
            goal_nl += _scoring_nl()
        task = EnvironmentTask(init_state,
                               goal_atoms,
                               goal_nl=goal_nl,
                               evaluator=evaluator)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _evaluator_for(self, init_state: State,
                       goal_atoms: Set[GroundAtom]) -> Optional[TaskEvaluator]:
        """A ``DominoEvaluator`` for this scene, or None if it cannot judge it.

        Attached on the same terms the task generator uses for the
        simulated envs (``DominoTaskGenerator._make_task``), because the
        certificate's causal model is what decides whether a verdict
        means anything -- not which env built the scene. Without one
        every episode scores 0.0, so an over-built chain reads the same
        as a minimal one.

        The two conditions are the generator's, for its reasons: targets
        must be roll-tracked dominoes, or the certificate cannot see a
        direct robot knock on one; and dominoes must be the only dynamic
        component, or something else topples them without a Push and the
        certificate rejects a legitimate cascade. A real scene is
        dominoes and nothing else, so the second holds by construction.
        """
        comp = self._domino_component
        assert comp is not None, "env has no domino component"
        others = [c for c in self._components if c is not comp]
        if not CFG.domino_use_domino_blocks_as_target or others:
            return None
        num_movables = sum(1
                           for dom in init_state.get_objects(comp.domino_type)
                           if comp._MovableBlock_holds(init_state, [dom]))  # pylint: disable=protected-access
        # DominoEvaluator asserts this itself, but it would fire mid-episode
        # on the real robot; name the scene's own numbers instead.
        if CFG.domino_block_cost * num_movables >= 1.0:
            raise ValueError(
                f"this scene stages {num_movables} movable dominoes, which "
                f"at domino_block_cost={CFG.domino_block_cost} cost at least "
                "as much as a success is worth -- a legitimate cascade would "
                "not outscore failing. Lower domino_block_cost or stage "
                "fewer movable dominoes.")
        return DominoEvaluator(goal_atoms, num_movables)

    def _build_task_from_scene(self) -> EnvironmentTask:
        """Build the captured-scene task with attached pybullet state."""
        with open(CFG.domino_real_scene, encoding="utf-8") as f:
            scene = json.load(f)
        return self._task_from_perceived(
            self._perceived_from_scene(scene["dominoes"]),
            scene.get("start_push_dir_base"))

    def task_from_observation(self,
                              obs: Any,
                              train_or_test: str = "test") -> EnvironmentTask:
        """Build a task from a live observation of the real scene.

        Same conversion as the captured scene, plus goal semantics. This
        env's goal does not depend on ``train_or_test`` -- both generate
        the same "topple the purple domino" task -- but the argument is
        part of the hook the real-world wrapper calls, since another
        environment's might.

        A live observation carries no ``start_push_dir_base``, so the
        start domino's yaw is canonicalized toward the target.
        """
        del train_or_test  # same goal either way for this env
        return self._task_from_perceived(self._perceived_from_observation(obs))

    def _target_xy(self, perceived: List[_PerceivedDomino],
                   prev_state: State) -> Optional[Tuple[float, float]]:
        """``(x, y)`` of the target domino, for orienting the start's yaw.

        Prefers what this observation saw. Falls back to the twin's last
        known pose when the observation does not name the target, which
        is the same "absent means unchanged" policy the rest of the
        correction follows -- a target hidden behind the arm must not
        silently cost the start its heading.
        """
        for pd in perceived:
            if pd.role == "target":
                world = pose_base_to_world(pd.pose_base, self._z_off)
                return (world.xyz[0], world.xyz[1])
        comp = self._domino_component
        assert comp is not None, "env has no domino component"
        for slot, capture_id in enumerate(self._scene_ids):
            if self._role_for_capture_id(capture_id) == "target":
                dom = comp.dominos[slot]
                return (prev_state.get(dom, "x"), prev_state.get(dom, "y"))
        return None

    def state_from_observation(self, obs: Any, prev_state: State) -> State:
        """Correct ``prev_state`` with what the cameras just saw.

        Only the dominoes the observation names are rewritten. Every
        other domino keeps its last known pose, and the robot's entry --
        including the joint positions in ``simulator_state``, which
        ``_set_state`` needs to avoid re-deriving the arm by IK -- is
        carried forward untouched. Observations carry no visibility flag
        by design, so "absent means unchanged" is the policy, and it
        lives here where it can be tested.

        The start domino's yaw IS canonicalized here, but only while it
        is still standing. Perception picks one of the two
        180-degree-symmetric headings arbitrarily, and Push takes its
        entire direction from that yaw
        (``push.py``: ``facing = (sin(yaw), cos(yaw))``), so a look that
        writes back the other branch turns the opening push around.
        Every option boundary before the push is such a look, which is
        how a plan that ends in Push(start) comes to shove the start
        away from the target.

        The guard is what keeps this honest: once the start has actually
        gone over, its heading is real and is left alone -- there is no
        push left to orient, and flipping it would misreport which way
        it fell. Same rule, and same ``fallen_threshold``, as
        ``_task_from_perceived`` applies when it builds the task.

        Knocked-over dominoes are read back as knocked over: the
        perceived orientation becomes the env's ``(yaw, roll)`` pair, and
        ``roll`` is the very feature ``Toppled`` is defined on. This is
        the point of looking mid-episode -- the twin's guess about which
        dominoes a cascade felled is exactly what perception is there to
        correct.
        """
        comp = self._domino_component
        assert comp is not None, "env has no domino component"
        state = prev_state.copy()
        perceived = self._perceived_from_observation(obs)
        target_xy = self._target_xy(perceived, prev_state)
        for pd in perceived:
            dom = comp.dominos[pd.slot]
            if prev_state.get(dom, "is_held") > 0.5:
                # Perception cannot see a domino in the gripper: it snaps
                # every pose to a resting one on the table, so a held domino
                # is reported lying where it would be if the hand let go.
                # Writing that in teleports it out of the gripper and makes
                # _set_state rebuild the grasp constraint around the wrong
                # offset, leaving the twin holding something that is not
                # there. The twin's belief wins for whatever it is holding.
                continue
            world = pose_base_to_world(pd.pose_base, self._z_off)
            roll, yaw = self._env_angles(world, pd.capture_id)
            # A live observation carries no explicit push direction, so
            # "toward the target" is the one available intent.
            if pd.role == "start" and \
                    abs(roll) < DominoComponent.fallen_threshold:
                yaw = self._canonical_start_yaw(yaw, world, None, target_xy)
            state.set(dom, "x", world.xyz[0])
            state.set(dom, "y", world.xyz[1])
            state.set(dom, "z", world.xyz[2])
            state.set(dom, "yaw", yaw)
            state.set(dom, "roll", roll)
        return state
