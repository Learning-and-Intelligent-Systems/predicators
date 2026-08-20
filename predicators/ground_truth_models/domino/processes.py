"""Ground-truth processes for the domino environment."""

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

from predicators.ground_truth_models import GroundTruthProcessFactory, \
    GroundTruthSamplerFactory
from predicators.settings import CFG
from predicators.structs import Array, CausalProcess, EndogenousProcess, \
    ExogenousProcess, GroundAtom, LiftedAtom, Object, ParameterizedOption, \
    ParameterizedSampler, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler, wrap_angle

# Fixed parameter values for domino environment. Both z offsets were tuned on
# the Fetch; see _hand_z_correction for what that means on another arm.
_DOMINO_GRASP_Z_OFFSET = 0.0825  # domino_height * 0.55
# Slightly above the legacy drop height. With the skill-factory Pick grasp
# transform, 0.5695 leaves the held domino penetrating the table at the
# collision-aware Place goal; 0.58 clears the table and still settles to the
# intended upright pose.
_DOMINO_DROP_Z = 0.58
_DOMINO_OFFSET_Z = 0.0825  # domino_height * 0.55

# How far each hand reaches below its tool frame, measured at home as the
# lowest finger-link AABB against the tool link.
_FINGERTIP_REACH_BELOW_TOOL = {
    "fetch": 0.0320,
    "mobile_fetch": 0.0320,
    "panda": 0.0152,
}


def _hand_z_correction() -> float:
    """How much lower to command the tool frame on a shorter-fingered hand.

    The z offsets above position the TOOL frame, but what has to clear or
    contact the domino is the hand hanging below it -- and the Fetch reaches
    1.68cm further down than the Panda. Left uncorrected, the same number puts
    the Fetch's fingertips at 84% of a 0.15m domino's height and the Panda's at
    95%, the very top edge: the Panda barely catches the top on a grasp and
    skims over it on a push.

    Zero for the Fetch, so every value tuned on it is preserved exactly.
    """
    fetch_reach = _FINGERTIP_REACH_BELOW_TOOL["fetch"]
    return fetch_reach - _FINGERTIP_REACH_BELOW_TOOL.get(
        CFG.pybullet_robot, fetch_reach)


def _domino_depth() -> float:
    """Thickness of the domino actually in play, in metres.

    ``pybullet_domino_real`` sizes its component from
    ``CFG.domino_real_domino_dims`` (L, W, thickness); every other
    domino env takes the class ClassVar.
    """
    # pylint: disable=import-outside-toplevel  # local: avoid import cycle
    from predicators.envs.pybullet_domino import PyBulletDominoEnv
    if CFG.env == "pybullet_domino_real":
        return float(CFG.domino_real_domino_dims[2])
    return float(PyBulletDominoEnv.domino_depth)


def _push_approach_distance() -> float:
    """How far behind the block the gripper descends before pushing it."""
    return 3.0 * _domino_depth()


def _grasp_z_offset() -> float:
    """Pick grasp height, corrected for the hand in use."""
    return _DOMINO_GRASP_Z_OFFSET - _hand_z_correction()


def _push_contact_z_offset() -> float:
    """Push contact height, corrected for the hand in use."""
    return _DOMINO_OFFSET_Z - _hand_z_correction()


def _pick_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return fixed grasp_z_offset for domino pick."""
    del state, goal, rng, objs
    return np.array([_grasp_z_offset()], dtype=np.float32)


def _push_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return fixed push params for domino push."""
    if not CFG.domino_use_skill_factories:
        return np.array([], dtype=np.float32)
    del state, goal, rng, objs
    return np.array([_push_approach_distance(),
                     _push_contact_z_offset()],
                    dtype=np.float32)


def _place_sampler(state: State, goal: Set[GroundAtom],
                   rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    """Return a generator-faithful placement for the open-loop oracle.

    ``objs = [robot, domino1, domino2, target_pos, rotation]``. The process
    planner picks a discrete grid cell (``target_pos``) and angle (``rotation``)
    for the held ``domino1`` next to the reference ``domino2``. The grid is a
    uniform lattice (see ``augment_task_with_helper_objects``), so a turn block
    lands at the *same* cell a straight block would, differing only in angle --
    the generator's inward ``domino_width/2`` corner offset is absent from the
    lattice. Placing the held domino at the bare cell stalls corner cascades.

    Instead pick from the placements the generator would lay next to ``domino2``
    (``_generator_placements``, which carry the corner offset), rank-summing
    three signals that each, alone, mishandle one case -- future-target bridge
    (greedy: pulls a straight run onto the target), grid-cell distance (a
    uniform-grid turn cell sits on the straight position, missing corners), and
    angle error (the planner stamps spurious turn angles on straight runs). The
    cascade-correct candidate is top-ranked on >=2 of the three. Deterministic;
    final tiebreak is the planner's cell; bare cell if no candidate at all.
    """
    if not CFG.domino_use_skill_factories:
        return np.array([], dtype=np.float32)
    del goal, rng
    # objs = [robot, domino1, domino2, target_pos, rotation]
    held = objs[1]
    ref = objs[2]
    target_pos = objs[3]
    rotation = objs[4]
    gx = float(target_pos.name.split("_")[1])
    gy = float(target_pos.name.split("_")[2])
    gyaw = np.radians(float(rotation.name.split("_")[-1]))

    rx = state.get(ref, "x")
    ry = state.get(ref, "y")
    ryaw = state.get(ref, "yaw")
    candidates = _generator_placements(rx, ry, ryaw)
    if not candidates:
        # Fallback: bare lattice cell (no generator candidate available).
        return np.array([gx, gy, _DOMINO_DROP_Z, gyaw], dtype=np.float32)
    bridges = [
        _future_target_bridge_score(state, held, c[0], c[1], c[2])
        for c in candidates
    ]
    dgrids = [float(np.hypot(c[0] - gx, c[1] - gy)) for c in candidates]
    angerrs = [abs(wrap_angle(c[2] - gyaw)) for c in candidates]

    def _rank(vals: List[float], i: int, higher_better: bool = False) -> int:
        # Number of candidates strictly better than ``i`` (ties share a rank).
        if higher_better:
            return sum(1 for v in vals if v > vals[i] + 1e-9)
        return sum(1 for v in vals if v < vals[i] - 1e-9)

    def _total(i: int) -> Tuple[int, float]:
        rank_sum = (_rank(bridges, i, higher_better=True) + _rank(dgrids, i) +
                    _rank(angerrs, i))
        return (rank_sum, dgrids[i])

    best_i = min(range(len(candidates)), key=_total)
    cx, cy, cyaw = candidates[best_i]
    return np.array([cx, cy, _DOMINO_DROP_Z, cyaw], dtype=np.float32)


class PyBulletDominoGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the domino grid environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "pybullet_domino_grid", "pybullet_domino", "pybullet_domino_real",
            "pybullet_domino_real_geometry"
        }

    @classmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        # These processes are defined over the grid (loc/angle/direction).
        # Only oracle / process-planning approaches request them, and they do
        # so unconditionally, so the grid is intrinsic to those approaches.

        # Types
        robot_type = types["robot"]
        domino_type = types["domino"]
        position_type = types["loc"]
        rotation_type = types["angle"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        InFront = predicates["InFront"]
        Upright = predicates["Upright"]
        StartBlock = predicates["InitialBlock"]
        Toppled = predicates["Toppled"]
        Tilting = predicates["Tilting"]
        DominoAtPos = predicates["DominoAtPos"]
        DominoAtRot = predicates["DominoAtRot"]
        MovableBlock = predicates["MovableBlock"]
        PosClear = predicates["PosClear"]
        AdjacentTo = predicates["AdjacentTo"]
        if CFG.domino_has_glued_dominos:
            DominoNotGlued = predicates["DominoNotGlued"]
        # Note: Tilting predicate exists but represents the goal state
        # Note: The "Falling" predicate from the sketch is not implemented in the current environment  # pylint: disable=line-too-long
        # We would need to add it to the environment for the DominoFall
        # exogenous process

        # Options
        Push = options["Push"]
        Pick = options["Pick"]
        Place = options["Place"]
        Wait = options["Wait"]

        processes: Set[CausalProcess] = set()

        # --- Endogenous Processes / Actions ---

        # PushStartBlock: Push the start block to initiate the domino chain
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        parameters = [robot, domino]
        # With restricted push the "Push" option finds the start block from
        # the state itself, so it takes only the robot. The unrestricted
        # option also takes the domino to push.
        if CFG.domino_restricted_push:
            option_vars = [robot]
        else:
            option_vars = [robot, domino]
        option = Push
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(StartBlock, [domino]),
            LiftedAtom(Upright, [domino]),
        }
        add_effects = {
            LiftedAtom(Tilting, [domino]),
        }
        delete_effects: Set[LiftedAtom] = {
            LiftedAtom(Upright, [domino]),
        }
        ignore_effects = {DominoAtPos, DominoAtRot, PosClear, AdjacentTo}
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        push_start_block_process = EndogenousProcess(
            "PushStartBlock", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution,
            torch.tensor(1.0), option, option_vars, _push_sampler,
            ignore_effects)
        processes.add(push_start_block_process)

        # PickDomino: Position-based pick process
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        position = Variable("?pos", position_type)
        rotation = Variable("?rot", rotation_type)
        parameters = [robot, domino, position, rotation]
        option_vars = [robot, domino]
        option = Pick
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino, position]),
            LiftedAtom(DominoAtRot, [domino, rotation]),
            LiftedAtom(MovableBlock, [domino]),
            LiftedAtom(Upright, [domino]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, domino]),
            LiftedAtom(PosClear, [position]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino, position]),
            LiftedAtom(DominoAtRot, [domino, rotation]),
        }
        ignore_effects = {
            Tilting, Upright, DominoAtRot, DominoAtPos, PosClear, Toppled
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        pick_domino_process = EndogenousProcess("PickDomino",
                                                parameters, condition_at_start,
                                                set(), set(), add_effects,
                                                delete_effects,
                                                delay_distribution,
                                                torch.tensor(1.0), option,
                                                option_vars, _pick_sampler,
                                                ignore_effects)
        processes.add(pick_domino_process)

        # PlaceDomino: Place domino at specific position and rotation
        # Not in will still be in front to something
        robot = Variable("?robot", robot_type)
        domino1 = Variable("?domino1", domino_type)
        domino2 = Variable("?domino2", domino_type)
        target_pos = Variable("?pos1", position_type)
        rotation = Variable("?rot", rotation_type)
        parameters = [robot, domino1, domino2, target_pos, rotation]
        option_vars = [robot]
        option = Place
        condition_at_start = {
            LiftedAtom(Holding, [robot, domino1]),
            LiftedAtom(PosClear, [target_pos]),
            LiftedAtom(Upright, [domino2]),
            LiftedAtom(AdjacentTo, [target_pos, domino2]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino1, target_pos]),
            LiftedAtom(DominoAtRot, [domino1, rotation]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, domino1]),
            LiftedAtom(PosClear, [target_pos]),
        }
        ignore_effects = {
            DominoAtRot, DominoAtPos, PosClear, Tilting, AdjacentTo
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(3.0),
                                                   sigma=torch.tensor(0.1))
        place_domino_process = EndogenousProcess("PlaceDomino", parameters,
                                                 condition_at_start, set(),
                                                 set(), add_effects,
                                                 delete_effects,
                                                 delay_distribution,
                                                 torch.tensor(1.0), option,
                                                 option_vars, _place_sampler,
                                                 ignore_effects)
        processes.add(place_domino_process)

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = Wait
        wait_delay_distribution = ConstantDelay(1)
        ignore_effects = {DominoAtRot, DominoAtPos, PosClear, AdjacentTo}
        wait_process = EndogenousProcess("Wait", parameters, set(), set(),
                                         set(), set(), set(),
                                         wait_delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler,
                                         ignore_effects)
        processes.add(wait_process)

        # --- Exogenous Processes ---

        # Note: The DominoFall process from the sketch requires a "Falling" predicate
        # which is not currently implemented in the environment.
        # This process would look like:
        domino1 = Variable("?d1", domino_type)
        domino2 = Variable("?d2", domino_type)
        parameters = [domino1, domino2]
        condition_at_start = {
            LiftedAtom(InFront, [domino1, domino2]),
            LiftedAtom(Tilting, [domino2]),
        }
        if CFG.domino_oracle_knows_glued_dominos:
            condition_at_start.update({
                LiftedAtom(DominoNotGlued, [domino1]),
            })
        condition_overall = condition_at_start.copy()
        add_effects = {
            LiftedAtom(Tilting, [domino1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(1.0),
                                                   sigma=torch.tensor(0.1))
        domino_fall_process = ExogenousProcess(
            "DominoFallFromBeingInFrontOfTilting", parameters,
            condition_at_start, condition_overall, set(), add_effects, set(),
            delay_distribution, torch.tensor(1.0))
        processes.add(domino_fall_process)

        # Individual Domino Fall from Tilting to Fall flat
        domino1 = Variable("?d1", domino_type)
        parameters = [domino1]
        condition_at_start = {
            LiftedAtom(Tilting, [domino1]),
        }
        condition_overall = condition_at_start.copy()
        add_effects = {
            LiftedAtom(Toppled, [domino1]),
        }
        delete_effects = {
            LiftedAtom(Tilting, [domino1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                   sigma=torch.tensor(0.1))
        domino_tilting_delete_process = ExogenousProcess(
            "DominoTiltingDelete", parameters, condition_at_start,
            condition_overall, set(), add_effects, delete_effects,
            delay_distribution, torch.tensor(1.0))
        processes.add(domino_tilting_delete_process)

        return processes


# ---------------------------------------------------------------------------
# Grid-free per-skill samplers (NSRTSampler / ParameterizedSampler
# signature) for
# bilevel refinement. The NSRT samplers above read the placement off grid
# ``loc``/``angle`` objects in ``objs``; these instead compute it
# geometrically from the step's ``InFront`` subgoal (passed in the atoms
# slot), so they work in the grid-free agent_bilevel path. Both versions
# coexist intentionally. Refinement clips the returned params to the box.
# ---------------------------------------------------------------------------

_DOMINO_POS_GAP = 0.098  # PyBulletDominoEnv.pos_gap (domino_width * 1.4)
_DOMINO_WIDTH = 0.07  # PyBulletDominoEnv.domino_width
_DOMINO_TARGET_COLOR = (0.85, 0.7, 0.85)
_DOMINO_COLOR_EPS = 1e-3


def _deterministic(sampler: ParameterizedSampler) -> ParameterizedSampler:
    """Flag a sampler as returning constant params (ignores state/rng).

    Backtracking refinement reads this flag to cap such a step's retries at
    1: re-drawing a constant sampler yields the identical option, so spending
    the full per-step budget re-descending through it on every backtrack is
    wasted work (it can never produce a different outcome).
    """
    setattr(sampler, "deterministic", True)
    return sampler


@_deterministic
def _pick_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                         rng: np.random.Generator,
                         objects: Sequence[Object]) -> Array:
    """Grid-free Pick sampler: fixed grasp height above the domino origin."""
    del state, subgoal_atoms, rng, objects
    return np.array([_grasp_z_offset()], dtype=np.float32)


@_deterministic
def _push_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                         rng: np.random.Generator,
                         objects: Sequence[Object]) -> Array:
    """Grid-free Push sampler: approach distance / contact height."""
    del state, subgoal_atoms, rng, objects
    return np.array([_push_approach_distance(),
                     _push_contact_z_offset()],
                    dtype=np.float32)


def _score_placement(state: State, subgoal_atoms: Set[GroundAtom],
                     held: Object, hx: float, hy: float, hyaw: float) -> int:
    """Count subgoal atoms that hold if ``held`` is placed at (hx, hy,
    hyaw)."""
    s2 = state.copy()
    s2.set(held, "x", hx)
    s2.set(held, "y", hy)
    s2.set(held, "yaw", hyaw)
    s2.set(held, "roll", 0.0)
    s2.set(held, "is_held", 0.0)
    return sum(1 for atom in subgoal_atoms if atom.holds(s2))


def _is_cardinal(angle: float) -> bool:
    """True when ``angle`` is within ~10 deg of a cardinal (axis-aligned) yaw.

    Mirrors the cardinal-facing gate in
    ``DominoComponent._InFront_holds``: a settled reference domino sits
    a degree or two off cardinal, so a hard equality would make chained
    placements onto it unsatisfiable.
    """
    card_thresh = float(np.sin(np.radians(10)))
    return bool(
        abs(np.sin(angle)) < card_thresh or abs(np.cos(angle)) < card_thresh)


def _generator_placements(xr: float, yr: float,
                          ryaw: float) -> List[Tuple[float, float, float]]:
    """Every placement the task generator would lay next to a reference.

    Reproduces ``DominoTaskGenerator._place_straight_domino`` /
    ``_place_turn90_domino`` exactly -- one ``pos_gap`` along a cardinal
    travel direction, with 45-deg turn blocks carrying the generator's
    half-width inward side offset -- expressed relative to a reference domino
    at ``(xr, yr, ryaw)``. Each returned ``(cx, cy, cyaw)`` is a valid
    ``InFront`` placement off the reference.

    A cardinal reference yields, for each of the two chain (forward / backward)
    directions, the straight successor and the two turn-start (``d1``) blocks
    (left / right). A non-cardinal reference -- an already-placed 45-deg
    turn-start block -- yields the turn-completing (``d2``) block that bends
    the chain the rest of the way through the corner.
    """
    gap = _DOMINO_POS_GAP
    s_off = -_DOMINO_WIDTH / 2  # generator's d1_side_offset / side_offset
    out: List[Tuple[float, float, float]] = []
    if _is_cardinal(ryaw):
        for rotation in (ryaw, wrap_angle(ryaw + np.pi)):
            # Straight successor: one gap along travel, same (box) yaw.
            out.append(
                (xr + gap * np.sin(rotation), yr + gap * np.cos(rotation),
                 wrap_angle(ryaw)))
            # Turn-start (d1): one gap ahead, nudged a half width orthogonal
            # to the post-turn travel direction, yaw stepped +-45.
            for turn in (1.0, -1.0):
                d1_dir = wrap_angle(rotation - turn * np.pi / 4)
                cx = xr + gap * np.sin(rotation) + turn * s_off * np.cos(
                    d1_dir)
                cy = yr + gap * np.cos(rotation) - turn * s_off * np.sin(
                    d1_dir)
                out.append((cx, cy, wrap_angle(ryaw + turn * np.pi / 4)))
    else:
        # Turn-completing block (d2) off an already-placed turn-start block.
        # Take whichever turn sign(s) leave the pre-turn travel cardinal.
        for turn in (1.0, -1.0):
            base = wrap_angle(ryaw - turn * np.pi / 4)
            if not _is_cardinal(base):
                continue
            d1_dir = wrap_angle(base - turn * np.pi / 4)
            d2_rot = wrap_angle(base - turn * np.pi / 2)
            cx = xr + gap * np.sin(d1_dir) + turn * s_off * np.cos(d2_rot)
            cy = yr + gap * np.cos(d1_dir) - turn * s_off * np.sin(d2_rot)
            out.append((cx, cy, wrap_angle(base + turn * np.pi / 2)))
    return out


def _is_target_domino(state: State, domino: Object) -> bool:
    """Check whether ``domino`` has the target-block color."""
    return all(
        abs(state.get(domino, feat) - val) < _DOMINO_COLOR_EPS
        for feat, val in zip(("r", "g", "b"), _DOMINO_TARGET_COLOR))


def _future_target_bridge_score(state: State, held: Object, hx: float,
                                hy: float, hyaw: float) -> float:
    """Tie-break score for placements that can be completed to a target.

    The immediate ``InFront(held, ref)`` subgoal underdetermines which
    side of the start domino to place the bridge on. Prefer placements
    for which one additional domino can be placed at the intersection of
    generator-faithful successors from the held domino and from a purple
    target domino. This keeps the sampler from spending most refinement
    attempts on locally valid but globally dead first placements.
    """
    dominoes = [o for o in state if o.type.name == "domino" and o is not held]
    targets = [d for d in dominoes if _is_target_domino(state, d)]
    if not targets:
        return 0.0
    held_next = _generator_placements(hx, hy, hyaw)
    if not held_next:
        return 0.0
    best_resid = float("inf")
    yaw_scale = _DOMINO_POS_GAP / np.pi
    for target in targets:
        tx = state.get(target, "x")
        ty = state.get(target, "y")
        tyaw = state.get(target, "yaw")
        for hx2, hy2, hyaw2 in held_next:
            for tx2, ty2, tyaw2 in _generator_placements(tx, ty, tyaw):
                yaw_resid = abs(wrap_angle(hyaw2 - tyaw2)) * yaw_scale
                resid = float(np.hypot(hx2 - tx2, hy2 - ty2) + yaw_resid)
                best_resid = min(best_resid, resid)
    if best_resid == float("inf"):
        return 0.0
    return -best_resid


def _place_option_sampler(state: State, subgoal_atoms: Set[GroundAtom],
                          rng: np.random.Generator,
                          objects: Sequence[Object]) -> Array:
    """Grid-free Place sampler that draws a generator-faithful placement.

    Builds the discrete set of placements the task generator could lay next
    to each reference domino named in an ``InFront`` subgoal -- straight, or a
    45-deg left / right turn block, in either chain direction (see
    ``_generator_placements``) -- scores each by how many of the step's
    subgoal atoms it satisfies, and draws one uniformly at random from those
    tied for the best score. Randomizing (rather than always returning the
    first / straight placement) is what lets backtracking that re-draws this
    step reach a turn when the lone subgoal (e.g. ``InFront(d1, d0)``) is
    satisfied equally by straight and by a turn and a later step needs the
    bend. No jitter is added -- the generator placements are already the
    exact, cascade-tuned poses. Raises (so refinement falls back to uniform)
    when the held domino or a usable reference can't be found.
    """
    del objects
    dominoes = [o for o in state if o.type.name == "domino"]
    held = [d for d in dominoes if state.get(d, "is_held") > 0.5]
    if len(held) != 1:
        raise ValueError(f"expected one held domino, found {len(held)}")
    held_d = held[0]

    refs = []
    for atom in subgoal_atoms:
        if atom.predicate.name != "InFront":
            continue
        d1, d2 = atom.objects
        if held_d is d1 and held_d is not d2:
            refs.append(d2)
        elif held_d is d2 and held_d is not d1:
            refs.append(d1)
    if not refs:
        raise ValueError("no InFront subgoal references the held domino")

    # Collect every generator-faithful candidate, scored by how many of the
    # step's subgoal atoms it satisfies. The candidates come straight from the
    # task generator's geometry, so each is a valid InFront placement off its
    # reference and the set is exactly what the generator could have laid.
    candidates: List[Tuple[int, float, float, float, float]] = []
    for ref in refs:
        xr = state.get(ref, "x")
        yr = state.get(ref, "y")
        rot = state.get(ref, "yaw")
        for cx, cy, cyaw in _generator_placements(xr, yr, rot):
            score = _score_placement(state, subgoal_atoms, held_d, cx, cy,
                                     cyaw)
            future_score = _future_target_bridge_score(state, held_d, cx, cy,
                                                       cyaw)
            candidates.append((score, future_score, cx, cy, cyaw))
    if not candidates:
        raise ValueError("no usable reference domino for placement")

    # Randomize among the placements tied for the best score, so backtracking
    # that re-draws this step explores a turn instead of always returning the
    # straight pose. Score alone disambiguates: a multi-edge step (a second
    # InFront naming the next block) is satisfied only by the turn block that
    # bends toward it, which no straight placement matches.
    best_score = max(c[0] for c in candidates)
    best_future_score = max(c[1] for c in candidates if c[0] == best_score)
    tied = [
        c for c in candidates
        if c[0] == best_score and abs(c[1] - best_future_score) < 1e-9
    ]
    _, _, cx, cy, cyaw = tied[int(rng.integers(len(tied)))]
    return np.array([cx, cy, _DOMINO_DROP_Z, cyaw], dtype=np.float32)


class PyBulletDominoGroundTruthSamplerFactory(GroundTruthSamplerFactory):
    """Ground-truth grid-free per-skill samplers for the domino env."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "pybullet_domino_grid", "pybullet_domino", "pybullet_domino_real",
            "pybullet_domino_real_geometry"
        }

    @classmethod
    def get_samplers(cls, env_name: str) -> Dict[str, ParameterizedSampler]:
        del env_name
        return {
            "Pick": _pick_option_sampler,
            "Push": _push_option_sampler,
            "Place": _place_option_sampler,
        }
