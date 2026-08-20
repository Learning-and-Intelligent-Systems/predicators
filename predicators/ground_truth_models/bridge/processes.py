"""Ground-truth processes for the bridge (glue construction) environment.

These drive the ``oracle_process_planning`` demo generator: endogenous
(option-backed) processes for pick / place / stack / butt-join / seat /
glue application, and three exogenous ``Cure*Joint`` processes encoding
the hidden dwell-time dynamics (a wet face held in aligned contact for
~``cure_threshold`` steps -> the pair is ``Attached`` and physically
welded).

Direction conventions the samplers rely on (mirrored by the env's task
generator): the span row grows in +x (``NextToEnd(right, left)`` = right
butts against left's ``end_b`` face), and goals pin every geometric
atom (AtSite / OnBlock / NextToEnd / SeatedOn), so the planner's
bindings always match a physically consistent left-to-right build.
"""
from typing import Dict, Sequence, Set

import numpy as np
import torch

from predicators.envs.pybullet_bridge import PyBulletBridgeEnv
from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.structs import Array, CausalProcess, DelayDistribution, \
    EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler

_ENV = PyBulletBridgeEnv
_LEG_H = 2 * _ENV.leg_half_extents[2]  # 0.10
_SPAN_LEN = 2 * _ENV.span_half_extents[0]  # 0.10
_SPAN_TH = 2 * _ENV.span_half_extents[2]  # 0.05
_TABLE = _ENV.table_height
# Release the held block ~1-1.5 cm above its resting height (probes
# show a welded span self-levels cleanly from up to ~2 cm; the lower
# bound must exceed the max pick grasp offset so a deep-grasped block
# never reaches the goal pose already in contact).
_DROP = 0.015
# EE-above-held-block-top offset at release: the block was grasped at
# its top, so the EE sits roughly at the block's top surface.
_LEG_TOP_EE = _TABLE + _LEG_H + _DROP  # place a leg on the table: 0.51
_SPAN_TABLE_EE = _TABLE + _SPAN_TH + _DROP  # place a span flat: 0.46
_STACK_EE = _TABLE + 2 * _LEG_H + _DROP  # stack a leg on a leg: 0.61


def _pick_sampler(state: State, goal: Set[GroundAtom],
                  rng: np.random.Generator, objs: Sequence[Object]) -> Array:
    del state, goal, objs
    # Descend to ~the grasp target (block top). Keep the range TIGHT:
    # the grasped block hangs a full grasp-offset lower relative to the
    # EE, and the fixed release heights only budget ~1.5 cm of drop --
    # a 1 cm offset put a span's underside 1.3 mm through the table at
    # the place descend goal, which BiRRT rejects forever.
    return np.array([rng.uniform(0.0, 0.005)], dtype=np.float32)


def _apply_glue_sampler(state: State, goal: Set[GroundAtom],
                        rng: np.random.Generator,
                        objs: Sequence[Object]) -> Array:
    del state, goal, objs
    return np.array([rng.uniform(0.0, 0.01)], dtype=np.float32)


def _place_leg_at_site_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
    del goal
    # objs = [robot, leg, site]. Small jitter so refinement retries
    # differ (a deterministic sampler just repeats an identical failure).
    site = objs[2]
    x = state.get(site, "x") + rng.uniform(-0.003, 0.003)
    y = state.get(site, "y") + rng.uniform(-0.003, 0.003)
    z = _LEG_TOP_EE + rng.uniform(0.0, 0.005)
    return np.array([x, y, z, 0.0], dtype=np.float32)


def _stack_leg_sampler(state: State, goal: Set[GroundAtom],
                       rng: np.random.Generator,
                       objs: Sequence[Object]) -> Array:
    del goal
    # objs = [robot, top, bottom, site]. Target the bottom leg's CURRENT
    # xy; tolerance is tight (the 5 cm column topples past ~2.5 cm).
    bottom = objs[2]
    x = state.get(bottom, "x") + rng.uniform(-0.003, 0.003)
    y = state.get(bottom, "y") + rng.uniform(-0.003, 0.003)
    z = _STACK_EE + rng.uniform(0.0, 0.005)
    return np.array([x, y, z, 0.0], dtype=np.float32)


def _place_next_to_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
    del goal
    # objs = [robot, right, left]: butt the held block against the left
    # block's end_b (+x) face, with a small nominal gap so the landing
    # does not shove the (wet-glued) left block out of alignment.
    left = objs[2]
    x = state.get(left, "x") + _SPAN_LEN + _ENV.lateral_place_gap + \
        rng.uniform(0.0, 0.004)
    y = state.get(left, "y") + rng.uniform(-0.003, 0.003)
    # Gentler landing than the generic places: any landing shift here
    # is FROZEN into the weld and transfers to the far seat joint, so
    # minimize the drop (the pick's grasp offset is capped at 5 mm, so
    # a 10 mm budget still never reaches the goal pose in contact).
    z = _TABLE + _SPAN_TH + 0.010 + rng.uniform(0.0, 0.004)
    return np.array([x, y, z, 0.0], dtype=np.float32)


def _seat_span_sampler(state: State, goal: Set[GroundAtom],
                       rng: np.random.Generator,
                       objs: Sequence[Object]) -> Array:
    del goal
    # SeatSpan2: objs = [robot, spanA, spanB, legL, legR, siteL, siteR]
    # SeatSpan3: objs = [robot, spanA, mid, spanB, legL, legR, baseL,
    # baseR]. The whole welded span assembly hangs from the grasped
    # spanA; place spanA so its outer end sits flush over legL (then
    # spanB lands over legR by the rigid geometry).
    span_a = objs[1]
    if len(objs) == 7:  # SeatSpan2
        span_b, leg_l = objs[2], objs[3]
    else:  # SeatSpan3
        span_b, leg_l = objs[3], objs[4]
    # spanA's center sits INBOARD of its leg -- shifted toward the rest
    # of the assembly -- by (span_half - leg_half). The inboard
    # direction is read from where the welded partner currently hangs
    # relative to spanA (the planner may ground spanA as EITHER end of
    # the assembly; a hardcoded +x here seated the assembly 5 cm off
    # when spanA was the right-end block, dropping the unsupported end
    # and pivoting the weldment into the open gripper).
    inboard = 1.0 if state.get(span_b, "x") > state.get(span_a, "x") \
        else -1.0
    offset = _ENV.span_half_extents[0] - _ENV.leg_half_extents[0]
    x = state.get(leg_l, "x") + inboard * offset + rng.uniform(-0.003, 0.003)
    y = state.get(leg_l, "y") + rng.uniform(-0.003, 0.003)
    # Release height from STATIC task geometry only. Samplers run at
    # planning time on predicted states, so live robot-relative reads
    # are stale garbage (a robot-z-based "hang" read the home pose,
    # blew past the release_z bound, and crash-dropped the assembly),
    # and the predicted z of a STACKED leg is unreliable. The leg-stack
    # height follows from the process arity: SeatSpan2 means 1-block
    # legs, SeatSpan3 means welded 2-block stacks.
    # EE-at-release = seat surface + span thickness + the ~8 mm the EE
    # sits above the grasped span's top + ~1.2 cm drop clearance.
    n_stack = 1 if len(objs) == 7 else 2
    seat_surface_z = _TABLE + n_stack * _LEG_H
    release_z = seat_surface_z + _SPAN_TH + 0.02
    return np.array([x, y, release_z, 0.0], dtype=np.float32)


def _footprint_radius(obj: Object) -> float:
    """Conservative horizontal footprint radius by object identity."""
    if obj.type.name == "bottle":
        return float(np.hypot(*_ENV.bottle_half_extents[:2]))
    if obj.name.startswith("span"):
        return float(np.hypot(*_ENV.span_half_extents[:2]))
    return float(np.hypot(*_ENV.leg_half_extents[:2]))


def _stage_spot_sampler(state: State, held: Object,
                        rng: np.random.Generator) -> np.ndarray:
    """A clear table spot for staging (escape hatch / bottle return).

    Candidates come from the env's staging grid, EXCLUDING the middle
    row: that row hosts the span-assembly strip, and the strip cells
    look empty until the very Place that needs them (returning the
    bottle there blocked the span row in early runs).

    Clearance is SIZE-AWARE (held footprint + neighbor footprint +
    margin): a blanket radius larger than the grid pitch rejects every
    cell in the packed full-variant grid -- even genuinely free ones --
    and the fallback then dropped the bottle on top of a staged block.
    """
    rows = [_ENV.stage_row_back, _ENV.stage_row_front, _ENV.stage_row_mid]
    candidates = []
    for row in rows:
        for col in _ENV.stage_cols:
            if np.hypot(col - _ENV.robot_base_pos[0],
                        row - _ENV.robot_base_pos[1]) > \
                    _ENV.reach_radius - 0.02:
                continue
            candidates.append((col, row))
    rng.shuffle(candidates)
    # Stable preference: back/front rows before the middle row.
    candidates.sort(key=lambda c: rows.index(c[1]))
    held_r = _footprint_radius(held)
    # Row-growth corridors: the span row grows in +x from each lying
    # span, so cells to a span's right at its y are future placement
    # targets even though they look empty now (the bottle parked there
    # once and the row build descended straight into it).
    corridors = []
    for o in state:
        if o.type.name == "block" and o != held and \
                state.get(o, "upright") <= 0.5:
            corridors.append((state.get(o, "x"), state.get(o, "y")))
    tx, ty = candidates[0]
    for col, row in candidates:
        clear = True
        for o in state:
            if o.type.name not in ("block", "bottle", "site") or o == held:
                continue
            if o.type.name == "site":
                # Keep sites usable for later leg placements.
                required = held_r + 0.045 + 0.01
            else:
                required = held_r + _footprint_radius(o) + 0.015
            if np.hypot(state.get(o, "x") - col,
                        state.get(o, "y") - row) < required:
                clear = False
                break
        if clear:
            for sx, sy in corridors:
                if abs(row - sy) < held_r + 0.06 and \
                        sx - 0.12 < col < sx + 0.35:
                    clear = False
                    break
        if clear:
            tx, ty = col, row
            break
    jitter = rng.uniform(-0.005, 0.005, size=2)
    return np.array([tx + jitter[0], ty + jitter[1]])


def _place_block_on_table_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    del goal
    held = objs[1]
    tx, ty = _stage_spot_sampler(state, held, rng)
    release_z = _LEG_TOP_EE if _ENV._is_leg_shaped(held) \
        else _SPAN_TABLE_EE  # pylint: disable=protected-access
    return np.array([tx, ty, release_z, 0.0], dtype=np.float32)


def _place_bottle_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
    del goal
    bottle = objs[1]
    tx, ty = _stage_spot_sampler(state, bottle, rng)
    release_z = _TABLE + 2 * _ENV.bottle_half_extents[2] + _DROP
    return np.array([tx, ty, release_z, 0.0], dtype=np.float32)


class PyBulletBridgeGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the bridge environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_bridge"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name
        robot_type = types["robot"]
        block_type = types["block"]
        bottle_type = types["bottle"]
        site_type = types["site"]

        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        HoldingBottle = predicates["HoldingBottle"]
        GlueTop = predicates["GlueTop"]
        GlueEndB = predicates["GlueEndB"]
        OnBlock = predicates["OnBlock"]
        NextToEnd = predicates["NextToEnd"]
        SeatedOn = predicates["SeatedOn"]
        AtSite = predicates["AtSite"]
        SiteFree = predicates["SiteFree"]
        Attached = predicates["Attached"]
        Standing = predicates["Standing"]
        Lying = predicates["Lying"]
        Loose = predicates["Loose"]
        Resting = predicates["Resting"]
        TopFree = predicates["TopFree"]

        PickBlock = options["PickBlock"]
        PickBottle = options["PickBottle"]
        Place = options["Place"]
        ApplyGlueTop = options["ApplyGlueTop"]
        ApplyGlueEndB = options["ApplyGlueEndB"]
        Wait = options["Wait"]

        processes: Set[CausalProcess] = set()

        def _delay(mu: float) -> DelayDistribution:
            return DiscreteGaussianDelay(mu=torch.tensor(mu),
                                         sigma=torch.tensor(0.1))

        # -- PickBlockFromTable ---------------------------------------------
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        processes.add(
            EndogenousProcess(
                "PickBlockFromTable",
                [robot, blk],
                {
                    LiftedAtom(HandEmpty, [robot]),
                    # A block with another resting on it cannot be
                    # top-grasped: the grasp goal puts the palm inside
                    # the covering block (a replan once tried to pick a
                    # leg from UNDER the seated span and BiRRT rejected
                    # the goal forever).
                    LiftedAtom(TopFree, [blk]),
                },
                set(),
                set(),
                {LiftedAtom(Holding, [robot, blk])},
                {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(Resting, [blk]),
                },
                _delay(2.0),
                torch.tensor(1.0),
                PickBlock,
                [robot, blk],
                _pick_sampler))

        # -- PickGlueBottle ---------------------------------------------------
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        processes.add(
            EndogenousProcess("PickGlueBottle", [robot, bottle],
                              {LiftedAtom(HandEmpty, [robot])}, set(), set(),
                              {LiftedAtom(HoldingBottle, [robot, bottle])},
                              {LiftedAtom(HandEmpty, [robot])}, _delay(2.0),
                              torch.tensor(1.0), PickBottle, [robot, bottle],
                              _pick_sampler))

        # -- PlaceBottleOnTable -----------------------------------------------
        robot = Variable("?robot", robot_type)
        bottle = Variable("?bottle", bottle_type)
        processes.add(
            EndogenousProcess("PlaceBottleOnTable", [robot, bottle],
                              {LiftedAtom(HoldingBottle, [robot, bottle])},
                              set(), set(), {LiftedAtom(HandEmpty, [robot])},
                              {LiftedAtom(HoldingBottle, [robot, bottle])},
                              _delay(3.0), torch.tensor(1.0), Place, [robot],
                              _place_bottle_sampler))

        # -- PlaceLegAtSite ---------------------------------------------------
        robot = Variable("?robot", robot_type)
        leg = Variable("?leg", block_type)
        site = Variable("?site", site_type)
        processes.add(
            EndogenousProcess(
                "PlaceLegAtSite", [robot, leg, site], {
                    LiftedAtom(Holding, [robot, leg]),
                    LiftedAtom(SiteFree, [site]),
                    LiftedAtom(Standing, [leg]),
                    LiftedAtom(Loose, [leg]),
                }, set(), set(), {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(AtSite, [leg, site]),
                    LiftedAtom(Resting, [leg]),
                }, {
                    LiftedAtom(Holding, [robot, leg]),
                    LiftedAtom(SiteFree, [site]),
                }, _delay(3.0), torch.tensor(1.0), Place, [robot],
                _place_leg_at_site_sampler))

        # -- StackLegOnLeg (full variant) -------------------------------------
        robot = Variable("?robot", robot_type)
        top = Variable("?top", block_type)
        bottom = Variable("?bottom", block_type)
        site = Variable("?site", site_type)
        processes.add(
            EndogenousProcess(
                "StackLegOnLeg", [robot, top, bottom, site], {
                    LiftedAtom(Holding, [robot, top]),
                    LiftedAtom(AtSite, [bottom, site]),
                    LiftedAtom(GlueTop, [bottom]),
                    LiftedAtom(Standing, [top]),
                    LiftedAtom(Loose, [top]),
                }, set(), set(), {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(OnBlock, [top, bottom]),
                    LiftedAtom(Resting, [top]),
                }, {
                    LiftedAtom(Holding, [robot, top]),
                    LiftedAtom(TopFree, [bottom]),
                }, _delay(3.0), torch.tensor(1.0), Place, [robot],
                _stack_leg_sampler))

        # -- PlaceSpanNextTo --------------------------------------------------
        robot = Variable("?robot", robot_type)
        right = Variable("?right", block_type)
        left = Variable("?left", block_type)
        processes.add(
            EndogenousProcess(
                "PlaceSpanNextTo",
                [robot, right, left],
                {
                    LiftedAtom(Holding, [robot, right]),
                    LiftedAtom(GlueEndB, [left]),
                    LiftedAtom(Lying, [right]),
                    LiftedAtom(Lying, [left]),
                    # A block welded into an assembly cannot be
                    # individually re-placed (the weld drags the whole
                    # assembly along and the intended adjacency never
                    # forms). This also makes wrong-direction row
                    # builds a dead end instead of an attractive
                    # abstract shortcut.
                    LiftedAtom(Loose, [right]),
                },
                set(),
                set(),
                {
                    LiftedAtom(HandEmpty, [robot]),
                    LiftedAtom(NextToEnd, [right, left]),
                    LiftedAtom(Resting, [right]),
                },
                {LiftedAtom(Holding, [robot, right])},
                _delay(3.0),
                torch.tensor(1.0),
                Place,
                [robot],
                _place_next_to_sampler))

        # -- PlaceBlockOnTable (staging escape hatch) -------------------------
        robot = Variable("?robot", robot_type)
        blk = Variable("?block", block_type)
        processes.add(
            EndogenousProcess("PlaceBlockOnTable", [robot, blk],
                              {LiftedAtom(Holding, [robot, blk])}, set(),
                              set(), {
                                  LiftedAtom(HandEmpty, [robot]),
                                  LiftedAtom(Resting, [blk]),
                              }, {LiftedAtom(Holding, [robot, blk])},
                              _delay(3.0), torch.tensor(1.0), Place, [robot],
                              _place_block_on_table_sampler))

        # -- ApplyGlueTop / ApplyGlueEndB -------------------------------------
        # Shape conditions prune groundings to the joints the tasks use:
        # top glue goes on standing legs, end glue on lying spans.
        for glue_pred, shape_pred, option, proc_name in ((GlueTop, Standing,
                                                          ApplyGlueTop,
                                                          "ApplyGlueTop"),
                                                         (GlueEndB, Lying,
                                                          ApplyGlueEndB,
                                                          "ApplyGlueEndB")):
            robot = Variable("?robot", robot_type)
            bottle = Variable("?bottle", bottle_type)
            blk = Variable("?block", block_type)
            processes.add(
                EndogenousProcess(
                    proc_name, [robot, bottle, blk], {
                        LiftedAtom(HoldingBottle, [robot, bottle]),
                        LiftedAtom(shape_pred, [blk]),
                    } | ({LiftedAtom(
                        TopFree, [blk])} if glue_pred is GlueTop else set()),
                    set(), set(), {LiftedAtom(glue_pred, [blk])}, set(),
                    _delay(4.0), torch.tensor(1.0), option,
                    [robot, bottle, blk], _apply_glue_sampler))

        # -- SeatSpan2 / SeatSpan3 (place the welded span assembly) -----------
        # Two arities: a 2-block assembly (simple variant, seated on
        # 1-block legs standing AT SITES) and a 3-block one (full,
        # seated on the upper legs of welded 2-block stacks). The
        # Attached chain forces the row to be fully welded before
        # seating, and the AtSite / OnBlock leg conditions force the
        # legs to actually be erected first -- without them the planner
        # happily seated the span onto legs still at their staged spots
        # and "moved" them to the sites afterwards.
        for n_span in (2, 3):
            robot = Variable("?robot", robot_type)
            span_a = Variable("?spanA", block_type)
            span_b = Variable("?spanB", block_type)
            leg_l = Variable("?legL", block_type)
            leg_r = Variable("?legR", block_type)
            if n_span == 2:
                span_vars = [span_a, span_b]
                chain = {LiftedAtom(Attached, [span_a, span_b])}
                site_l = Variable("?siteL", site_type)
                site_r = Variable("?siteR", site_type)
                leg_vars = [leg_l, leg_r, site_l, site_r]
                legs_ready = {
                    LiftedAtom(AtSite, [leg_l, site_l]),
                    LiftedAtom(AtSite, [leg_r, site_r]),
                }
            else:
                mid = Variable("?spanMid", block_type)
                span_vars = [span_a, mid, span_b]
                chain = {
                    LiftedAtom(Attached, [span_a, mid]),
                    LiftedAtom(Attached, [mid, span_b]),
                    LiftedAtom(Lying, [mid]),
                }
                base_l = Variable("?baseL", block_type)
                base_r = Variable("?baseR", block_type)
                leg_vars = [leg_l, leg_r, base_l, base_r]
                legs_ready = {
                    LiftedAtom(OnBlock, [leg_l, base_l]),
                    LiftedAtom(OnBlock, [leg_r, base_r]),
                }
            parameters = [robot] + span_vars + leg_vars
            condition = {
                LiftedAtom(Holding, [robot, span_a]),
                LiftedAtom(GlueTop, [leg_l]),
                LiftedAtom(GlueTop, [leg_r]),
                LiftedAtom(Lying, [span_a]),
                LiftedAtom(Lying, [span_b]),
                LiftedAtom(Standing, [leg_l]),
                LiftedAtom(Standing, [leg_r]),
            } | chain | legs_ready
            add_effects = {
                LiftedAtom(HandEmpty, [robot]),
                LiftedAtom(SeatedOn, [span_a, leg_l]),
                LiftedAtom(SeatedOn, [span_b, leg_r]),
                LiftedAtom(Resting, [span_a]),
            }
            seat_deletes = {
                LiftedAtom(Holding, [robot, span_a]),
                LiftedAtom(TopFree, [leg_l]),
                LiftedAtom(TopFree, [leg_r]),
            }
            processes.add(
                EndogenousProcess(f"SeatSpan{n_span}", parameters, condition,
                                  set(), set(), add_effects, seat_deletes,
                                  _delay(3.0), torch.tensor(1.0), Place,
                                  [robot], _seat_span_sampler))

        # -- Wait -------------------------------------------------------------
        robot = Variable("?robot", robot_type)
        processes.add(
            EndogenousProcess("Wait", [robot], set(), set(),
                              set(), set(), set(), ConstantDelay(1),
                              torch.tensor(1.0), Wait, [robot], null_sampler))

        # -- Exogenous cure processes (hidden dwell-time dynamics) ------------
        cure_delay = _delay(float(PyBulletBridgeEnv.cure_threshold))

        top = Variable("?top", block_type)
        bottom = Variable("?bottom", block_type)
        condition = {
            LiftedAtom(GlueTop, [bottom]),
            LiftedAtom(OnBlock, [top, bottom]),
            LiftedAtom(Resting, [top]),
            LiftedAtom(Resting, [bottom]),
        }
        processes.add(
            ExogenousProcess(
                "CureStackJoint", [top, bottom], condition, condition.copy(),
                set(), {
                    LiftedAtom(Attached, [top, bottom]),
                    LiftedAtom(Attached, [bottom, top]),
                }, {
                    LiftedAtom(GlueTop, [bottom]),
                    LiftedAtom(Loose, [top]),
                    LiftedAtom(Loose, [bottom]),
                }, cure_delay, torch.tensor(1.0)))

        right = Variable("?right", block_type)
        left = Variable("?left", block_type)
        condition = {
            LiftedAtom(GlueEndB, [left]),
            LiftedAtom(NextToEnd, [right, left]),
            LiftedAtom(Resting, [right]),
            LiftedAtom(Resting, [left]),
        }
        processes.add(
            ExogenousProcess(
                "CureLateralJoint",
                [right, left],
                condition,
                condition.copy(),
                set(),
                {
                    # ONLY the goal-order atom (goals write
                    # Attached(left, right)). Adding both orders let the
                    # planner satisfy an Attached goal via the REVERSED
                    # row arrangement and then "rearrange" the welded
                    # blocks -- a physically impossible 33-step plan.
                    # The real classifier stays symmetric, so replans
                    # (whose initial atoms come from the classifier)
                    # are unaffected.
                    LiftedAtom(Attached, [left, right]),
                },
                {
                    LiftedAtom(GlueEndB, [left]),
                    LiftedAtom(Loose, [right]),
                    LiftedAtom(Loose, [left]),
                },
                cure_delay,
                torch.tensor(1.0)))

        span = Variable("?span", block_type)
        leg = Variable("?leg", block_type)
        condition = {
            LiftedAtom(GlueTop, [leg]),
            LiftedAtom(SeatedOn, [span, leg]),
            LiftedAtom(Resting, [span]),
            LiftedAtom(Resting, [leg]),
        }
        processes.add(
            ExogenousProcess(
                "CureSeatJoint", [span, leg], condition, condition.copy(),
                set(), {
                    LiftedAtom(Attached, [span, leg]),
                    LiftedAtom(Attached, [leg, span]),
                }, {
                    LiftedAtom(GlueTop, [leg]),
                    LiftedAtom(Loose, [span]),
                    LiftedAtom(Loose, [leg]),
                }, cure_delay, torch.tensor(1.0)))

        return processes
