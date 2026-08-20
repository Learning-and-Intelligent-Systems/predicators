"""Compute K* (minimum blocks to topple a target) by simulation.

Used by the ``domino_min_block_tasks`` mode: for a task whose start (green)
and target (purple) dominoes sit a fixed distance apart, K* is the smallest
number of evenly-spaced movable (blue) dominoes for which a real robot Push on
the start topples the target *at the env's true friction*. Because the cascade
reach depends on friction, K* does too - which is exactly what makes a
high-friction (no-learning) planner under-build and fail. See
``settings.domino_min_block_tasks``.

The simulation drives the real Push option and steps PyBullet, so it mutates
the env; callers must run it before any episode (each episode re-sets state).
"""
# This module is a thin helper over env internals (state I/O, stepping, the
# domino component), so protected-member access is expected throughout.
# pylint: disable=protected-access
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from predicators.envs.pybullet_domino import geometry
from predicators.settings import CFG
from predicators.structs import Action, Object, State

# End-effector Push continuous params (approach_distance, contact_z_offset):
# the push agents actually commit to (every executed plan in the baseline /
# oracle runs used contact_z 0.05), so believed-K / true-K* reflect the
# chains agents really build. A higher probe contact (formerly 0.08) let the
# filter keep tasks whose under-built plans the agent's own weaker-push
# rollouts rejected, masking the miscalibration it was meant to expose.
_PUSH_PARAMS = np.array([0.04, 0.05], dtype=np.float32)
_OOV = 10.0  # park unused blues far out of view
_SETTLE_STEPS = 40  # no-op steps after the Push to let the cascade settle
# Canonical pushable anchor for span probes (robot reach verified in the
# calibration sweeps); spans up to ~0.35 m stay on the table from here.
_PROBE_ANCHOR = (0.55, 1.20)

# Turn-K* layout-search family (see _candidate_turn_layouts): a straight
# diagonal, a SINGLE corner blue leaning f of the way into the 90-deg
# turn (fall axis rotated toward the exit - the layout style agents
# actually build, cf. the oracle run's corner blue), and the legacy
# 45-degree pair corner. Configs (f, g1, g2) = yaw fraction, corner
# approach gap, corner-exit gap, from the canonical sim sweep
# (2026-07-03; corner parity fixed 2026-07-09, configs retained - the
# family is re-certified end-to-end by every generated task's rollout).
# The corner slides along the entry line via the entry per-gap choices -
# the "stretched corner" strategy that beats evenly-spaced L-chains.
_CORNER_CONFIGS = (
    (0.4, 0.08, 0.05),  # topples at friction 0.1
    (0.5, 0.08, 0.08),  # topples at friction 0.1
    (0.6, 0.08, 0.06),  # topples at friction 0.1
    (0.6, 0.10, 0.05),  # topples at friction 0.1
    (0.6, 0.08, 0.08),  # topples at friction 0.5
)
_ENTRY_GAPS = (0.10, 0.13, 0.15)
_MAX_GAP = 0.20  # beyond any measured topple reach at any friction
_MIN_GAP = 0.03  # bodies would spawn overlapping

# Memo for straight-span probes. Straight-chain reach is a step function
# of span, so results are reused across task attempts after bucketing the
# span to _SPAN_BUCKET - the turn tasks' per-leg certificate re-probes the
# same narrow leg bands on every attempt, and uncached this dominated
# generation time (each probe is a full robot Push rollout). Keyed on the
# component's live physical-param override so probes at the true and
# planning frictions never mix. Cleared once per generation run.
_SPAN_BUCKET = 0.01
_span_probe_memo: Dict[Tuple[Any, ...], Optional[int]] = {}


def clear_probe_memo() -> None:
    """Reset the probe memos (call at the start of a generation run)."""
    _span_probe_memo.clear()
    _dogleg_probe_memo.clear()
    _swerve_probe_memo.clear()


def _gap_ok(gap: float) -> bool:
    """Whether a per-domino spacing is physically plausible: above body overlap
    (``_MIN_GAP``) and below any measured topple reach (``_MAX_GAP``)."""
    return _MIN_GAP < gap < _MAX_GAP


def _place_even_run(comp: Any,
                    od: dict,
                    slot: int,
                    base: np.ndarray,
                    direction: np.ndarray,
                    gap: float,
                    count: int,
                    yaw: float,
                    start: int = 1) -> List[Tuple[float, float]]:
    """Place ``count`` dominoes evenly along ``direction`` and return their xy.

    Block ``i`` (of ``count``) goes at ``base + (start + i) * gap * direction``
    into ``od[comp.dominos[slot + i]]`` at ``yaw``. ``start`` is the offset of
    the first block from ``base`` in gaps (1 = one gap ahead, the common case;
    0 = at ``base`` itself). Object slots ``slot .. slot + count - 1`` are
    consumed; the caller advances its own slot cursor by ``count``.
    """
    pts: List[Tuple[float, float]] = []
    for i in range(count):
        p = base + (start + i) * gap * direction
        od[comp.dominos[slot + i]] = comp.place_domino(slot + i, float(p[0]),
                                                       float(p[1]), yaw)
        pts.append((float(p[0]), float(p[1])))
    return pts


def _feat_index(domino_type: Any, feat: str) -> int:
    return list(domino_type.feature_names).index(feat)


def _set_pose(state: State, dom: Object, x: float, y: float, z: float,
              yaw: float, roll: float) -> None:
    """Overwrite a domino's pose features in-place in ``state``."""
    arr = state.data[dom].copy()
    ti = dom.type
    arr[_feat_index(ti, "x")] = x
    arr[_feat_index(ti, "y")] = y
    arr[_feat_index(ti, "z")] = z
    arr[_feat_index(ti, "yaw")] = yaw
    arr[_feat_index(ti, "roll")] = roll
    state.data[dom] = arr


def _get_push_option(env: Any) -> Any:
    # pylint: disable=import-outside-toplevel
    from predicators.ground_truth_models import get_gt_options
    opts = get_gt_options(env.get_name())
    return next(o for o in opts if o.name == "Push")


def _ground_push(push_opt: Any, robot: Object, start: Object) -> Any:
    """Ground Push for the restricted ([robot]) or full ([robot, domino])
    variant, matching the option's declared arity."""
    if len(push_opt.types) == 1:
        return push_opt.ground([robot], _PUSH_PARAMS)
    return push_opt.ground([robot, start], _PUSH_PARAMS)


def _execute_push(env: Any,
                  state: State,
                  push_opt: Any,
                  start: Object,
                  max_steps: int = 300) -> bool:
    """Roll out the Push option against the real env; return True if it ran to
    completion.

    Returns False if the push can't be executed here - e.g. the robot
    can't reach behind this start placement (IK failure). K* treats that
    as "this k doesn't topple," so a task whose start is unpushable ends
    up with K*=None and is dropped rather than crashing task generation.
    """
    # OptionExecutionFailure is raised by the skill policy on IK/collision
    # failures; catch it so an unreachable placement is a soft miss.
    from predicators.utils import \
        OptionExecutionFailure  # pylint: disable=import-outside-toplevel
    robot = next(o for o in state if o.type.name == "robot")
    opt = _ground_push(push_opt, robot, start)
    try:
        if not opt.initiable(state):
            return False
        s: State = state
        for _ in range(max_steps):
            act = opt.policy(s)
            s = env.step(act)
            if opt.terminal(s):
                break
    except OptionExecutionFailure:
        return False
    return True


def _settle(env: Any, steps: int) -> State:
    noop = Action(np.array(env._pybullet_robot.initial_joint_positions))
    for _ in range(steps):
        env.step(noop)
    return env._get_state()


def _chain_topples(env: Any, init_state: State, start: Object, target: Object,
                   blues: List[Object], k: int, push_opt: Any) -> bool:
    """Place k evenly-spaced blues between start and target, push, and report
    whether the target topples at the env's current (true) friction."""
    comp = env._domino_component
    sx, sy = init_state.get(start, "x"), init_state.get(start, "y")
    tx, ty = init_state.get(target, "x"), init_state.get(target, "y")
    yaw = init_state.get(start, "yaw")
    z = init_state.get(start, "z")

    state = init_state.copy()
    _set_pose(state, start, sx, sy, z, yaw, 0.0)
    _set_pose(state, target, tx, ty, z, yaw, 0.0)
    for i, blue in enumerate(blues):
        if i < k:
            frac = (i + 1) / (k + 1)
            _set_pose(state, blue, sx + frac * (tx - sx),
                      sy + frac * (ty - sy), z, yaw, 0.0)
        else:
            _set_pose(state, blue, _OOV + i * 0.1, _OOV + i * 0.1, z, yaw, 0.0)

    env._set_state(state)
    # Pin the arm to its initial joints (see _layout_topples): probes
    # must not inherit arm-configuration drift from prior rollouts.
    env._pybullet_robot.set_joints(env._pybullet_robot.initial_joint_positions)
    if not _execute_push(env, state, push_opt, start):
        return False
    final = _settle(env, _SETTLE_STEPS)
    return abs(final.get(target, "roll")) >= comp.fallen_threshold


def build_turn_layout(env: Any, gen: Any, rng: Any, n1: int, n2: int,
                      gap: float) -> Optional[dict]:
    """Build an L-shaped (one 90-deg turn) domino chain and verify it topples.

    Reuses the generator's tested straight/turn geometry
    (``_place_straight_domino`` / ``_place_turn90_domino``) to lay a chain:
    start -> ``n1`` straight blues -> 90-deg turn (2 blues) -> ``n2`` straight
    blues -> target (the chain's endpoint). Assembles it, pushes the start, and
    returns a dict with the placed poses only if the endpoint topples at the
    env's true friction (so K* = number of blues is a *verified*, reach-limited
    count). Returns None if placement fails or the chain doesn't topple.

    The returned dict has: ``obj_dict`` {domino_obj: pose_dict}, ``start``,
    ``target``, ``blues`` (movable, i.e. non-start/target), ``k_star`` (len
    blues). Poses use the standard upright z.
    """
    comp = env._domino_component
    dominos = comp.dominos
    n_needed = 1 + n1 + 2 + n2 + 1  # start + seg1 + turn(2) + seg2 + target
    if n_needed > len(dominos):
        return None

    # 1) Lay the chain in a temporary obj_dict via the tested geometry.
    def _no_bounds(_a: float, _b: float) -> bool:
        return True

    sx = rng.uniform(comp.domino_x_lb, comp.domino_x_ub)
    sy = rng.uniform(comp.domino_y_lb, comp.domino_y_ub)
    rot0 = float(rng.choice([0.0, np.pi / 2, -np.pi / 2]))
    obj_dict: dict = {
        dominos[0]:
        comp.place_domino(0, sx, sy, rot0, is_start_block=True, rng=rng)
    }
    x, y, rot, dc, byaw = sx, sy, rot0, 1, None
    for _ in range(n1):
        r = gen._place_straight_domino(rng, obj_dict, x, y, rot, gap, dc,
                                       _no_bounds, None, byaw)
        x, y, rot, dc = r.x, r.y, r.rotation, r.domino_count
    r = gen._place_turn90_domino(rng,
                                 obj_dict,
                                 x,
                                 y,
                                 rot,
                                 gap,
                                 dc,
                                 n_dominos=len(dominos),
                                 n_targets=0,
                                 _in_bounds=_no_bounds,
                                 task_idx=None,
                                 should_place_target_at_end=False,
                                 block_yaw=byaw)
    if not r.success:
        return None
    x, y, rot, dc, byaw = r.x, r.y, r.rotation, r.domino_count, r.block_yaw
    for _ in range(n2):
        r = gen._place_straight_domino(rng, obj_dict, x, y, rot, gap, dc,
                                       _no_bounds, None, byaw)
        x, y, rot, dc = r.x, r.y, r.rotation, r.domino_count
    placed = [d for d in dominos if d in obj_dict]
    if len(placed) < 3:
        return None
    # Keep the chain on the table (full workspace, not the narrower reachable
    # placement band - the start is already sampled inside that band, and the
    # push simulation below filters out any chain the robot can't actually
    # push). The y placement band is very tight (~0.19 m), so requiring the
    # whole L-chain to fit it would reject almost every turn.
    for d in placed:
        px, py = obj_dict[d]["x"], obj_dict[d]["y"]
        if not (comp.x_lb < px < comp.x_ub and comp.y_lb < py < comp.y_ub):
            return None
    start, target = placed[0], placed[-1]
    blues = placed[1:-1]

    # 2) Assemble and verify the whole chain topples the endpoint.
    if not _layout_topples(env, obj_dict, start, target):
        return None
    return {
        "obj_dict": obj_dict,
        "start": start,
        "target": target,
        "blues": blues,
        "k_star": len(blues)
    }


def _layout_topples(env: Any,
                    obj_dict: dict,
                    start: Object,
                    target: Object,
                    push_opt: Any = None) -> bool:
    """Assemble ``obj_dict``, push ``start``, and report whether ``target``
    topples at the env's current friction."""
    comp = env._domino_component
    z = comp.z_lb + comp.domino_height / 2
    state = _assembled_state(env, comp, obj_dict, target, z,
                             comp.target_domino_color)
    env._set_state(state)
    # Pin the arm to its exact initial joints: _set_state reaches the
    # robot pose by IK FROM THE CURRENT configuration, so consecutive
    # probes inherit ~1e-3 rad wrist differences from wherever the
    # previous rollout parked the arm - enough to flip knife-edge
    # layouts under motion-planned pushes (observed as the same layout
    # alternating topple/no-topple). Episodes start from the initial
    # joints too, so this also matches execution-time conditions.
    env._pybullet_robot.set_joints(env._pybullet_robot.initial_joint_positions)
    joints = env._pybullet_robot.get_joints()
    from predicators.utils import \
        PyBulletState  # pylint: disable=import-outside-toplevel
    pstate = PyBulletState(state.data.copy(), simulator_state=joints)
    env._current_observation = pstate
    if push_opt is None:
        push_opt = _get_push_option(env)
    if not _execute_push(env, pstate, push_opt, start):
        return False
    final = _settle(env, _SETTLE_STEPS)
    return abs(final.get(target, "roll")) >= comp.fallen_threshold


def _assembled_state(env: Any, comp: Any, obj_dict: dict, target: Any,
                     z: float, tgt_col: Any) -> State:
    """State with the L-chain assembled and every other domino out of view."""
    from predicators.utils import \
        create_state_from_dict  # pylint: disable=import-outside-toplevel
    init_dict: dict = {env._robot: env.robot_init_state_dict()}
    for i, dom in enumerate(comp.dominos):
        if dom in obj_dict:
            d = obj_dict[dom]
            col = tgt_col if dom is target else (d["r"], d["g"], d["b"])
            init_dict[dom] = {
                "x": d["x"],
                "y": d["y"],
                "z": z,
                "yaw": d["yaw"],
                "roll": 0.0,
                "r": col[0],
                "g": col[1],
                "b": col[2],
                "is_held": 0.0
            }
        else:
            init_dict[dom] = {
                "x": _OOV + i * 0.1,
                "y": _OOV + i * 0.1,
                "z": z,
                "yaw": 0.0,
                "roll": 0.0,
                "r": 0.6,
                "g": 0.8,
                "b": 1.0,
                "is_held": 0.0
            }
    return create_state_from_dict(init_dict)


def compute_k_star(env: Any,
                   init_state: State,
                   budget: Optional[int] = None) -> Optional[int]:
    """Smallest #blues in [0, budget] whose chain topples the target, else
    None.

    ``budget`` defaults to the number of movable blues actually present
    in the task (capped by ``CFG.domino_min_block_num_blues``).
    """
    comp = env._domino_component
    if comp is None:
        return None
    dominoes = init_state.get_objects(comp.domino_type)
    # pylint: disable=protected-access
    start = next(
        (d for d in dominoes if comp._StartBlock_holds(init_state, [d])), None)
    target = next(
        (d for d in dominoes if comp._TargetDomino_holds(init_state, [d])),
        None)
    blues = [d for d in dominoes if comp._MovableBlock_holds(init_state, [d])]
    if start is None or target is None:
        return None
    if budget is None:
        budget = CFG.domino_min_block_num_blues
    budget = min(budget, len(blues))

    push_opt = _get_push_option(env)
    span = float(
        np.hypot(
            init_state.get(target, "x") - init_state.get(start, "x"),
            init_state.get(target, "y") - init_state.get(start, "y")))
    for k in range(budget + 1):
        # Geometric prune: a per-gap beyond any topple reach (or below body
        # overlap) cannot work at any friction - skip the simulation.
        gap = span / (k + 1)
        if not _gap_ok(gap):
            continue
        if _chain_topples(env, init_state, start, target, blues, k, push_opt):
            return k
    return None


def straight_span_k_star(env: Any,
                         span: float,
                         budget: Optional[int] = None) -> Optional[int]:
    """Minimum blues whose evenly-spaced STRAIGHT chain crosses ``span`` at the
    env's CURRENT friction, or None if no k within budget works.

    Probed with a real Push at a canonical pushable pose. Used by the turn
    tasks' per-leg differentiation certificate: each leg of a turn is a
    straight sub-chain, so "how many blues does this friction believe the
    leg needs" is measured by an actual rollout of a chain of that length -
    independent of any corner (whose cost is the same on both sides of the
    comparison and cancels).

    Results are memoized per (span bucket, budget, physics override); two
    spans within _SPAN_BUCKET of each other share one probe.
    """
    comp = env._domino_component
    if comp is None:
        return None
    if budget is None:
        budget = CFG.domino_min_block_num_blues
    budget = min(budget, len(comp.dominos) - 2)
    memo_key = (round(span / _SPAN_BUCKET), budget,
                tuple(sorted(comp._physical_param_override.items())))
    if memo_key in _span_probe_memo:
        return _span_probe_memo[memo_key]
    push_opt = _get_push_option(env)
    doms = comp.dominos
    sx, sy = _PROBE_ANCHOR
    syaw = np.pi / 2  # chain runs along +x
    result: Optional[int] = None
    for k in range(budget + 1):
        gap = span / (k + 1)
        if not _gap_ok(gap):
            continue
        od = {
            doms[0]:
            comp.place_domino(0, sx, sy, syaw, is_start_block=True),
            doms[1]:
            comp.place_domino(1, sx + span, sy, syaw, is_target_block=True),
        }
        _place_even_run(comp, od, 2, np.array([sx, sy]), np.array([1.0, 0.0]),
                        gap, k, syaw)
        if _layout_topples(env, od, doms[0], doms[1], push_opt):
            result = k
            break
    _span_probe_memo[memo_key] = result
    return result


def _on_table(comp: Any, pts: List[Any]) -> bool:
    """All (x, y) points inside the full table workspace."""
    return all(
        comp.x_lb < float(px) < comp.x_ub and comp.y_lb < float(py) < comp.y_ub
        for px, py in pts)


def _candidate_turn_layouts(comp: Any, k: int, start_pose: Any,
                            target_pose: Any) -> Any:
    """``_candidate_turn_layouts_labeled`` without the family label."""
    for _fam, od, start, target in _candidate_turn_layouts_labeled(
            comp, k, start_pose, target_pose):
        yield od, start, target


def _candidate_turn_layouts_labeled(comp: Any, k: int, start_pose: Any,
                                    target_pose: Any) -> Any:
    """Yield labeled candidate k-blue layouts for a cornered target.

    Each candidate is ``(family, obj_dict, start, target)`` using
    ``comp.dominos[0]`` as the (green) start, ``dominos[1]`` as the (purple)
    target and ``dominos[2:2+k]`` as blues - object identity is irrelevant
    here, only the physics matters. ``family`` names the sub-family below
    (``"straight"`` / ``"corner"`` / ``"pair"``) so probe tooling can report
    WHICH layout style topples - single natural corners are the style agents
    actually build, so a task family whose only true-physics solution is the
    knife-edge pair corner is agent-intractable. Three sub-families:

    * straight line: k blues evenly spaced from start to target with
      their fall axes ALONG the line - offered only within ~30 degrees
      of the start's push axis (near-axis targets and the k < 2 cases);
      beyond that the oblique first hit makes the cascade knife-edge;
    * corner search: k1 entry blues at per-gap ``g in _ENTRY_GAPS`` along
      the start's push line, ONE corner blue leaning f of the way into
      the turn with sim-calibrated approach/exit gaps
      (``_CORNER_CONFIGS``), then ``k2 = k - 1 - k1`` blues evenly spaced
      from the corner exit to the target. Sliding the corner via ``g``
      realises the "stretched corner" plans an optimising agent would
      find;
    * 45-degree PAIR corner: the legacy (pre-min-block) generator's turn
      construction - TWO blues stepping the yaw 45 degrees each with
      half-width inward nudges; it is what the legacy generator's solved
      turn tasks used at friction 0.5 (the default), and it redirects
      there where single-corner variants were never observed to.

    Layouts with a gap outside ``(_MIN_GAP, _MAX_GAP)`` or blocks off the
    table are pruned without simulation.
    """
    sx, sy, syaw = (float(v) for v in start_pose)
    tx, ty, tyaw = (float(v) for v in target_pose)
    start, target = comp.dominos[0], comp.dominos[1]
    s_pt = np.array([sx, sy])
    t_pt = np.array([tx, ty])
    # Push line of the start. State yaw is a CCW z-rotation
    # (getQuaternionFromEuler), so a block's true fall (thin) axis is
    # (-sin yaw, cos yaw); the samplers only produce axis-aligned starts,
    # where (sin, cos) is collinear with it, and the Push skill pushes
    # along this (sin, cos) direction.
    u_vec = np.array([np.sin(syaw), np.cos(syaw)])
    w_vec = t_pt - s_pt

    def base() -> dict:
        return {
            start: comp.place_domino(0, sx, sy, syaw, is_start_block=True),
            target: comp.place_domino(1, tx, ty, tyaw, is_target_block=True),
        }

    yield from _straight_line_layouts(comp, k, start, target, base, s_pt,
                                      u_vec, w_vec)
    if k < 2:
        return
    cross = float(u_vec[0] * w_vec[1] - u_vec[1] * w_vec[0])
    if abs(cross) < 1e-6:
        return  # target on the fall line: the straight family covers it
    t_dir = 1.0 if cross > 0 else -1.0
    yield from _corner_layouts(comp, k, start, target, base, s_pt, t_pt, u_vec,
                               syaw, t_dir)
    yield from _pair_corner_layouts(comp, k, start, target, base, s_pt, t_pt,
                                    u_vec, w_vec, syaw, t_dir)


def _straight_line_layouts(comp: Any, k: int, start: Object, target: Object,
                           base: Any, s_pt: np.ndarray, u_vec: np.ndarray,
                           w_vec: np.ndarray) -> Any:
    """Straight-line family (a): k blues evenly spaced start -> target.

    Only offered when the line stays within the measured ~33-degree
    per-knock propagation tolerance of the start's push axis (the swerve
    calibration sweep): beyond it the start's first hit is oblique and the
    cascade, when it topples at all, is contact-history knife-edge - a K*
    budget built on it is unreproducible in a fresh sim.
    """
    dist = float(np.linalg.norm(w_vec))
    gap = dist / (k + 1)
    u_dot_line = float(np.dot(w_vec, u_vec)) / max(dist, 1e-9)
    if _gap_ok(gap) and u_dot_line > np.cos(np.radians(30.0)):
        d_vec = w_vec / dist
        # Fall axis along the line: yaw = arctan2(-dx, dy) makes the thin
        # axis (-sin, cos) parallel to d_vec.
        line_yaw = geometry.yaw_along(d_vec[0], d_vec[1])
        od = base()
        pts = _place_even_run(comp, od, 2, s_pt, d_vec, gap, k, line_yaw)
        if _on_table(comp, pts):
            yield "straight", od, start, target


def _corner_layouts(comp: Any, k: int, start: Object, target: Object,
                    base: Any, s_pt: np.ndarray, t_pt: np.ndarray,
                    u_vec: np.ndarray, syaw: float, t_dir: float) -> Any:
    """Single-natural-corner family (b): k1 entry blues, ONE natural-yaw.

    corner blue, and k2 = k - 1 - k1 exit blues.

    The corner faces ``f`` of the way through the 90-deg turn (its local
    travel direction) - the agent-buildable layout style - with
    sim-calibrated approach/exit gaps (g1, g2). Sliding the corner along
    the entry line via ``g_entry`` realises the "stretched corner" plans an
    optimising agent would find.
    """
    for k1 in range(k):
        k2 = k - 1 - k1
        # g_entry only matters when there ARE entry blues to space.
        entry_gaps = _ENTRY_GAPS if k1 > 0 else _ENTRY_GAPS[:1]
        for g_entry in entry_gaps:
            for f_yaw, g1, g2 in _CORNER_CONFIGS:
                od = base()
                slot = 2
                pts = _place_even_run(comp, od, slot, s_pt, u_vec, g_entry, k1,
                                      syaw)
                slot += k1
                c_pt = s_pt + (k1 * g_entry + g1) * u_vec
                # Natural mid-turn lean: rotating the block CCW by psi
                # (yaw + psi) rotates its fall axis CCW by psi, which in
                # the compass encoding (sin, cos) is angle syaw - psi.
                psi = t_dir * f_yaw * np.pi / 2
                c_yaw = syaw + psi
                c_dir = np.array([np.sin(syaw - psi), np.cos(syaw - psi)])
                od[comp.dominos[slot]] = comp.place_domino(
                    slot, float(c_pt[0]), float(c_pt[1]), float(c_yaw))
                pts.append((c_pt[0], c_pt[1]))
                slot += 1
                if k2 == 0:
                    # Corner blue must topple the target directly.
                    if not _gap_ok(float(np.linalg.norm(t_pt - c_pt))):
                        continue
                else:
                    # First exit blue one g2 past the corner along its fall
                    # line; the rest evenly spaced toward the target.
                    b1_pt = c_pt + g2 * c_dir
                    e_vec = t_pt - b1_pt
                    e_len = float(np.linalg.norm(e_vec))
                    per = e_len / k2
                    if not _gap_ok(per):
                        continue
                    e_dir = e_vec / e_len
                    e_yaw = geometry.yaw_along(e_dir[0], e_dir[1])
                    pts += _place_even_run(comp,
                                           od,
                                           slot,
                                           b1_pt,
                                           e_dir,
                                           per,
                                           k2,
                                           e_yaw,
                                           start=0)
                    slot += k2
                if _on_table(comp, pts):
                    yield "corner", od, start, target


def _pair_corner_layouts(comp: Any, k: int, start: Object, target: Object,
                         base: Any, s_pt: np.ndarray, t_pt: np.ndarray,
                         u_vec: np.ndarray, w_vec: np.ndarray, syaw: float,
                         t_dir: float) -> Any:
    """Legacy 45-degree PAIR-corner family (c).

    d1 one corner-gap past the last entry blue ON the entry fall line, yaw
    stepped 45 degrees INTO the bend (fall axis along the mid-turn travel)
    with a half-width inward nudge; d2 one corner-gap along the 45-degree
    travel direction completing the turn (same nudge); k2 = k - 2 - k1 exit
    blues evenly to the target. Position transforms are verbatim from the
    legacy generator's turn placement. The entry per-gap is SOLVED so d2
    lands on the target's perpendicular approach line (a sweep would
    misalign the exit run).
    """
    if k < 3:
        return
    half_w = comp.domino_width / 2
    d1_dir = syaw - t_dir * np.pi / 4  # travel one step into the turn
    d1_dir_vec = np.array([np.sin(d1_dir), np.cos(d1_dir)])
    d2_rot = syaw - t_dir * np.pi / 2  # post-turn travel direction
    d1_nudge = t_dir * -half_w * np.array([np.cos(d1_dir), -np.sin(d1_dir)])
    d2_nudge = t_dir * -half_w * np.array([np.cos(d2_rot), -np.sin(d2_rot)])
    for g_c in (comp.pos_gap, 0.12):
        # Advance of the whole pair along the entry fall line.
        pair_adv = g_c * (1.0 + float(np.dot(d1_dir_vec, u_vec))) + \
            float(np.dot(d1_nudge + d2_nudge, u_vec))
        for k1 in range(1, k - 1):
            k2 = k - 2 - k1
            g_e = (float(np.dot(w_vec, u_vec)) - pair_adv) / k1
            if not _gap_ok(g_e):
                continue
            last_pt = s_pt + k1 * g_e * u_vec
            d1_pt = last_pt + g_c * u_vec + d1_nudge
            d2_pt = d1_pt + g_c * d1_dir_vec + d2_nudge
            e_vec = t_pt - d2_pt
            e_len = float(np.linalg.norm(e_vec))
            per = e_len / (k2 + 1)
            if not _gap_ok(per):
                continue
            e_dir = e_vec / e_len
            # Yaws follow the LEGACY parity exactly, so state yaw values
            # step smoothly by 45 deg per block through the turn (e.g.
            # pi/2 -> pi/4 -> 0), matching the pre-min-block generator's
            # tasks. Because state yaw is a CCW z-rotation, yaw + t*pi/4
            # rotates d1's fall axis INTO the bend (thin axis along the
            # mid-turn travel) -- the natural alignment, and the only
            # parity that redirects at friction 0.5 (2026-07-08
            # fresh-process sweep: the across-bend parity syaw - t*pi/4
            # dies in every probed configuration). d2's parity is
            # outcome-free per the same sweep.
            e_yaw = geometry.yaw_along(e_dir[0], e_dir[1])
            d1_yaw = syaw + t_dir * np.pi / 4
            d2_yaw = syaw + t_dir * np.pi / 2
            od = base()
            slot = 2
            pts = _place_even_run(comp, od, slot, s_pt, u_vec, g_e, k1, syaw)
            slot += k1
            od[comp.dominos[slot]] = comp.place_domino(slot, float(d1_pt[0]),
                                                       float(d1_pt[1]),
                                                       float(d1_yaw))
            pts.append((d1_pt[0], d1_pt[1]))
            slot += 1
            od[comp.dominos[slot]] = comp.place_domino(slot, float(d2_pt[0]),
                                                       float(d2_pt[1]),
                                                       float(d2_yaw))
            pts.append((d2_pt[0], d2_pt[1]))
            slot += 1
            pts += _place_even_run(comp, od, slot, d2_pt, e_dir, per, k2,
                                   e_yaw)
            if _on_table(comp, pts):
                yield "pair", od, start, target


# First-exit-blue gaps swept by the dogleg probe (distance from the gray
# bend link to the first exit blue along the gray's fall line) - mirrors
# the corner family's calibrated approach/exit gap treatment.
_DOGLEG_EXIT_GAPS = (0.06, 0.08, 0.10)

# Memo for dogleg probes, mirroring the straight-span memo above. Valid
# because callers probe at the canonical anchor (translation/rotation
# invariant physics, known-pushable start); keyed on the RELATIVE
# geometry plus the live physics override.
_dogleg_probe_memo: Dict[Tuple[Any, ...], Optional[int]] = {}


def heavy_dogleg_k_star(env: Any,
                        start_pose: Any,
                        target_pose: Any,
                        heavy_pose: Any,
                        budget: int,
                        only_k: Optional[int] = None) -> Optional[int]:
    """Minimum blues whose dogleg chain THROUGH the heavy (gray) block topples
    the target at the env's CURRENT physics, or None.

    The gray block stands on (or slightly off) the start's fall line and
    acts as a free bend link: k1 blues run evenly from the start to the
    gray, the chain bends at the gray, and k2 = k - k1 blues run from
    just past the gray
    (first exit gap swept over ``_DOGLEG_EXIT_GAPS``, the rest evenly to
    the target). ALL splits are tried, so the result is the best cost a
    planner could commit to within this natural family. Probed at
    whatever friction / ``block_mass`` the env currently has, so
    callers flip between the believed physics (normal mass - the chain
    runs through) and the true physics (untopple-able - the chain dies
    at the gray).

    ``only_k`` restricts the scan to that single blue count - the
    true-dead certificate uses it with k = the believed cost: a
    block-minimizing planner only ever builds a believed-cheapest
    layout, so other counts cannot leak.

    Callers should pass canonical-anchor poses (see ``_PROBE_ANCHOR``):
    results are memoized on the relative geometry, which is only sound
    where the push is known to be executable.
    """
    comp = env._domino_component
    if comp is None:
        return None
    doms = comp.dominos
    budget = min(budget, len(doms) - 3)  # gray takes the last slot
    heavy = doms[-1]
    sx, sy, syaw = (float(v) for v in start_pose)
    tx, ty, tyaw = (float(v) for v in target_pose)
    hx, hy, hyaw = (float(v) for v in heavy_pose)
    s_pt, t_pt, h_pt = (np.array([sx, sy]), np.array([tx,
                                                      ty]), np.array([hx, hy]))
    len1 = float(np.linalg.norm(h_pt - s_pt))
    len2 = float(np.linalg.norm(t_pt - h_pt))
    if min(len1, len2) <= 0:
        return None
    d1_vec = (h_pt - s_pt) / len1
    yaw1 = geometry.heading_yaw(d1_vec[0], d1_vec[1])
    h_dir = np.array([np.sin(hyaw), np.cos(hyaw)])
    bend = geometry.wrap_angle(hyaw - syaw)
    # Signed perpendicular offset of the gray from the start->target
    # line: a 2-3 cm off-line gray changes len1/len2 only at second
    # order (sub-bucket) yet bends the dogleg by ~10 degrees, so the
    # lengths alone would alias physically different lures.
    st_vec = t_pt - s_pt
    st_len = float(np.linalg.norm(st_vec))
    h_perp = 0.0 if st_len <= 0 else float(
        (st_vec[0] * (h_pt[1] - s_pt[1]) - st_vec[1] *
         (h_pt[0] - s_pt[0])) / st_len)
    memo_key = (round(len1 / _SPAN_BUCKET), round(len2 / _SPAN_BUCKET),
                round(h_perp / _SPAN_BUCKET), round(bend, 2), budget, only_k,
                tuple(sorted(comp._physical_param_override.items())))
    if memo_key in _dogleg_probe_memo:
        return _dogleg_probe_memo[memo_key]
    push_opt = _get_push_option(env)
    result: Optional[int] = None
    k_values = range(budget + 1) if only_k is None else [min(only_k, budget)]
    for k in k_values:
        for k1 in range(k + 1):
            k2 = k - k1
            gap1 = len1 / (k1 + 1)
            if not _gap_ok(gap1):
                continue
            exit_gaps: Tuple[Optional[float], ...] = \
                _DOGLEG_EXIT_GAPS if k2 > 0 else (None,)
            for g2 in exit_gaps:
                od = {
                    doms[0]:
                    comp.place_domino(0, sx, sy, syaw, is_start_block=True),
                    doms[1]:
                    comp.place_domino(1, tx, ty, tyaw, is_target_block=True),
                    heavy:
                    comp.place_domino(0, hx, hy, hyaw, is_heavy_block=True),
                }
                slot = 2
                _place_even_run(comp, od, slot, s_pt, d1_vec, gap1, k1, yaw1)
                slot += k1
                if k2 == 0:
                    # The gray itself must reach the target.
                    if not _gap_ok(len2):
                        continue
                else:
                    assert g2 is not None
                    b1_pt = h_pt + g2 * h_dir
                    e_vec = t_pt - b1_pt
                    e_len = float(np.linalg.norm(e_vec))
                    per = e_len / k2
                    if not _gap_ok(per):
                        continue
                    e_dir = e_vec / e_len
                    e_yaw = geometry.yaw_along(e_dir[0], e_dir[1])
                    _place_even_run(comp,
                                    od,
                                    slot,
                                    b1_pt,
                                    e_dir,
                                    per,
                                    k2,
                                    e_yaw,
                                    start=0)
                    slot += k2
                if _layout_topples(env, od, doms[0], doms[1], push_opt):
                    result = k
                    break
            if result is not None:
                break
        if result is not None:
            break
    _dogleg_probe_memo[memo_key] = result
    return result


# Sideways displacements of the detour's own corner from the gray block,
# swept by ``_candidate_detour_layouts``. Calibrated by the 2026-07-25
# placement-noise probe on the two shipping turn geometries: at 0.08-0.10
# the doglegs topple 9-10/10 under 5 mm / 0.02 rad blue-pose noise; 0.06
# is marginal (5-8/10) but occasionally the only fit on short legs.
_DETOUR_CORNER_OFFSETS = (0.06, 0.08, 0.10)


def _candidate_detour_layouts(comp: Any, k: int, start_pose: Any,
                              target_pose: Any, heavy_pose: Any) -> Any:
    """Yield k-blue displaced-corner doglegs AROUND the gray block.

    One blue is an own corner standing at the gray's position displaced
    sideways off the start->target line (both sides, offsets from
    ``_DETOUR_CORNER_OFFSETS``); the remaining ``k - 1`` blues split over
    the two legs (all splits), falling along their leg, with the corner
    blue leaning between the leg headings. This is the robust detour
    family for heavy TURN tasks: the pre-existing corner family is
    obstacle-cramped around the gray (most candidates pruned, survivors
    knife-edge), while these doglegs pass placement-noise replays (see
    ``heavy_detour_k_star_robust``).
    """
    if k < 1:
        return
    sx, sy, syaw = (float(v) for v in start_pose)
    tx, ty, tyaw = (float(v) for v in target_pose)
    hx, hy, hyaw = (float(v) for v in heavy_pose)
    s_pt = np.array([sx, sy])
    t_pt = np.array([tx, ty])
    h_pt = np.array([hx, hy])
    st_vec = t_pt - s_pt
    st_len = float(np.linalg.norm(st_vec))
    if st_len <= 0:
        return
    n_vec = np.array([-st_vec[1], st_vec[0]]) / st_len
    start, target = comp.dominos[0], comp.dominos[1]
    heavy = comp.dominos[-1]
    for d_off in _DETOUR_CORNER_OFFSETS:
        for side in (1.0, -1.0):
            c_pt = h_pt + side * d_off * n_vec
            len1 = float(np.linalg.norm(c_pt - s_pt))
            len2 = float(np.linalg.norm(t_pt - c_pt))
            if min(len1, len2) <= 1e-6:
                continue
            d1_vec = (c_pt - s_pt) / len1
            d2_vec = (t_pt - c_pt) / len2
            mid = d1_vec + d2_vec
            mid_len = float(np.linalg.norm(mid))
            if mid_len < 1e-6:
                continue
            yaw1 = geometry.yaw_along(d1_vec[0], d1_vec[1])
            yaw2 = geometry.yaw_along(d2_vec[0], d2_vec[1])
            cyaw = geometry.yaw_along(mid[0] / mid_len, mid[1] / mid_len)
            for k1 in range(k):
                k2 = k - 1 - k1
                gap1 = len1 / (k1 + 1)
                gap2 = len2 / (k2 + 1)
                if not (_gap_ok(gap1) and _gap_ok(gap2)):
                    continue
                od = {
                    start:
                    comp.place_domino(0, sx, sy, syaw, is_start_block=True),
                    target:
                    comp.place_domino(1, tx, ty, tyaw, is_target_block=True),
                    heavy:
                    comp.place_domino(0, hx, hy, hyaw, is_heavy_block=True),
                }
                pts = []
                slot = 2
                for i in range(k1):
                    pt = s_pt + gap1 * (i + 1) * d1_vec
                    od[comp.dominos[slot]] = comp.place_domino(
                        slot, float(pt[0]), float(pt[1]), yaw1)
                    pts.append((float(pt[0]), float(pt[1])))
                    slot += 1
                od[comp.dominos[slot]] = comp.place_domino(
                    slot, float(c_pt[0]), float(c_pt[1]), cyaw)
                pts.append((float(c_pt[0]), float(c_pt[1])))
                slot += 1
                for i in range(k2):
                    pt = c_pt + gap2 * (i + 1) * d2_vec
                    od[comp.dominos[slot]] = comp.place_domino(
                        slot, float(pt[0]), float(pt[1]), yaw2)
                    pts.append((float(pt[0]), float(pt[1])))
                    slot += 1
                if any(
                        np.hypot(px - hx, py - hy) < 0.05  # body clearance
                        for px, py in pts):
                    continue
                if not _on_table(comp, pts):
                    continue
                yield od


def _layout_noise_robust(env: Any,
                         comp: Any,
                         od: Any,
                         push_opt: Any,
                         seed: int,
                         trials: int = 6,
                         min_ok: int = 5,
                         sig_xy: float = 0.005,
                         sig_yaw: float = 0.02) -> bool:
    """Whether a toppling layout survives blue-pose placement noise.

    Replays the layout ``trials`` times with each blue's (x, y, yaw)
    jittered by Gaussian noise at the real Place skill's residual
    scatter scale, requiring >= ``min_ok`` topples. The noise rng is
    seeded deterministically (from ``seed``) so generation stays
    reproducible. Early-exits both ways once the verdict is decided.
    """
    start, target = comp.dominos[0], comp.dominos[1]
    heavy = comp.dominos[-1]
    rng = np.random.default_rng(1_000_003 + seed)
    ok = 0
    for t in range(trials):
        if ok >= min_ok:
            break
        if ok + (trials - t) < min_ok:
            break
        noisy = {o: dict(d) for o, d in od.items()}
        for o, d in noisy.items():
            if o in (start, target, heavy):
                continue
            d["x"] += float(rng.normal(0, sig_xy))
            d["y"] += float(rng.normal(0, sig_xy))
            d["yaw"] += float(rng.normal(0, sig_yaw))
        if _layout_topples(env, noisy, start, target, push_opt):
            ok += 1
    return ok >= min_ok


def heavy_detour_k_star_robust(env: Any,
                               start_pose: Any,
                               target_pose: Any,
                               heavy_pose: Any,
                               budget: int,
                               min_hits: int = 2) -> Optional[int]:
    """Minimum blues whose detour AROUND the gray block topples the target at
    the env's CURRENT physics AND survives placement noise, or None.

    The robust replacement for certifying heavy TURN tasks via
    ``compute_turn_k_star(..., extra=gray)``: nominal-only certification
    accepts knife-edge layouts (2026-07-25 probe on a shipping task: the
    family collapses to one winner that topples only 5/20 under 5 mm /
    0.02 rad blue-pose noise), which both flip under contact-solver
    history perturbations - the same task generating or not depending on
    unrelated code changes - and are unbuildable by the real Place skill
    (~1-2 cm pose-dependent settle scatter). A hit here must topple
    nominally AND pass ``_layout_noise_robust``; ``min_hits`` such
    layouts are required (accumulated across k <= budget) before the
    first (cheapest) toppling k is returned.

    Candidates are the agent-style corner family (gray merged in as a
    pruning obstacle, as before) plus the displaced-corner dogleg family
    ``_candidate_detour_layouts`` - the family the noise gate actually
    passes; without it certification would go from knife-edge to
    near-impossible instead of robust.
    """
    comp = env._domino_component
    if comp is None:
        return None
    budget = min(budget, len(comp.dominos) - 3)
    if budget < 1:
        return None
    push_opt = _get_push_option(env)
    hx, hy, _ = (float(v) for v in heavy_pose)
    clearance = comp.domino_width
    heavy = comp.dominos[-1]
    start, target = comp.dominos[0], comp.dominos[1]
    first_k: Optional[int] = None
    hits = 0
    cand_idx = 0
    for k in range(1, budget + 1):
        merged = []
        for od, s_, t_ in _candidate_turn_layouts(comp, k, start_pose,
                                                  target_pose):
            blue_pts = [(d["x"], d["y"]) for o, d in od.items()
                        if o not in (s_, t_)]
            if any(
                    np.hypot(bx - hx, by - hy) < clearance
                    for bx, by in blue_pts):
                continue
            od[heavy] = comp.place_domino(0, *heavy_pose, is_heavy_block=True)
            merged.append(od)
        merged.extend(
            _candidate_detour_layouts(comp, k, start_pose, target_pose,
                                      heavy_pose))
        for od in merged:
            cand_idx += 1
            if not _layout_topples(env, od, start, target, push_opt):
                continue
            if not _layout_noise_robust(env, comp, od, push_opt, cand_idx):
                continue
            hits += 1
            if first_k is None:
                first_k = k
            if hits >= min_hits:
                return first_k
    return None


# Peak swerve headings (off the start->target line) scanned by the
# half-circle family. Calibrated by the 2026-07-03 sweep: per-knock
# heading changes stay within the ~33-degree propagation tolerance at
# phi <= 40 with 3-4 blues, and the peak lateral offset clears the gray
# block from ~30 degrees up.
_SWERVE_PHIS = (25.0, 30.0, 35.0, 40.0)

# Memo for swerve probes; keyed on the rounded ABSOLUTE pose as well as
# the relative geometry, so canonical-anchor probes are shared across
# attempts while real-pose re-verifications get their own entries.
_swerve_probe_memo: Dict[Tuple[Any, ...], Optional[int]] = {}


def _candidate_swerve_layouts(comp: Any, k: int, start_pose: Any,
                              target_pose: Any, heavy_pose: Any) -> Any:
    """Yield k-blue "half-circle" swerves around a near-line gray block.

    The heavy block sits on (or slightly off) the segment from start to
    target, facing along it. Each candidate follows the heading profile
    m_i = phi * sin(2*pi*(i+0.5)/(k+1)): aligned with the line at both
    ends (head-on first knock, head-on target hit), bulging sideways
    mid-path to clear the gray block, with net lateral displacement
    ~zero. Both sides and all ``_SWERVE_PHIS`` peaks are scanned.
    """
    sx, sy, syaw = (float(v) for v in start_pose)
    tx, ty, _ = (float(v) for v in target_pose)
    hx, hy, hyaw = (float(v) for v in heavy_pose)
    dominos = comp.dominos
    s_pt = np.array([sx, sy])
    t_pt = np.array([tx, ty])
    span = float(np.linalg.norm(t_pt - s_pt))
    if span <= 0 or k < 2:
        return
    u_vec = (t_pt - s_pt) / span
    line_yaw = geometry.heading_yaw(u_vec[0], u_vec[1])
    steps = k + 1
    for phi_deg in _SWERVE_PHIS:
        for side in (1.0, -1.0):
            m_arr = np.radians(phi_deg) * side * np.sin(
                2 * np.pi * (np.arange(steps) + 0.5) / steps)
            gap = span / float(np.sum(np.cos(m_arr)))
            if not _gap_ok(gap):
                continue
            od = {
                dominos[0]:
                comp.place_domino(0, sx, sy, syaw, is_start_block=True),
                dominos[1]:
                comp.place_domino(1, tx, ty, line_yaw, is_target_block=True),
                dominos[-1]:
                comp.place_domino(0, hx, hy, hyaw, is_heavy_block=True),
            }
            pos = s_pt.copy()
            pts = []
            for i in range(k):
                yaw_i = line_yaw - float(m_arr[i])
                pos = pos + gap * np.array([np.sin(yaw_i), np.cos(yaw_i)])
                od[dominos[2 + i]] = comp.place_domino(2 + i, float(pos[0]),
                                                       float(pos[1]), yaw_i)
                pts.append((float(pos[0]), float(pos[1])))
            # Prune candidates that would spawn a blue inside the gray.
            if any(
                    np.hypot(px - hx, py - hy) < 0.05  # body clearance
                    for px, py in pts):
                continue
            if _on_table(comp, pts):
                yield od


def swerve_k_star(env: Any,
                  start_pose: Any,
                  target_pose: Any,
                  heavy_pose: Any,
                  budget: int,
                  min_hits: int = 1) -> Optional[int]:
    """Minimum blues whose half-circle swerve AROUND the near-line gray block
    topples the target at the env's CURRENT physics, or None.

    The constructive counterpart of ``heavy_dogleg_k_star``'s straight
    lure: same start/target line, but the chain leaves the line, clears
    the gray block sideways, and rejoins to hit the target head-on.
    Results are memoized (pose included in the key, so canonical-anchor
    certification and real-pose re-verification never mix).
    ``min_hits`` demands that many distinct toppling swerves before the
    first (cheapest) k is returned - the robustness margin against
    solver-history sensitivity (see ``compute_turn_k_star``).
    """
    comp = env._domino_component
    if comp is None:
        return None
    doms = comp.dominos
    budget = min(budget, len(doms) - 3)
    sx, sy, syaw = (float(v) for v in start_pose)
    tx, ty, _ = (float(v) for v in target_pose)
    hx, hy, _ = (float(v) for v in heavy_pose)
    # Signed perpendicular offset of the gray from the start->target
    # line (see ``heavy_dogleg_k_star``): distances alone alias
    # off-line grays, which need different swerve depths per side.
    st_len = float(np.hypot(tx - sx, ty - sy))
    h_perp = 0.0 if st_len <= 0 else float(
        ((tx - sx) * (hy - sy) - (ty - sy) * (hx - sx)) / st_len)
    memo_key = (round(sx / _SPAN_BUCKET), round(sy / _SPAN_BUCKET),
                round(syaw,
                      2), round(np.hypot(tx - sx, ty - sy) / _SPAN_BUCKET),
                round(np.hypot(hx - sx, hy - sy) / _SPAN_BUCKET),
                round(h_perp / _SPAN_BUCKET), budget, min_hits,
                tuple(sorted(comp._physical_param_override.items())))
    if memo_key in _swerve_probe_memo:
        return _swerve_probe_memo[memo_key]
    push_opt = _get_push_option(env)
    result: Optional[int] = None
    hits = 0
    for k in range(2, budget + 1):
        for od in _candidate_swerve_layouts(comp, k, start_pose, target_pose,
                                            heavy_pose):
            if _layout_topples(env, od, doms[0], doms[1], push_opt):
                hits += 1
                if result is None:
                    result = k
                if hits >= min_hits:
                    break
        if hits >= min_hits:
            break
    if hits < min_hits:
        result = None
    _swerve_probe_memo[memo_key] = result
    return result


def compute_turn_k_star(env: Any,
                        start_pose: Any,
                        target_pose: Any,
                        budget: Optional[int] = None,
                        extra: Optional[dict] = None,
                        min_hits: int = 1) -> Optional[int]:
    """Minimum blues that topple a cornered target, by layout search.

    For each k (ascending), simulates every candidate in
    :func:`_candidate_turn_layouts` at the env's CURRENT friction and returns
    the first k with a toppling layout, else ``None``. Unlike the straight
    :func:`compute_k_star`, even spacing is not optimal around a corner -
    sliding the corner toward the start ("stretched corner") can beat the
    evenly-spaced L by a block - so K* must be a minimum over layouts, not a
    count of one constructed chain. The family contains only AGENT-BUILDABLE
    layouts (straight lines, single natural corners, and the legacy
    45-degree pair corner, see :func:`_candidate_turn_layouts`). The
    family is coarse, so the result is an upper bound on the true
    minimum over that natural class.

    ``start_pose`` = (x, y, yaw) of the green start block, ``target_pose`` =
    (x, y, yaw) of the purple target. ``budget`` caps k (default
    ``CFG.domino_min_block_num_blues``).

    ``extra`` optionally maps additional dominoes (e.g. a heavy gray
    obstacle, using slots ABOVE the blues') to their pose dicts; they are
    merged into every candidate scene, and candidates that would spawn a
    blue overlapping an extra body are pruned without simulation
    (PyBullet resolves spawn penetration explosively, which would fake a
    topple).

    ``min_hits`` is a ROBUSTNESS margin: the scan keeps going until that
    many distinct layouts (across all k <= budget) have toppled, and
    returns the first (cheapest) k only then. Knife-edge layouts are
    sensitive to the simulator's contact-solver history - a task whose
    only solution toppled once during generation can be unsolvable
    under a fresh simulator - so shipping tasks should demand >= 2
    independent topplers.
    """
    comp = env._domino_component
    if comp is None:
        return None
    if budget is None:
        budget = CFG.domino_min_block_num_blues
    n_extra = len(extra) if extra else 0
    budget = min(budget, len(comp.dominos) - 2 - n_extra)
    if budget < 0:
        return None
    extra_pts = [(d["x"], d["y"]) for d in extra.values()] if extra else []
    clearance = comp.domino_width
    push_opt = _get_push_option(env)
    first_k: Optional[int] = None
    hits = 0
    for k in range(budget + 1):
        for od, start, target in _candidate_turn_layouts(
                comp, k, start_pose, target_pose):
            if extra:
                blue_pts = [(d["x"], d["y"]) for dom, d in od.items()
                            if dom not in (start, target)]
                if any(
                        np.hypot(bx - ex, by - ey) < clearance
                        for bx, by in blue_pts for ex, ey in extra_pts):
                    continue
                od.update(extra)
            if _layout_topples(env, od, start, target, push_opt):
                hits += 1
                if first_k is None:
                    first_k = k
                if hits >= min_hits:
                    return first_k
    return None


def dual_valid_turn_layout_exists(env: Any, start_pose: Any, target_pose: Any,
                                  k: int, believed_friction: float,
                                  true_friction: float) -> bool:
    """True if some k-blue turn candidate topples at BOTH frictions.

    The differentiation property a mismatch task actually needs is that
    every believed-valid plan within the K* budget FAILS under true
    physics (by dying - over-builds fail on budget by themselves). The
    scalar comparison ``believed K* > true K*`` is sufficient but
    contact-history knife-edge: a fresh sim can find a k_true-blue
    believed corner where generation probed k_true + 1. Candidates with
    FEWER than k_true blues die at the true friction by K*'s definition,
    so only the k = k_true layer can hide a believed-valid plan that
    also truly succeeds - this scans exactly that layer.
    """
    comp = env._domino_component
    if comp is None:
        return False
    push_opt = _get_push_option(env)
    found = False
    try:
        for od, start, target in _candidate_turn_layouts(
                comp, k, start_pose, target_pose):
            env.set_domino_physical_params(lateral_friction=believed_friction)
            if not _layout_topples(env, od, start, target, push_opt):
                continue
            env.set_domino_physical_params(lateral_friction=true_friction)
            if _layout_topples(env, od, start, target, push_opt):
                found = True
                break
    finally:
        env.set_domino_physical_params(lateral_friction=true_friction)
    return found
