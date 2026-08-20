"""Min-block / system-ID task generation for the domino environment.

Builds the reach-limited "minimum-blocks" tasks: start/target pairs whose
gap sits near the topple-reach limit, each carrying a ``DominoEvaluator``
plus an env-side-only ``offline_task_metrics["k_star"]`` recording the
simulated minimum number of movable blues that topple the target at the
true friction (metrics only - K* never enters the success criterion or
anything agent-reachable). A differentiation filter keeps only
tasks that separate a friction-calibrated planner from a miscalibrated
one. Finished tasks are cached on disk, keyed by config + seed + a source
digest.

Every function takes the composed domino env as its first argument; this
module owns the generation pipeline while the env owns the physics and
reward semantics (``DominoEvaluator``, ``count_movable_blocks_used``).
"""

import contextlib
import functools
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, \
    Optional, Tuple

import numpy as np

from predicators.envs.pybullet_domino import geometry
from predicators.envs.pybullet_domino.task_generators import goal_text
from predicators.envs.pybullet_domino.task_generators import \
    min_block_utils as mbu
from predicators.envs.pybullet_domino.task_generators.min_block_utils import \
    _PROBE_ANCHOR, clear_probe_memo, compute_k_star, compute_turn_k_star, \
    dual_valid_turn_layout_exists, heavy_dogleg_k_star, straight_span_k_star, \
    swerve_k_star
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, GroundAtom, Object, State

if TYPE_CHECKING:
    from predicators.envs.pybullet_domino.env import PyBulletDominoComposedEnv
    # pylint: disable-next=line-too-long
    from predicators.envs.pybullet_domino.task_generators.domino_task_generator import \
        DominoTaskGenerator


@functools.lru_cache(maxsize=1)
def _domino_code_digest() -> str:
    """Digest of the source code that determines min-block task generation.

    Covers the domino env package (task generators, K* search, the env)
    and the domino skills (push geometry feeds the simulated K*). Any
    edit to these files changes the digest and invalidates cached tasks.
    """
    import predicators  # pylint: disable=import-outside-toplevel
    base = Path(predicators.__file__).parent
    digest = hashlib.sha256()
    for rel in ("envs/pybullet_domino", "ground_truth_models/domino",
                "ground_truth_models/skill_factories"):
        for source in sorted((base / rel).rglob("*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    return digest.hexdigest()


# Default L-shape leg-sampling bands (entry_leg, exit_leg) for turn tasks
# when no explicit differentiating band is configured - the natural-corner
# region shared by the plain and heavy turn variants. Explicit differentiating
# bands come from CFG.domino_min_block_turn_* (probe with
# scripts/domino_debug/probe_min_block_bands.py when the friction pair moves).
_DEFAULT_TURN_ENTRY_BAND = (0.26, 0.34)
_DEFAULT_TURN_EXIT_BAND = (0.18, 0.26)
# Heavy turn variant: legs from the 2026-07-25 canonical-anchor design
# grid, where the whole certificate chain holds together: the believed
# corner blueprint exists at k_full=3 (entry blue + corner + exit blue,
# so the gray-corner lure costs k_bel=2), the lure propagates believedly
# and dies truly, AND the noise-robust skip-around detour costs k=3 -
# STRICTLY dearer than the lure. Strictness matters: a noise-robust
# gray-free detour at the lure's own cost would let a block-minimizing
# believed planner tie-break AWAY from the gray and solve with no
# learning (the maker enforces this with the k_true <= k_bel drop; the
# bands only control how often a sample certifies). Shorter chords
# admit robust 2-blue detours (tie), longer ones lose the blueprint,
# and exits past ~0.20 make the believed LURE itself knife-edge (the
# gray's believed fall barely relays the long exit) - every grid edge
# fails some certificate leg, so keep the bands inside the verified
# window.
_HEAVY_TURN_ENTRY_BAND = (0.26, 0.29)
_HEAVY_TURN_EXIT_BAND = (0.17, 0.20)
# Heavy straight variant: start->target span, the gray block's fraction
# along that span, and its small signed perpendicular offset OFF the
# line (sign sampled). An off-line gray still lures the believed
# planner (the dogleg family bends through it) but shrinks the swerve
# depth needed on the far side, so the true solution's arc is gentler.
_HEAVY_SPAN_BAND = (0.36, 0.44)
_HEAVY_GRAY_FRACTIONS = (0.45, 0.5, 0.55)
_HEAVY_GRAY_LATERAL_OFFSETS = (0.02, 0.03)


def _early_stop_bar(k_star: float) -> float:
    """Reward of an optimal legitimate solve: the early-stopping bar.

    Mirrors ``DominoEvaluator.reward`` at ``k_used == K*``, so a
    training episode counts toward early stopping only when it solves
    with the oracle-minimal block count (modulo the run's configured
    slack). Computed at task-construction time, never cached, so
    ``domino_block_cost`` stays a run-time knob.
    """
    return 1.0 - CFG.domino_block_cost * k_star


@contextlib.contextmanager
def _believed_physics(env: "PyBulletDominoComposedEnv",
                      believed_heavy_mass: bool = False) -> Iterator[None]:
    """Temporarily switch the env to the planner's believed physics.

    Sets the domino friction to ``CFG.domino_planning_friction`` (when a
    planning mismatch is configured) and, when ``believed_heavy_mass``, the
    gray block to a normal domino mass (an ordinary chain link). Restores the
    true friction - and, when the heavy mass was overridden, the true heavy
    mass - on exit. Used both for the straight/turn planning-K* probes and for
    the heavy-block lure probes.
    """
    comp = env._domino_component  # pylint: disable=protected-access
    assert comp is not None
    believed: Dict[str, float] = {}
    if CFG.domino_planning_friction is not None:
        believed["lateral_friction"] = CFG.domino_planning_friction
    restore: Dict[str, float] = {"lateral_friction": CFG.domino_true_friction}
    if believed_heavy_mass:
        believed["block_mass"] = comp.domino_mass
        restore["block_mass"] = comp.heavy_block_true_mass
    if believed:
        env.set_domino_physical_params(**believed)
    try:
        yield
    finally:
        env.set_domino_physical_params(**restore)


def make_min_block_tasks(env: "PyBulletDominoComposedEnv",
                         generator: "DominoTaskGenerator",
                         generate_batch: Callable[[int],
                                                  List[EnvironmentTask]],
                         num_tasks: int, rng: np.random.Generator,
                         cache_tag: str,
                         turn_ratio: float) -> List[EnvironmentTask]:
    """Generate (or reload) the min-block task set.

    Min-block generation is expensive (each kept task runs simulated K*
    searches), but fully deterministic given the seed, the config, and
    the code - so finished tasks are cached and reloaded on repeat runs.

    A fraction (``turn_ratio``, from ``domino_{train,test}_turn_ratio``
    per task set) of tasks are L-shaped
    (one 90-degree domino turn), the rest straight. Both drop tasks that
    can't be pushed / don't topple, so the quota loop keeps going until
    enough of each survive (or the attempt cap is hit). ``rng`` is
    stateful, so each attempt yields fresh tasks.

    In heavy-block mode (``domino_heavy_block_tasks``) the mix instead
    comes from the two natural-alignment heavy variants - straight tasks
    (gray block dead ahead on the line, solved by a half-circle swerve)
    fill the straight quota and turn tasks (gray block at the L's
    natural corner, solved by skipping around it) fill the turn quota;
    the quota loop, cache, and reward machinery are shared.
    """
    cache_path = _min_block_cache_path(env, cache_tag, num_tasks)
    cached = _load_min_block_cache(env, cache_path)
    if cached is not None:
        return cached

    clear_probe_memo()
    _corner_blueprint_memo.clear()
    maker_t = Callable[[
        "PyBulletDominoComposedEnv", "DominoTaskGenerator", np.random.Generator
    ], Optional[EnvironmentTask]]
    turn_maker: maker_t
    straight_maker: Optional[maker_t] = None
    n_turn = int(round(num_tasks * turn_ratio))
    n_straight = num_tasks - n_turn
    if CFG.domino_heavy_block_tasks:
        turn_maker = _make_heavy_turn_task
        straight_maker = _make_heavy_straight_task
    else:
        turn_maker = _make_turn_task
    turns: List[EnvironmentTask] = []
    straights: List[EnvironmentTask] = []
    # The differentiating span/leg windows are narrow relative to the
    # sampling bands (~25% of straight attempts survive the filters, and
    # turn attempts only ~1-in-30: the per-leg certificate's dead band
    # dominates - the 2026-07-03 oracle run got 1 turn from 34 attempts
    # and shipped a 4/5 PARTIAL set at the old 12x+20 cap). Rejected
    # attempts are nearly free (memoized probes), so the cap is
    # generous; results are cached.
    max_attempts = 36 * num_tasks + 40
    for _ in range(max_attempts):
        if len(turns) >= n_turn and len(straights) >= n_straight:
            break
        if len(turns) < n_turn:
            turn_task = turn_maker(env, generator, rng)
            if turn_task is not None:
                turns.extend(
                    env._add_pybullet_state_to_tasks(  # pylint: disable=protected-access
                        [turn_task]))
        if len(straights) < n_straight:
            if straight_maker is not None:
                straight_task = straight_maker(env, generator, rng)
                if straight_task is not None:
                    straights.extend(
                        env._add_pybullet_state_to_tasks(  # pylint: disable=protected-access
                            [straight_task]))
            else:
                batch = env._add_pybullet_state_to_tasks(  # pylint: disable=protected-access
                    generate_batch(1))
                straights.extend(_assign_min_blocks(env, batch))
    survivors = turns[:n_turn] + straights[:n_straight]
    if len(survivors) < num_tasks:
        logging.warning(
            "Min-block: generated only %d/%d tasks (%d turn, %d straight) "
            "after %d attempts; widen the span/gap bands or raise "
            "domino_min_block_num_blues.", len(survivors), num_tasks,
            len(turns), len(straights), max_attempts)
    _save_min_block_cache(cache_path, survivors, num_tasks)
    return survivors


# ── Min-block task cache ─────────────────────────────────────

# Flags matched by the cache key's prefixes that only affect RENDERING -
# they cannot change generation physics or the tasks themselves, so they
# are excluded from the key (a camera-resolution change must not orphan
# a 20-minute generation cache).
_RENDER_ONLY_FLAGS = frozenset({
    "pybullet_camera_width",
    "pybullet_camera_height",
    "pybullet_draw_debug",
})


def _min_block_cache_path(env: "PyBulletDominoComposedEnv", cache_tag: str,
                          num_tasks: int) -> Optional[Path]:
    """Cache file for this (config, seed, code) combination, or None.

    The key hashes every ``domino_``/``pybullet_``/``skill_phase_`` CFG
    flag (except the render-only ones above), the seed and task counts,
    AND a digest of the domino env + domino skill source code - so any
    change to the physics config or the generation/skill code
    automatically invalidates the cache.
    """
    cache_dir = CFG.domino_min_block_task_cache_dir
    if not cache_dir or not cache_tag:
        return None
    cfg_items = {}
    for name in dir(CFG):
        if name.startswith(("domino_", "pybullet_", "skill_phase_")) \
                and name not in _RENDER_ONLY_FLAGS:
            value = getattr(CFG, name)
            if not callable(value):
                cfg_items[name] = value
    blob = json.dumps([
        env.get_name(), cache_tag, num_tasks, CFG.seed, cfg_items,
        _domino_code_digest()
    ],
                      sort_keys=True,
                      default=str)
    key = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"{cache_tag}_{key}.json"


def _load_min_block_cache(
        env: "PyBulletDominoComposedEnv",
        path: Optional[Path]) -> Optional[List[EnvironmentTask]]:
    """Rebuild cached tasks, or None on a cache miss."""
    # pylint: disable=import-outside-toplevel,protected-access
    from predicators.envs.pybullet_domino.env import DominoEvaluator
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        num_requested = raw["num_requested"]
        entries = raw["tasks"]
    else:  # legacy format: bare task list, request size unknown
        num_requested, entries = None, raw
    pred_map = {p.name: p for p in env.predicates}
    # Map names to the env's LIVE object instances: they carry the
    # PyBullet body ids that state I/O needs (fresh Object()s would
    # compare equal but have id=None).
    live_objs = {env._robot.name: env._robot}
    for comp in env._components:
        for obj in comp.get_objects():
            live_objs[obj.name] = obj
    tasks: List[EnvironmentTask] = []
    for entry in entries:
        objs = {name: live_objs[name] for name, _tname in entry["objects"]}
        state = State({
            objs[name]: np.array(vals, dtype=np.float64)
            for name, vals in entry["data"].items()
        })
        goal = {
            GroundAtom(pred_map[pname], [objs[oname] for oname in onames])
            for pname, onames in entry["goal"]
        }
        k_star = entry["k_star"]
        plain = EnvironmentTask(
            state,
            goal,
            goal_nl=entry["goal_nl"],
            evaluator=DominoEvaluator(goal) if k_star is not None else None,
            offline_task_metrics=({
                "k_star": float(k_star)
            } if k_star is not None else {}),
            early_stop_min_reward=(_early_stop_bar(float(k_star))
                                   if k_star is not None else None))
        # Re-run the standard PyBullet conversion (joints, optional
        # rendering) instead of caching simulator state.
        tasks.extend(env._add_pybullet_state_to_tasks([plain]))
    if num_requested is not None and len(tasks) < num_requested:
        logging.warning(
            "Min-block: cache %s holds a PARTIAL set (%d/%d tasks - the "
            "generating run hit its attempt cap). Runs will evaluate on "
            "the reduced set; delete the file to retry generation, or "
            "widen the span/gap bands.", path, len(tasks), num_requested)
    logging.info("Min-block: loaded %d cached tasks from %s.", len(tasks),
                 path)
    return tasks


def _save_min_block_cache(path: Optional[Path], tasks: List[EnvironmentTask],
                          num_requested: int) -> None:
    """Serialize finished tasks (init data, goal, K*) to the cache.

    Only the offline K* is stored; the ``DominoEvaluator`` is rebuilt on
    load. ``num_requested`` is stored alongside so a partial set (the
    quota loop hit its attempt cap) is flagged loudly on every reload
    instead of silently shrinking the eval.
    """
    if path is None:
        return
    payload = []
    for env_task in tasks:
        init = env_task.init
        k_star = env_task.offline_task_metrics.get("k_star")
        payload.append({
            "objects": [(o.name, o.type.name) for o in init],
            "data": {o.name: [float(v) for v in init.data[o]]
                     for o in init},
            "goal": [(a.predicate.name, [o.name for o in a.objects])
                     for a in env_task.goal],
            "goal_nl":
            env_task.goal_nl,
            "k_star":
            k_star,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "num_requested": num_requested,
            "tasks": payload
        }))
    logging.info("Min-block: cached %d tasks at %s.", len(tasks), path)


# ── Straight tasks: K* assignment + differentiation filter ──────


def _assign_min_blocks(env: "PyBulletDominoComposedEnv",
                       tasks: List[EnvironmentTask]) -> List[EnvironmentTask]:
    """Attach each min-block task's ``DominoEvaluator`` and its env-side
    ``offline_task_metrics["k_star"]`` (K*, computed by simulation).

    K* is the minimum number of blues whose evenly-spaced chain lets a real
    Push on the start topple the target at this env's true friction (see
    ``min_block_utils.compute_k_star``). Tasks whose target is unreachable
    within the blue budget, or that need no blues at all (K*=0, trivially
    solved by a direct push), are dropped - neither exercises the reach
    model. Computing K* pushes/steps the sim; that's fine here because it
    runs before any episode and every episode re-sets state.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.envs.pybullet_domino.env import DominoEvaluator
    out: List[EnvironmentTask] = []
    for env_task in tasks:
        k_star = compute_k_star(env, env_task.init)
        if k_star is None or k_star < 1:
            logging.warning(
                "Dropping min-block task (K*=%s): target unreachable "
                "within budget or solvable with a direct push.", k_star)
            continue
        if k_star >= CFG.domino_min_block_num_blues:
            logging.warning(
                "Dropping min-block task (K*=%d == staged blues %d): no "
                "spare blue for the over-build check.", k_star,
                CFG.domino_min_block_num_blues)
            continue
        # Differentiation filter (only when a planning-friction mismatch
        # is configured), direction-aware:
        #   * planning > true (over-reach): keep only believed < true -
        #     an uncalibrated minimal planner structurally UNDER-builds
        #     (chain dies, target never topples);
        #   * planning < true (under-reach): keep only believed > true
        #     (and expressible within the staged blues) - the planner
        #     OVER-builds, topples the target, but pays the per-block
        #     reward cost.
        # Dead-band spans where both frictions agree cannot separate the
        # calibrated from the uncalibrated model and are dropped.
        direction = _planning_mismatch_direction()
        if direction is not None:
            k_believed = _planning_k_star(env, env_task.init)
            if not _believed_k_differentiates(direction, k_star, k_believed):
                logging.warning(
                    "Dropping min-block task (true K*=%d, planner "
                    "believes %s, %s): does not differentiate calibrated "
                    "vs uncalibrated reach.", k_star, k_believed, direction)
                continue
        out.append(
            EnvironmentTask(env_task.init_obs,
                            env_task.goal_description,
                            alt_goal_desc=env_task.alt_goal_desc,
                            goal_nl=env_task.goal_nl,
                            evaluator=DominoEvaluator(env_task.goal),
                            offline_task_metrics={"k_star": float(k_star)},
                            early_stop_min_reward=_early_stop_bar(k_star)))
    logging.info("Min-block tasks: kept %d/%d with K* assigned.", len(out),
                 len(tasks))
    return out


def _planning_mismatch_direction() -> Optional[str]:
    """Direction of the configured planning-friction mismatch.

    ``"over_reach"`` when the planning sim's friction is HIGHER than the
    true friction (planner over-estimates topple reach -> under-
    builds), ``"under_reach"`` when lower (planner over-builds),
    ``None`` when no mismatch is configured (differentiation filters
    disabled).
    """
    planning = CFG.domino_planning_friction
    if planning is None or abs(planning - CFG.domino_true_friction) < 1e-9:
        return None
    return ("over_reach"
            if planning > CFG.domino_true_friction else "under_reach")


def _believed_k_differentiates(direction: str, k_true: int,
                               k_believed: Optional[int]) -> bool:
    """Whether a task with these K*s forces the uncalibrated planner to fail.

    * over_reach: the planner must believe FEWER blues suffice (it then
      under-builds and the chain dies short of the target);
    * under_reach: the planner must believe MORE blues are needed - but
      no more than the staged budget, so its over-built plan is
      physically expressible and pays the per-block reward cost rather
      than failing on a muddled "can't build my plan" path.
    """
    if k_believed is None:
        return False
    if direction == "over_reach":
        return k_believed < k_true
    return k_true < k_believed <= CFG.domino_min_block_num_blues


def _planning_k_star(env: "PyBulletDominoComposedEnv",
                     init_state: State) -> Optional[int]:
    """K* as the (miscalibrated) planning sim would compute it.

    Temporarily switches the env's domino friction to
    ``CFG.domino_planning_friction``, computes K*, then restores the true
    friction. Returns ``None`` when no planning-friction mismatch is
    configured, which disables the caller's differentiation filter.
    """
    if _planning_mismatch_direction() is None:
        return None
    with _believed_physics(env):
        return compute_k_star(env, init_state)


def _planning_turn_k_star(env: "PyBulletDominoComposedEnv", start_pose: Any,
                          target_pose: Any, budget: int) -> Optional[int]:
    """Turn K* as the (miscalibrated) planning sim would compute it.

    The corner analogue of ``_planning_k_star``: temporarily switches
    the env to ``CFG.domino_planning_friction``, runs the full corner
    layout search, then restores the true friction. Returns ``None``
    when no planning-friction mismatch is configured.
    """
    if _planning_mismatch_direction() is None:
        return None
    with _believed_physics(env):
        return compute_turn_k_star(env, start_pose, target_pose, budget=budget)


# ── Turn tasks ───────────────────────────────────────────────


def _make_turn_task(env: "PyBulletDominoComposedEnv",
                    gen: "DominoTaskGenerator",
                    rng: np.random.Generator) -> Optional[EnvironmentTask]:
    """Build one L-shaped (90-degree turn) min-block task, or None.

    Pipeline:

    1. Sample the geometry directly: start pose in the pushable band and
       a target one 90-degree turn away, at leg lengths drawn from the
       empirically differentiating region (long entry legs).
    2. Cheap pre-filters on memoized straight-leg probes: a feasibility
       bound (the legs' own straight-chain minima already exceed the
       staged budget) and the per-LEG differentiation certificate (see
       inline comment) - full believed corner plans rarely validate at
       the planning friction, so differentiation is certified on the
       straight legs instead. These run BEFORE the layout search below,
       which costs dozens of Push rollouts per attempt and dominated
       generation time when it ran on every attempt.
    3. K* = ``compute_turn_k_star`` at the true friction - the minimum
       blues over a layout SEARCH of agent-buildable candidates
       (straight-line probes, natural-yaw corners, and the legacy
       45-degree pair corner), because around a corner an evenly-spaced
       chain is not minimal: sliding the corner toward the start can
       save a block. The winning layout doubles as the proof the task
       is solvable. under_reach additionally requires the believed full
       corner K* to over-spend within the staged budget (the strong
       certificate, feasible at planning friction 0.1).
    4. Stage ``domino_min_block_num_blues`` blues (more than K*), so
       over-building is possible and penalized by the per-block reward
       cost.
    """
    comp = env._domino_component  # pylint: disable=protected-access
    if comp is None:
        return None
    # Synthesize the start/target geometry directly from sampled leg
    # lengths (the K* search below is itself the constructive proof of
    # solvability, so no chain needs to be pre-built). Leg bands are
    # DIRECTION-specific, each from its own empirical leg scan:
    #   * over_reach (low-friction arm): long ENTRY legs are what make a
    #     turn differentiate (short-legged corners cost the same at both
    #     frictions); the band targets the K*=3-vs-believed-2 region -
    #     longer legs give K*>=4, which num_blues=4 cannot host.
    #   * under_reach (high-friction arm): short legs around the probed
    #     (0.30, 0.24) cell - true K*=3 via the 45-degree pair corner vs
    #     believed 4 (probed 2026-07-08). Longer legs invert or tie the
    #     comparison (believed pair plans get cheap faster than the true
    #     ones) and are re-rejected per attempt by the believed-K* check
    #     below.
    # Legs snap to the probe-memo lattice (1 cm, matching _SPAN_BUCKET):
    # repeats of a shape are guaranteed memo hits (no bucket-edge double
    # probes), and the drop log becomes a readable coverage map over the
    # finite (entry, exit) cells - the data for any future retuning.
    direction = _planning_mismatch_direction()
    if CFG.domino_min_block_turn_entry_lo is not None:
        # Explicitly configured bands - probe with
        # scripts/domino_debug/probe_min_block_bands.py when the
        # friction pair changes (the differentiating cells move with the
        # frictions). The shipping under_reach arm's bands live in
        # scripts/configs/predicatorv3/envs/all.yaml.
        assert CFG.domino_min_block_turn_entry_hi is not None
        assert CFG.domino_min_block_turn_exit_lo is not None
        assert CFG.domino_min_block_turn_exit_hi is not None
        entry_leg = round(
            float(
                rng.uniform(CFG.domino_min_block_turn_entry_lo,
                            CFG.domino_min_block_turn_entry_hi)), 2)
        exit_leg = round(
            float(
                rng.uniform(CFG.domino_min_block_turn_exit_lo,
                            CFG.domino_min_block_turn_exit_hi)), 2)
    elif direction == "under_reach":
        # No silent fallback for under_reach: the legacy hardcoded band
        # (entry 0.34, exit 0.28, K*=3) ships tasks whose only true-
        # physics turn solution is the knife-edge 45-degree pair corner,
        # which LLM planner arms cannot discover (retune 2026-07-12).
        raise ValueError(
            "under_reach turn tasks require explicit "
            "domino_min_block_turn_{entry,exit}_{lo,hi} flags (probe with "
            "scripts/domino_debug/probe_min_block_bands.py; see the "
            "domino_high_friction block in envs/all.yaml).")
    else:
        entry_leg = round(float(rng.uniform(*_DEFAULT_TURN_ENTRY_BAND)), 2)
        exit_leg = round(float(rng.uniform(*_DEFAULT_TURN_EXIT_BAND)), 2)
    sx = float(rng.uniform(comp.domino_x_lb, comp.domino_x_ub))
    sy = float(rng.uniform(comp.domino_y_lb, comp.domino_y_ub))
    syaw = float(rng.choice([0.0, np.pi / 2, -np.pi / 2]))
    side = float(rng.choice([-1.0, 1.0]))
    u_vec = np.array([np.sin(syaw), np.cos(syaw)])
    p_vec = side * np.array([-u_vec[1], u_vec[0]])  # turn side
    t_pt = np.array([sx, sy]) + entry_leg * u_vec + exit_leg * p_vec
    if not (env.x_lb < t_pt[0] < env.x_ub and env.y_lb < t_pt[1] < env.y_ub):
        return None
    # Fall axis (-sin, cos) along the exit direction.
    tyaw = geometry.yaw_along(p_vec[0], p_vec[1])
    start = comp.dominos[0]
    target = comp.dominos[1]
    start_pose = (sx, sy, syaw)
    target_pose = (float(t_pt[0]), float(t_pt[1]), tyaw)
    num_blues = min(CFG.domino_min_block_num_blues, len(comp.dominos) - 2)
    # Cheap memoized leg probes FIRST - the corner layout search below is
    # by far the most expensive step (dozens of real Push rollouts per
    # attempt), so attempts are pre-filtered on straight-leg reach alone.
    legs = (entry_leg, exit_leg)
    true_legs = [
        straight_span_k_star(env, leg, budget=num_blues) for leg in legs
    ]
    true_counts = [v for v in true_legs if v is not None]
    if len(true_counts) < len(legs):
        logging.warning(
            "Dropping turn task (true-friction leg probe failed: legs=%s "
            "-> %s).", legs, true_legs)
        return None
    if sum(true_counts) >= num_blues:
        # Feasibility bound: a turn chain cannot beat its legs' own
        # straight-chain minima (the stretched corner saves at most the
        # corner blue itself), so K* >= sum(true legs) - which already
        # leaves no spare blue. Skip the layout search outright.
        logging.warning(
            "Dropping turn task (legs alone need %s blues >= staged "
            "blues %d).", true_legs, num_blues)
        return None
    bel_legs: List[Optional[int]] = []
    if direction is not None:
        # Relaxed per-LEG certificate. For over_reach the strong "a
        # cheaper believed plan validates in the wrong sim" check is
        # unusable for turns: natural corners barely propagate at the
        # planning friction (high friction grips the base - redirection
        # is what's hard), so a full believed corner plan almost never
        # exists and every turn task would drop. Instead certify reach
        # differentiation where it actually lives - on the straight
        # LEGS (under_reach adds the strong check AFTER the K* search,
        # where it is both feasible and necessary): probe how
        # many blues each friction needs for a straight chain of each
        # leg's length (real rollouts). The corner's own cost is the
        # same on both sides of the comparison and cancels.
        env.set_domino_physical_params(
            lateral_friction=CFG.domino_planning_friction)
        try:
            bel_legs = [
                straight_span_k_star(env, leg, budget=num_blues)
                for leg in legs
            ]
        finally:
            env.set_domino_physical_params(
                lateral_friction=CFG.domino_true_friction)
        bel_counts = [v for v in bel_legs if v is not None]
        if len(bel_counts) < len(legs):
            logging.warning(
                "Dropping turn task (planning-friction leg probe failed: "
                "legs=%s true=%s believed=%s).", legs, true_legs, bel_legs)
            return None
        t_sum, b_sum = sum(true_counts), sum(bel_counts)
        # over_reach: the legs themselves must differentiate (believed
        # cheaper), since a full believed corner plan rarely exists at
        # friction 0.5 to certify against. under_reach: since the
        # 2026-07-09 yaw-parity fix the differentiation lives in the
        # CORNER premium (the believed corner build needs an extra blue
        # at planning friction), which per-leg straight probes cannot
        # see - the validated leg cells tie per-leg at both frictions.
        # So under_reach only pre-filters on believed expressibility
        # (b_sum + 1 corner within the staged blues) and defers to the
        # strong full-family certificate below.
        differentiates = (b_sum < t_sum if direction == "over_reach" else
                          b_sum + 1 <= num_blues)
        if not differentiates:
            logging.warning(
                "Dropping turn task (legs true=%s believed=%s, %s): "
                "does not differentiate calibrated vs uncalibrated "
                "reach.", true_legs, bel_legs, direction)
            return None
    # under_reach K* demands TWO independent topplers: at true friction
    # 0.5 the only working corner is the legacy 45-degree pair (see
    # _candidate_turn_layouts) and it is knife-edge - a task whose only
    # solution toppled once during generation can be unsolvable under a
    # fresh simulator's contact history. over_reach keeps min_hits=1 so
    # the (already-run) low-friction arm's task sets are unchanged.
    k_true = compute_turn_k_star(
        env,
        start_pose,
        target_pose,
        budget=num_blues,
        min_hits=2 if direction == "under_reach" else 1)
    if k_true is None or k_true < 1:
        logging.warning("Dropping turn task (searched K*=%s).", k_true)
        return None
    if k_true >= num_blues:
        # No spare blue would remain: the over-build side of the reward
        # would be vacuous (the staging grid caps the scene's blues).
        logging.warning(
            "Dropping turn task (K*=%d == staged blues %d): no spare "
            "blue for the over-build check.", k_true, num_blues)
        return None
    k_bel_full: Optional[int] = None
    if direction == "under_reach":
        # STRONG certificate, feasible in this direction (corners DO
        # validate at planning friction 0.1): the believed minimum over
        # the full layout family must strictly exceed the true K* (else
        # the believed build fits the budget and nothing separates the
        # models - neighboring leg cells invert, e.g. entry 0.44 probes
        # believed 4 < true 5) and keep a spare blue below the staged
        # count. STRICTLY below, mirroring the straight span-band
        # rationale: a believed plan that needs every staged blue is
        # knife-edge across the PyBullet settle round-trip (a certified
        # K*=4-vs-believed-5 turn regenerated as unreproducible from
        # its staged poses).
        k_bel_full = _planning_turn_k_star(env, start_pose, target_pose,
                                           num_blues)
        if k_bel_full is None or not k_true < k_bel_full < num_blues:
            logging.warning(
                "Dropping turn task (true K*=%d, believed full K*=%s, "
                "under_reach): believed corner build does not over-spend "
                "within the staged budget.", k_true, k_bel_full)
            return None
    if direction is not None:
        logging.info(
            "Turn task differentiates (%s): true K*=%d, believed full "
            "K*=%s, legs true=%s believed=%s.", direction, k_true, k_bel_full,
            true_legs, bel_legs)
    # Build the scene: start/target fixed at the chain's endpoints,
    # num_blues blues staged (scattered by the staging pass).
    scene: Dict[Object, Dict[str, Any]] = {
        start:
        comp.place_domino(0,
                          start_pose[0],
                          start_pose[1],
                          start_pose[2],
                          is_start_block=True),
        target:
        comp.place_domino(1,
                          target_pose[0],
                          target_pose[1],
                          target_pose[2],
                          is_target_block=True),
    }
    blues = [d for d in comp.dominos if d not in (start, target)]
    for blue in blues[:num_blues]:
        # Initial position is irrelevant - the staging pass re-places
        # every movable (blue) block on the staging grid.
        scene[blue] = comp.place_domino(0, start_pose[0], start_pose[1], 0.0)
    staged = gen.stage_movable_blocks(scene)
    if staged is None:
        return None
    if direction == "under_reach":
        # Re-run the strong believed certificate AFTER the staging pass,
        # TWICE: staging's own rollouts perturb the contact-solver
        # history, which is the variance a fresh evaluation sim
        # exhibits. A believed corner that only over-spends under some
        # histories is knife-edge (observed: believed full K* flipping
        # 4 -> 3 between processes, and - short-leg retune probes,
        # 2026-07-12 - believed k=2 corners toppling in 1-of-3 repeat
        # probes on band cells). Each independent repetition roughly
        # halves the odds of shipping a task whose believed side flips
        # in the miscalibrated planner's fresh sim (which would let it
        # match the true K* and erase the task's signal - a lost
        # differentiation, not a correctness hole).
        assert CFG.domino_planning_friction is not None
        for recheck_round in range(2):
            k_bel_recheck = _planning_turn_k_star(env, start_pose, target_pose,
                                                  num_blues)
            if k_bel_recheck is None or not k_true < k_bel_recheck < num_blues:
                logging.warning(
                    "Dropping turn task (believed full K* recheck %d=%s, "
                    "first probe=%s, true K*=%d): believed margin is "
                    "contact-history knife-edge.", recheck_round,
                    k_bel_recheck, k_bel_full, k_true)
                return None
            # History-robust certificate, interleaved so each scan runs
            # under a different solver history: no K*-blue candidate may
            # topple at BOTH frictions. Then whichever cheap plan a
            # fresh believed planner knife-edges onto, executing it
            # still fails truly (death), while believed over-builds
            # fail on budget.
            if dual_valid_turn_layout_exists(env, start_pose, target_pose,
                                             k_true,
                                             CFG.domino_planning_friction,
                                             CFG.domino_true_friction):
                logging.warning(
                    "Dropping turn task (true K*=%d, round %d): a K*-blue "
                    "layout validates at the planning friction AND topples "
                    "at the true friction - the believed planner could "
                    "truly succeed within budget.", k_true, recheck_round)
                return None
    return _finish_min_block_task(env, comp, staged, k_true,
                                  goal_text.MIN_BLOCK_GOAL_NL)


# ── Heavy-block (immovable obstacle) tasks ───────────────────


def _to_real_pose(x: float, y: float, yaw: float, d_yaw: float, ax: float,
                  ay: float, sx: float,
                  sy: float) -> Tuple[float, float, float]:
    """Map a canonical-anchor pose to a real start pose rotated by ``d_yaw``.

    State yaw is a CCW z-rotation, so adding d_yaw to every yaw pairs
    with rotating offsets CCW by [[c, -s], [s, c]] (callers only pass
    d_yaw in {0, -pi}, where this equals the old mirrored matrix).
    """
    dx, dy = x - ax, y - ay
    c, s = np.cos(d_yaw), np.sin(d_yaw)
    return (sx + c * dx - s * dy, sy + s * dx + c * dy, yaw + d_yaw)


def _stage_heavy_scene(
        gen: "DominoTaskGenerator", comp: Any, num_blues: int, start_pose: Any,
        target_pose: Any,
        heavy_pose: Any) -> Optional[Dict[Object, Dict[str, Any]]]:
    """Scene dict with start/target/gray fixed and blues staged, or None."""
    # pylint: disable=protected-access
    start, target, heavy_obj = comp.dominos[0], comp.dominos[1], \
        comp.dominos[-1]
    scene: Dict[Object, Dict[str, Any]] = {
        start:
        comp.place_domino(0,
                          start_pose[0],
                          start_pose[1],
                          start_pose[2],
                          is_start_block=True),
        target:
        comp.place_domino(1,
                          target_pose[0],
                          target_pose[1],
                          target_pose[2],
                          is_target_block=True),
        heavy_obj:
        comp.place_domino(0,
                          heavy_pose[0],
                          heavy_pose[1],
                          heavy_pose[2],
                          is_heavy_block=True),
    }
    blues = [d for d in comp.dominos if d not in scene]
    for blue in blues[:num_blues]:
        # Initial position is irrelevant -- the staging pass re-places
        # every movable (blue) block on the staging grid (the gray block
        # is exempt and stays where the task put it).
        scene[blue] = comp.place_domino(0, start_pose[0], start_pose[1], 0.0)
    return gen.stage_movable_blocks(scene)


def _finish_min_block_task(env: "PyBulletDominoComposedEnv", comp: Any,
                           staged: Dict[Object, Dict[str, Any]], k_star: int,
                           goal_nl: str) -> EnvironmentTask:
    """Assemble the EnvironmentTask from a staged min-block scene.

    The target is the purple ``comp.dominos[1]``. ``k_star`` is both the
    offline ``k_star`` metric and the early-stop bar: the searched true K*
    for the straight/turn tasks, or the STAGED blue count for the heavy
    tasks (whose corner/swerve minima are solver-history sensitive at the
    margin - a layout that barely topples during generation can need one
    more blue under a fresh simulator's contact state - so there the K*
    search is a solvability certificate rather than the shipped count).
    Either way the per-block reward cost penalizes over-building.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from predicators.envs.pybullet_domino.env import DominoEvaluator
    from predicators.utils import create_state_from_dict
    init_dict: Dict[Object, Dict[str, Any]] = {
        env._robot: env.robot_init_state_dict()
    }
    init_dict.update(staged)
    init_state = create_state_from_dict(init_dict)
    goal_atoms = {GroundAtom(comp.Toppled, [comp.dominos[1]])}
    return EnvironmentTask(init_state,
                           goal_atoms,
                           goal_nl=goal_nl,
                           evaluator=DominoEvaluator(goal_atoms),
                           offline_task_metrics={"k_star": float(k_star)},
                           early_stop_min_reward=_early_stop_bar(k_star))


def _make_heavy_straight_task(
        env: "PyBulletDominoComposedEnv", gen: "DominoTaskGenerator",
        rng: np.random.Generator) -> Optional[EnvironmentTask]:
    """Build one straight-variant heavy-block task, or None.

    Natural alignment: start and target sit on one line, and the GRAY
    block stands a small sampled offset OFF that line
    (``_HEAVY_GRAY_LATERAL_OFFSETS``), all facing along it. The believed
    physics (normal gray mass, see the env init / ``block_mass``
    override) makes the cheapest plan a shallow dogleg THROUGH the gray
    -- a free link. At the true mass the chain dies against the gray,
    and the real solution is a half-circle swerve around it (the
    ``swerve_k_star`` family: aligned at both ends, bulging sideways to
    clear the block - gentler on the side the gray leans away from).

    Certificate (lure probes at the canonical anchor, memoized):
    1. believed straight-through exists (k_bel blues);
    2. true straight-through is dead for every stageable split;
    3. a true swerve exists with k_bel < K* <= staged blues,
       re-verified at the real pose (which doubles as the push-
       reachability check). K* certifies SOLVABILITY only; the reward
       budget is the staged blues (see ``_finish_min_block_task``).
    """
    # pylint: disable=protected-access
    comp = env._domino_component
    if comp is None:
        return None
    span = round(float(rng.uniform(*_HEAVY_SPAN_BAND)), 2)
    h_frac = float(rng.choice(_HEAVY_GRAY_FRACTIONS))
    h_off = float(rng.choice(_HEAVY_GRAY_LATERAL_OFFSETS)) * float(
        rng.choice([-1.0, 1.0]))
    num_blues = min(CFG.domino_min_block_num_blues, len(comp.dominos) - 3)
    ax, ay = _PROBE_ANCHOR
    # Canonical line runs along +x (start yaw pi/2), so the gray's
    # small lateral offset is along +-y; it keeps facing the line.
    c_start = (ax, ay, np.pi / 2)
    c_heavy = (ax + h_frac * span, ay + h_off, np.pi / 2)
    c_target = (ax + span, ay, np.pi / 2)
    # 1) The believed straight-through must exist (the lure): with the
    # off-line gray this is a shallow dogleg bending through it.
    with _believed_physics(env, believed_heavy_mass=True):
        k_bel = heavy_dogleg_k_star(env, c_start, c_target, c_heavy, num_blues)
    if k_bel is None:
        logging.warning(
            "Dropping heavy straight task (no believed straight-through "
            "within %d blues, span %.2f, gray offset %.2f).", num_blues, span,
            h_off)
        return None
    # 2) The true straight-through must be dead AT THE BELIEVED COST: a
    # block-minimizing planner only builds believed-cheapest (k_bel)
    # layouts, so a freak jump-over at some other count cannot leak -
    # restricting the scan to k_bel cuts the sweep ~4x.
    k_dead = heavy_dogleg_k_star(env,
                                 c_start,
                                 c_target,
                                 c_heavy,
                                 num_blues,
                                 only_k=k_bel)
    if k_dead is not None:
        logging.warning(
            "Dropping heavy straight task (true straight-through still "
            "topples at the believed cost k=%d).", k_dead)
        return None
    # 3) A swerve around the gray must exist, strictly dearer than the
    # lure (structural -- the lure gets the gray link for free -- but
    # verified).
    k_star_c = swerve_k_star(env, c_start, c_target, c_heavy, num_blues)
    if k_star_c is None or k_bel >= k_star_c:
        logging.warning(
            "Dropping heavy straight task (swerve K*=%s vs believed "
            "k=%d, span %.2f, gray offset %.2f).", k_star_c, k_bel, span,
            h_off)
        return None
    # Place the certified shape; retry poses on placement-local failures.
    staged = None
    k_true: Optional[int] = None
    for _ in range(12):
        syaw = float(rng.choice([np.pi / 2, -np.pi / 2]))
        if syaw > 0:  # falls toward +x
            sx = float(rng.uniform(comp.domino_x_lb, env.x_ub - span - 0.03))
        else:
            sx = float(rng.uniform(env.x_lb + span + 0.03, comp.domino_x_ub))
        sy = float(rng.uniform(comp.domino_y_lb, comp.domino_y_ub))
        u_vec = np.array([np.sin(syaw), np.cos(syaw)])
        # Same perpendicular convention as the canonical frame: for the
        # canonical u = (1, 0) this perp is (0, 1), i.e. +y = +h_off.
        perp_vec = np.array([-u_vec[1], u_vec[0]])
        h_pt = np.array([sx, sy]) + h_frac * span * u_vec + h_off * perp_vec
        t_pt = np.array([sx, sy]) + span * u_vec
        # The swerve needs sideways room on at least one side.
        if not all(env.x_lb < pt[0] < env.x_ub and env.y_lb +
                   0.09 < pt[1] < env.y_ub - 0.09 for pt in (h_pt, t_pt)):
            continue
        start_pose = (sx, sy, syaw)
        heavy_pose = (float(h_pt[0]), float(h_pt[1]), syaw)
        target_pose = (float(t_pt[0]), float(t_pt[1]), syaw)
        # Staging FIRST (pure geometry, no sim): the staging grid is a
        # single row, so this is the common per-pose failure and must
        # cost nothing. Only a staged pose pays the sim re-verification.
        staged = _stage_heavy_scene(gen, comp, num_blues, start_pose,
                                    target_pose, heavy_pose)
        if staged is None:
            continue
        # Re-verify the swerve at THIS pose (real push reachability),
        # demanding TWO independent toppling swerves: a task whose only
        # solution is one knife-edge layout can flip under a fresh
        # simulator's contact-solver history.
        k_true = swerve_k_star(env,
                               start_pose,
                               target_pose,
                               heavy_pose,
                               num_blues,
                               min_hits=2)
        if k_true is None or k_true < 1:
            staged = None
            continue
        break
    if staged is None:
        logging.warning(
            "Dropping heavy straight task (no placement for span %.2f).", span)
        return None
    assert k_true is not None
    logging.info(
        "Heavy straight task differentiates: believed through-gray k=%d, "
        "true dead, swerve K*=%d.", k_bel, k_true)
    return _finish_min_block_task(env, comp, staged, num_blues,
                                  goal_text.HEAVY_GOAL_NL)


# Blueprint memo for the turn variant: the believed-physics corner
# layout for an L shape, found once at the canonical anchor and reused
# across attempts and placements (keyed on the shape lattice + the
# frictions that define the believed physics).
_corner_blueprint_memo: Dict[Any, Optional[Any]] = {}


def _mirror_od_about_anchor(
        od: Dict[Object, Dict[str, Any]]) -> Dict[Object, Dict[str, Any]]:
    """Reflect a canonical-anchor layout across the anchor's fall line.

    (y = anchor_y): (x, y, yaw) -> (x, 2*ay - y, pi - yaw).
    """
    _, ay = _PROBE_ANCHOR
    out: Dict[Object, Dict[str, Any]] = {}
    for obj, pose in od.items():
        q = dict(pose)
        q["y"] = 2 * ay - pose["y"]
        q["yaw"] = geometry.wrap_angle(np.pi - pose["yaw"])
        out[obj] = q
    return out


def _believed_corner_blueprint(env: "PyBulletDominoComposedEnv", comp: Any,
                               entry: float, exit_leg: float, side: float,
                               num_blues: int) -> Optional[Any]:
    """Cheapest believed-physics corner layout for this L shape, at the
    canonical anchor: (layout od, corner object, k_full) or None.

    Enumerates the SAME candidate family the K* search uses and keeps
    the first toppling layout -- but only if it has a mid-chain corner
    (>= 1 entry blue): the gray block will replace that corner, and a
    detour must have room to bend before it. If the cheapest believed
    layout is cornerless (straight) or start-adjacent, the shape cannot
    host the lure and is rejected.

    The sweep runs for the LEFT-turning side only and mirrors the result
    for the other side (chain physics is mirror-symmetric; the caller's
    gray-substituted lure rollout re-verifies the mirrored geometry, so
    any residual push asymmetry is caught rather than trusted). This
    halves the expensive blueprint misses.
    """
    # pylint: disable=protected-access
    key = (entry, exit_leg, num_blues, CFG.domino_planning_friction,
           CFG.domino_true_friction)
    ax, ay = _PROBE_ANCHOR
    c_start = (ax, ay, np.pi / 2)
    c_target = (ax + entry, ay + exit_leg, 0.0)
    push_opt = mbu._get_push_option(env)

    def _probe() -> Optional[Any]:
        # Cap the sweep: shapes where NO corner layout propagates used to
        # sweep every candidate at every k (~minutes) before concluding
        # None. The chord's straight-chain minimum (memoized, ~free)
        # bounds where a corner plan could plausibly first appear - a
        # corner path is longer and lossier than the chord, so if
        # nothing has toppled within two counts past that minimum,
        # corners don't work for this shape (heuristic: may rarely skip
        # a viable shape, never keeps a wrong one).
        chord = float(np.hypot(entry, exit_leg))
        c_min = straight_span_k_star(env, chord, budget=num_blues + 1)
        if c_min is None:
            return None
        k_hi = min(num_blues + 1, c_min + 2)
        # k_full may exceed the staged blues by one: the gray replaces
        # the corner blue, so the LURE costs k_full - 1 blues. Scan ALL
        # candidates of each k: the gray layout must sit at the family's
        # MINIMUM cost (else a cheaper own-corner plan would dodge the
        # gray), but within that minimum any mid-chain-corner layout
        # qualifies - the cheapest topplers are often start-adjacent
        # (k1=0) while equally-cheap mid-chain ones follow.
        for k in range(2, k_hi + 1):
            any_topple = False
            for od, s_, t_ in mbu._candidate_turn_layouts(
                    comp, k, c_start, c_target):
                if not mbu._layout_topples(env, od, s_, t_, push_opt):
                    continue
                any_topple = True
                blues = [o for o in od if o not in (s_, t_)]
                entry_blues = [
                    o for o in blues
                    if abs(geometry.wrap_angle(od[o]["yaw"] -
                                               np.pi / 2)) < 0.15
                ]
                corners = [
                    o for o in blues
                    if 0.3 < abs(geometry.wrap_angle(od[o]["yaw"] -
                                                     np.pi / 2)) < 1.2
                ]
                if len(corners) == 1 and entry_blues:
                    return od, corners[0], k
            if any_topple:
                # The family minimum topples only via straight or
                # start-adjacent-corner layouts: no natural mid-chain
                # corner spot to occupy at the believed-best cost.
                return None
        return None

    if key in _corner_blueprint_memo:
        result = _corner_blueprint_memo[key]
    else:
        with _believed_physics(env, believed_heavy_mass=True):
            result = _probe()
        _corner_blueprint_memo[key] = result
    if result is None or side > 0:
        return result
    od_bel, corner_obj, k_full = result
    return _mirror_od_about_anchor(od_bel), corner_obj, k_full


def _make_heavy_turn_task(
        env: "PyBulletDominoComposedEnv", gen: "DominoTaskGenerator",
        rng: np.random.Generator) -> Optional[EnvironmentTask]:
    """Build one turn-variant heavy-block task, or None.

    Natural alignment: an L-shaped start/target pair whose believed-
    cheapest plan turns at a corner -- and the GRAY block stands exactly
    where that natural corner blue would go, at the corner's natural
    yaw. In the believed physics (normal gray mass) it is a ready-made
    corner FOR FREE, one blue cheaper than any own-corner plan, so a
    block-minimizing planner routes through it; at the true mass the
    chain dies there, and the real solution skips around it (an own
    corner elsewhere on the entry line -- the detour search with the
    gray as obstacle).

    Certificate: believed corner blueprint exists with a mid-chain
    corner; the gray-substituted lure still propagates believedly and
    dies at the true physics; the true detour K* (noise-robust, see
    ``heavy_detour_k_star_robust``) fits the staged blues with at least
    one blue of SLACK and is STRICTLY dearer than the lure (legs
    sampled from the short ``_HEAVY_TURN_*`` bands so both certify
    often). K* certifies SOLVABILITY only; the reward budget is the
    staged blues (see ``_finish_min_block_task``).
    """
    # pylint: disable=protected-access
    comp = env._domino_component
    if comp is None:
        return None
    entry = round(float(rng.uniform(*_HEAVY_TURN_ENTRY_BAND)), 2)
    exit_leg = round(float(rng.uniform(*_HEAVY_TURN_EXIT_BAND)), 2)
    side = float(rng.choice([-1.0, 1.0]))
    num_blues = min(CFG.domino_min_block_num_blues, len(comp.dominos) - 3)
    bp = _believed_corner_blueprint(env, comp, entry, exit_leg, side,
                                    num_blues)
    if bp is None:
        logging.warning(
            "Dropping heavy turn task (no believed mid-chain corner plan "
            "for entry=%.2f exit=%.2f).", entry, exit_leg)
        return None
    od_bel, corner_obj, k_full = bp
    k_bel = k_full - 1
    if k_bel > num_blues:
        logging.warning(
            "Dropping heavy turn task (lure needs %d blues > staged %d).",
            k_bel, num_blues)
        return None
    heavy_obj = comp.dominos[-1]
    start, target = comp.dominos[0], comp.dominos[1]
    cp = od_bel[corner_obj]
    corner_pose_c = (float(cp["x"]), float(cp["y"]), float(cp["yaw"]))
    lure_od = {o: dict(p) for o, p in od_bel.items() if o is not corner_obj}
    lure_od[heavy_obj] = comp.place_domino(0,
                                           *corner_pose_c,
                                           is_heavy_block=True)
    push_opt = mbu._get_push_option(env)
    # The gray-substituted lure must still propagate in the believed
    # physics (it is body-identical to the corner blue there)...
    with _believed_physics(env, believed_heavy_mass=True):
        ok_bel = mbu._layout_topples(env, lure_od, start, target, push_opt)
    if not ok_bel:
        logging.warning(
            "Dropping heavy turn task (gray-substituted lure fails "
            "believedly, entry=%.2f exit=%.2f).", entry, exit_leg)
        return None
    # ...and must DIE at the true physics.
    if mbu._layout_topples(env, lure_od, start, target, push_opt):
        logging.warning(
            "Dropping heavy turn task (lure survives the true physics: "
            "chain passes the gray corner).")
        return None
    ax, ay = _PROBE_ANCHOR
    c_tx, c_ty = ax + entry, ay + side * exit_leg
    c_tyaw = 0.0 if side > 0 else np.pi
    staged = None
    k_true: Optional[int] = None
    # Generous pose retries: besides bounds/staging, a pose must now
    # also pass the real-pose lure re-verification below, and knife-edge
    # pose transfer fails often. Failed poses are cheap (staging is pure
    # geometry; a lure fail costs ~2 rollouts before the detour search).
    for _ in range(24):
        syaw = float(rng.choice([np.pi / 2, -np.pi / 2]))
        d_yaw = syaw - np.pi / 2
        if syaw > 0:
            sx = float(rng.uniform(comp.domino_x_lb, env.x_ub - entry - 0.05))
        else:
            sx = float(rng.uniform(env.x_lb + entry + 0.05, comp.domino_x_ub))
        sy = float(rng.uniform(comp.domino_y_lb, comp.domino_y_ub))
        start_pose = (sx, sy, syaw)
        target_pose = _to_real_pose(c_tx, c_ty, c_tyaw, d_yaw, ax, ay, sx, sy)
        heavy_pose = _to_real_pose(*corner_pose_c, d_yaw, ax, ay, sx, sy)
        if not all(env.x_lb < px < env.x_ub and env.y_lb < py < env.y_ub
                   for px, py in ((target_pose[0], target_pose[1]),
                                  (heavy_pose[0], heavy_pose[1]))):
            continue
        # Staging FIRST (pure geometry, no sim): the staging grid is a
        # single row, so this is the common per-pose failure and must
        # cost nothing. Only a staged pose pays the sim checks below.
        staged = _stage_heavy_scene(gen, comp, num_blues, start_pose,
                                    target_pose, heavy_pose)
        if staged is None:
            continue
        # Re-verify the LURE at THIS pose: the anchor checks above are
        # cheap pre-filters, but pose transfer is only approximately
        # physics-preserving and knife-edge corners can flip under it -
        # a task whose real-pose lure fails believedly would not lure
        # the baseline at all. The lure must also be believedly
        # NOISE-ROBUST (its blues survive the same placement scatter the
        # detour certificate demands): a knife-edge lure fails the
        # believed baseline's own flaky-plan validation, and a
        # block-minimizing believed planner would then walk off the
        # gray onto the (dearer but robust) detour and solve with no
        # learning.
        lure_real = {}
        for lure_obj, lure_p in lure_od.items():
            rx, ry, ryaw = _to_real_pose(lure_p["x"], lure_p["y"],
                                         lure_p["yaw"], d_yaw, ax, ay, sx, sy)
            lure_real[lure_obj] = comp.place_domino(
                0,
                rx,
                ry,
                ryaw,
                is_start_block=lure_obj is start,
                is_target_block=lure_obj is target,
                is_heavy_block=lure_obj is heavy_obj)

        pose_seed = int(round(sx * 1e3)) * 7919 + int(round(sy * 1e3))
        with _believed_physics(env, believed_heavy_mass=True):
            ok_real = mbu._layout_topples(env, lure_real, start, target,
                                          push_opt) \
                and mbu._layout_noise_robust(env, comp, lure_real, push_opt,
                                             pose_seed)
        if not ok_real:
            staged = None
            continue
        if mbu._layout_topples(env, lure_real, start, target, push_opt):
            # Jump-over leak at this pose (true physics).
            staged = None
            continue
        # The detour must exist at THIS pose (skip around the gray with
        # an own corner; doubles as the push-reachability check) - with
        # TWO independent toppling layouts that also SURVIVE placement
        # noise: nominal-only certification accepts knife-edge layouts
        # that flip under a fresh simulator's contact-solver history
        # (the task becomes unreproducible) and that the real Place
        # skill's ~1-2 cm settle scatter cannot build (2026-07-25 runs:
        # 0/3 turn seeds solved). The detour must also leave at least
        # ONE blue of slack under the staged budget (scan capped at
        # num_blues - 1, so the exact-budget layer is never even
        # probed): an exact-budget task forces the solver onto a single
        # knife-edge layout family.
        k_true = mbu.heavy_detour_k_star_robust(env,
                                                start_pose,
                                                target_pose,
                                                heavy_pose,
                                                budget=num_blues - 1,
                                                min_hits=2)
        if k_true is None or k_true < 1:
            staged = None
            continue
        # The robust detour must be STRICTLY dearer than the believed
        # gray-corner lure: at a tie, a block-minimizing believed
        # planner may tie-break away from the gray onto the (robust,
        # gray-free, physics-mismatch-immune) detour and solve with no
        # learning at all - the task would stop differentiating.
        if k_true <= k_bel:
            staged = None
            continue
        break
    if staged is None:
        logging.warning(
            "Dropping heavy turn task (no placement for entry=%.2f "
            "exit=%.2f).", entry, exit_leg)
        return None
    assert k_true is not None
    logging.info(
        "Heavy turn task differentiates: believed gray-corner k=%d, lure "
        "dead at true physics, detour K*=%d.", k_bel, k_true)
    return _finish_min_block_task(env, comp, staged, num_blues,
                                  goal_text.HEAVY_GOAL_NL)
