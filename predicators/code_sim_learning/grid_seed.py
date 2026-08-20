"""Coordinate grid sweeps over the rollout SSE landscape.

Grid seeding relocates each physical param's LM start into the right
basin (flat-set selection, anchor-nearest choice, flat-edge bisection),
and the same candidate grid backs the per-trajectory explainability
sweep (:func:`min_explainable_rms`). See the
:mod:`predicators.code_sim_learning.physical_sysid` module docstring
for the identification problem these serve.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import ParamSpec, is_log, \
    scalar_from_fit_space, scalar_to_fit_space
from predicators.code_sim_learning.rollout_env import RolloutTrajectory
from predicators.code_sim_learning.rollout_objective import \
    compute_rollout_sse, per_trajectory_rms
from predicators.code_sim_learning.trajectory_prep import ResidualScaling

logger = logging.getLogger(__name__)

# Fit-space (z) bisection tolerance when refining a flat-interval edge.
_FLAT_EDGE_Z_TOL = 1e-3


def grid_candidates(spec: ParamSpec, num_points: int) -> np.ndarray:
    """Sweep values across ``spec``'s box, evenly spaced in its FIT space.

    Linear params get ``linspace``; log params get ``geomspace`` — equal
    resolution per decade. This matters when the box spans orders of
    magnitude: ``linspace(0.01, 2.0, 8)`` puts 7 of 8 points above 0.29
    and NOTHING between 0.01 and 0.29, so a true friction of 0.1 has no
    nearby candidate and the sweep jumps to the 0.01 endpoint (measured
    on run_20260706_171526: fitted 0.0114 vs true 0.1). ``geomspace``
    puts a candidate at ~0.098.
    """
    assert spec.lo is not None and spec.hi is not None
    if is_log(spec):
        return np.geomspace(spec.lo, spec.hi, num_points)
    return np.linspace(spec.lo, spec.hi, num_points)


def _flat_candidates(
    pool: Sequence[Tuple[float, float]],
    noise_floor: float,
    flat_frac: float,
) -> Tuple[List[Tuple[float, float]], float, float]:
    """Split a ``(value, SSE)`` pool into its data-equivalent flat set.

    Candidates whose SSE is within ``max(noise_floor, flat_frac *
    best_SSE)`` of the best candidate are indistinguishable on this
    data: the tolerance is relative to the best achievable SSE, so a
    sharp basin (tiny best SSE) admits only true equals while a
    misfit-dominated landscape (large best SSE) treats its whole
    saturated shelf as one plateau. Returns ``(flat_members, best_sse,
    tolerance)``; with ``flat_frac`` and ``noise_floor`` both 0 the
    flat set degenerates to the exact argmin.
    """
    best_sse = min(sse for _v, sse in pool)
    tol = max(noise_floor, flat_frac * best_sse)
    flat = [(v, sse) for v, sse in pool if sse <= best_sse + tol]
    return flat, best_sse, tol


def _closest_to_anchor(spec: ParamSpec, flat: Sequence[Tuple[float, float]],
                       anchor: float) -> float:
    """The flat-set member nearest the anchor in fit space (ties: lower SSE).

    Among data-equivalent values the anchor-nearest one is the MAP
    choice: the data expresses no preference within the flat set, so the
    prior (centered on the anchor) decides. This is also what keeps the
    fit stable across cycles - a jitter-argmin wanders around the
    plateau as segments come and go, the anchor-side edge does not.
    """
    z_anchor = scalar_to_fit_space(spec, anchor)

    def _rank(entry: Tuple[float, float]) -> Tuple[float, float]:
        value, sse = entry
        return (abs(scalar_to_fit_space(spec, value) - z_anchor), sse)

    return min(flat, key=_rank)[0]


def _refine_flat_edge(spec: ParamSpec, pool: List[Tuple[float, float]],
                      anchor: float, noise_floor: float, refine_evals: int,
                      sse_for: Callable[[ParamSpec, float],
                                        float], flat_frac: float) -> float:
    """Bisect the anchor-side edge of ``spec``'s flat set to sub-grid
    resolution.

    The coarse grid quantizes the seed to ~one grid gap
    (``geomspace(0.01, 2, 7)`` has 2.4x spacing), and LM cannot repair
    that on a chaotic replay landscape whose fine-scale gradient is
    noise (measured on run_20260711_224624: LM moved the seeded lateral
    friction by <0.3% and "converged", so the fit reported the 0.827
    grid point every cycle against a true 0.5 sitting mid-gap). Each
    iteration evaluates the fit-space midpoint between the current
    chosen value (the anchor-nearest flat member) and the nearest
    evaluated non-flat value on its anchor side - every evaluated value
    strictly between the anchor and the chosen one is non-flat, else it
    would itself be chosen. A flat midpoint moves the edge toward the
    anchor; a non-flat one tightens the bracket. The flat set is
    recomputed from the full pool each iteration, so a midpoint that
    reveals a genuinely better basin re-anchors everything on it.
    Appends its evaluations to ``pool`` and returns the refined choice.
    """
    z_anchor = scalar_to_fit_space(spec, anchor)
    for _ in range(refine_evals):
        flat, _best, _tol = _flat_candidates(pool, noise_floor, flat_frac)
        chosen = _closest_to_anchor(spec, flat, anchor)
        z_chosen = scalar_to_fit_space(spec, chosen)
        if z_chosen == z_anchor:
            break  # The anchor itself became data-equivalent.
        lo_z, hi_z = sorted((z_anchor, z_chosen))
        between = [
            scalar_to_fit_space(spec, v) for v, _s in pool
            if lo_z < scalar_to_fit_space(spec, v) < hi_z
        ]
        if z_chosen > z_anchor:
            z_far = max(between) if between else z_anchor
        else:
            z_far = min(between) if between else z_anchor
        if abs(z_chosen - z_far) <= _FLAT_EDGE_Z_TOL:
            break
        mid = scalar_from_fit_space(spec, 0.5 * (z_chosen + z_far))
        pool.append((mid, sse_for(spec, mid)))
    flat, _best, _tol = _flat_candidates(pool, noise_floor, flat_frac)
    return _closest_to_anchor(spec, flat, anchor)


def _resolved_interval(spec: ParamSpec, pool: Sequence[Tuple[float, float]],
                       flat_values: Sequence[float]) -> Tuple[float, float]:
    """The interval the sweep actually RESOLVED the flat set to.

    The flat interval spans only the evaluated flat members, but the
    sweep established nothing about the landscape between a flat edge
    and the nearest evaluated value it REJECTED - the true
    data-equivalent region can extend anywhere in that gap. Each edge of
    the resolved interval is therefore the fit-space midpoint between
    the flat edge and its nearest rejected neighbor (the edge itself
    when no rejected value exists on that side, e.g. a flat set running
    to the box bound). This is what keeps a bisection-collapsed
    single-point flat set (best SSE dropped, relative tolerance shrank,
    every other candidate fell out) from reading as zero posterior
    width: the collapse says nothing about values the sweep never
    evaluated.
    """
    z_flat = [scalar_to_fit_space(spec, v) for v in flat_values]
    z_lo, z_hi = min(z_flat), max(z_flat)
    z_rejected = [
        scalar_to_fit_space(spec, v) for v, _sse in pool
        if v not in set(flat_values)
    ]
    below = [z for z in z_rejected if z < z_lo]
    above = [z for z in z_rejected if z > z_hi]
    z_res_lo = 0.5 * (max(below) + z_lo) if below else z_lo
    z_res_hi = 0.5 * (min(above) + z_hi) if above else z_hi
    return (float(scalar_from_fit_space(spec, z_res_lo)),
            float(scalar_from_fit_space(spec, z_res_hi)))


def _grid_seed_physical_specs(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    rules: Sequence[Any],
    rule_specs: Sequence[ParamSpec],
    latent_init: Any,
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
    noise_floor: float = 0.0,
    config: Optional[SysIdConfig] = None,
) -> Tuple[List[ParamSpec], Dict[str, Dict[str, Any]]]:
    """Relocate each physical param's LM start via coordinate grid sweeps.

    The rollout SSE landscape can be flat around the declared init — for
    the domino env, topple reach saturates above friction ~0.5, so from
    an init of 0.5 the LM finite differences see no gradient and the fit
    stalls even on clean, informative data (verified 2026-07-06). The
    sweep is greedy per pass (declaration order, other params held at
    their current values; the grid plus the declared init, the anchor
    and the incumbent are always candidates), and runs up to
    ``code_sim_learning_rollout_grid_sweep_passes`` passes so a param
    swept early is re-examined after its neighbors moved, stopping as
    soon as a full pass moves nothing. Rule params are not swept (they
    fit fine locally and gridding them would explode combinatorially).

    Candidate selection is NOT the raw argmin: candidates whose SSE is
    data-equivalent to the best (:func:`_flat_candidates`) form a flat
    set, and the anchor-nearest member wins. This is the MAP choice
    within the data's resolution, and it prevents the two measured
    pathologies of argmin seeding (run_20260711_224624): a param
    compensating another's quantization error for an insignificant SSE
    gain (spinning_friction 0.5 -> 0.024, true 0.5, for 1.6%), and a
    saturated landscape reported as whichever interior grid point the
    replay chaos favored. Each moved param's anchor-side flat edge is
    then bisected to sub-grid resolution (:func:`_refine_flat_edge`),
    which is what lets a true value sitting mid-gap (lateral friction
    0.5 between grid candidates 0.342 and 0.827) be expressed at all.

    All rollout evaluations within one call are memoized (fresh-env
    rollouts are deterministic), so a converged extra pass costs no new
    rollouts. Pools from the final pass back both the returned per-param
    sweep info dict (``span`` — the sensitivity screen's raw material —
    and ``flat_interval``, the data-equivalent range in external units)
    and the refinement; a pool's held-at context can be one refinement
    tolerance stale for params refined before it, which is within the
    flat set's own resolution.
    """
    config = config or SysIdConfig.from_cfg()
    num_points = config.grid_seed_points
    num_passes = max(1, config.grid_sweep_passes)
    refine_evals = config.grid_refine_evals
    flat_frac = config.grid_flat_frac
    physical_names = [s.name for s in physical_specs]
    all_specs = list(physical_specs) + list(rule_specs)
    names = [s.name for s in all_specs]
    anchors = anchors or {}
    anchor_of = {
        s.name: float(anchors.get(s.name, s.init_value))
        for s in physical_specs
    }
    current = {
        s.name: float(anchors.get(s.name, s.init_value))
        for s in all_specs
    }

    memo: Dict[Tuple[float, ...], float] = {}

    def _sse_for(spec: ParamSpec, value: float) -> float:
        trial = dict(current)
        trial[spec.name] = float(value)
        key = tuple(trial[n] for n in names)
        if key not in memo:
            memo[key] = compute_rollout_sse(base_env, trajectories, trial,
                                            residual_features, physical_names,
                                            rules, latent_init, scaling)
        return memo[key]

    pools: Dict[str, List[Tuple[float, float]]] = {}
    for pass_idx in range(num_passes):
        moved_any = False
        for spec in physical_specs:
            assert spec.lo is not None and spec.hi is not None, \
                "Grid seeding needs bounded physical params."
            values = [float(v) for v in grid_candidates(spec, num_points)]
            values += [
                float(spec.init_value), anchor_of[spec.name],
                current[spec.name]
            ]
            pool = [(v, _sse_for(spec, v)) for v in dict.fromkeys(values)]
            pools[spec.name] = pool
            flat, _best, tol = _flat_candidates(pool, noise_floor, flat_frac)
            chosen = _closest_to_anchor(spec, flat, anchor_of[spec.name])
            sse_of = dict(pool)
            if chosen != current[spec.name]:
                logger.info(
                    "Rollout grid sweep (pass %d): %s %.4g -> %.4g "
                    "(SSE %.4g -> %.4g over %d candidates, flat tol %.3g).",
                    pass_idx + 1, spec.name, current[spec.name], chosen,
                    sse_of[current[spec.name]], sse_of[chosen], len(pool), tol)
                current[spec.name] = chosen
                moved_any = True
            else:
                raw_best = min(pool, key=lambda vs: vs[1])[0]
                if raw_best != chosen:
                    logger.info(
                        "Rollout grid sweep (pass %d): %s stays at %.4g - "
                        "best candidate %.4g is data-equivalent (SSE %.4g "
                        "vs %.4g, flat tol %.3g), so the anchor-nearest "
                        "value wins.", pass_idx + 1, spec.name, chosen,
                        raw_best, sse_of[raw_best], sse_of[chosen], tol)
        if not moved_any:
            break

    if refine_evals > 0:
        for spec in physical_specs:
            if current[spec.name] == anchor_of[spec.name]:
                continue
            refined = _refine_flat_edge(spec, pools[spec.name],
                                        anchor_of[spec.name], noise_floor,
                                        refine_evals, _sse_for, flat_frac)
            if refined != current[spec.name]:
                logger.info(
                    "Rollout grid refine: %s %.4g -> %.4g (anchor-side "
                    "flat edge at sub-grid resolution).", spec.name,
                    current[spec.name], refined)
                current[spec.name] = refined

    seeded: List[ParamSpec] = []
    sweep_info: Dict[str, Dict[str, Any]] = {}
    for spec in physical_specs:
        pool = pools[spec.name]
        sses = [sse for _v, sse in pool]
        flat, _best, _tol = _flat_candidates(pool, noise_floor, flat_frac)
        flat_values = [v for v, _sse in flat]
        sweep_info[spec.name] = {
            "span": float(max(sses) - min(sses)),
            "flat_interval":
            (float(min(flat_values)), float(max(flat_values))),
            "resolved_interval": _resolved_interval(spec, pool, flat_values),
        }
        seeded.append(
            ParamSpec(spec.name,
                      current[spec.name],
                      lo=spec.lo,
                      hi=spec.hi,
                      scale=spec.scale))
    return seeded, sweep_info


def min_explainable_rms(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    extra_candidates: Sequence[Dict[str, float]] = (),
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
    config: Optional[SysIdConfig] = None,
) -> List[float]:
    """Best achievable RMS of each trajectory over a candidate param grid.

    Explainability must be judged per trajectory against its OWN best
    parameters, not against a pooled fit: a poisoned pooled fit makes
    even a clean recording look unexplainable (measured: the clean push
    scored RMS 0.158 at a chaos-dragged fit vs 3e-4 at the true
    friction). The candidate set is the anchor point (env-registry
    baselines where revealed, declared inits otherwise) plus the same
    coordinate sweep the grid seeding uses (one physical param varied
    at a time, others held at the anchors) plus any
    ``extra_candidates``; the minimum RMS over it answers "can ANY
    reasonable parameter setting explain this recording?" — chaos
    cannot be explained by any, a clean topple is explained
    near-perfectly by the right one.

    The candidate set deliberately does NOT include the declared inits:
    anchoring it on the (per-phase stable) registry baselines makes the
    explainability verdict a function of the DATA alone, where an
    init-dependent grid flipped the same recording between explainable
    and unexplainable as the agent re-declared inits across calls
    (observed on run_20260711_141026, cycle 2).
    """
    best, _ = min_explainable_fits(base_env,
                                   trajectories,
                                   physical_specs,
                                   residual_features,
                                   rules=rules,
                                   rule_specs=rule_specs,
                                   latent_init=latent_init,
                                   extra_candidates=extra_candidates,
                                   scaling=scaling,
                                   anchors=anchors,
                                   config=config)
    return best


def min_explainable_fits(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    extra_candidates: Sequence[Dict[str, float]] = (),
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
    config: Optional[SysIdConfig] = None,
) -> Tuple[List[float], List[Dict[str, float]]]:
    """:func:`min_explainable_rms` plus each trajectory's argmin params.

    The second return is, per trajectory, the PHYSICAL portion of the
    candidate that achieved its best RMS. This is each recording's own
    preferred explanation, and disagreement between them is the honest
    uncertainty signal the consistency loop acts on: when a segment
    gets dropped, its argmin becomes a hull candidate that widens the
    physics-margin sweep instead of vanishing with the data (see
    ``fit_params_rollout_trimmed``). The sweep is unchanged, so the
    argmins are grid-resolution (coarse) - adequate for a hull bound,
    not a point estimate.
    """
    config = config or SysIdConfig.from_cfg()
    anchors = anchors or {}
    base = {
        s.name: anchors.get(s.name, s.init_value)
        for s in list(physical_specs) + list(rule_specs)
    }
    candidates: List[Dict[str, float]] = [dict(base)]
    num_points = config.grid_seed_points
    if num_points > 0:
        for spec in physical_specs:
            assert spec.lo is not None and spec.hi is not None
            for val in grid_candidates(spec, num_points):
                cand = dict(base)
                cand[spec.name] = float(val)
                candidates.append(cand)
    candidates.extend(dict(c) for c in extra_candidates)
    physical_names = [s.name for s in physical_specs]
    best = [float("inf")] * len(trajectories)
    best_params: List[Dict[str, float]] = [{
        n: float(base[n])
        for n in physical_names
    } for _ in trajectories]
    for params in candidates:
        rms = per_trajectory_rms(base_env, trajectories, params,
                                 residual_features, physical_names, rules,
                                 latent_init, scaling)
        for i, r in enumerate(rms):
            if r < best[i]:
                best[i] = r
                best_params[i] = {n: float(params[n]) for n in physical_names}
    return best, best_params
