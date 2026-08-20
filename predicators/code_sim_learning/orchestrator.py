"""One shared orchestration of the rollout system-ID flow.

The full fit pipeline (residual scaling, pre-SSE, explainability
trimming + consistency loop, identifiability report on the survivors,
trust selection) used to exist twice - once in the approach's
final-commit fit and once behind the agent's ``sim.fit`` tool - and the
two copies had to be kept in sync by hand (the ``sim_env`` omission bug
class). :func:`run_rollout_sysid` is now the single flow; the callers
only differ in what they do with the returned :class:`SysIdOutcome`
(the approach applies and records, the tool renders a report).

The orchestrator also memoizes the WHOLE fit per (artifact, data)
signature. The fit is a pure function of its inputs now that every
rollout runs in a fresh deterministic env, and agents hammer the
canonical ``sim.fit`` repeatedly within one learn phase (audited on
run_20260708_101431: 7 pipeline invocations in one phase, several on
identical artifact versions and identical data).
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    scalar_to_fit_space
from predicators.code_sim_learning.identifiability import \
    identifiability_report, select_trustworthy_params
from predicators.code_sim_learning.physical_sysid import \
    _explainability_cache_key, fit_params_rollout_trimmed
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    num_rollouts_run
from predicators.code_sim_learning.rollout_objective import compute_rollout_sse
from predicators.code_sim_learning.trajectory_prep import \
    compute_residual_scaling

logger = logging.getLogger(__name__)

# A report adjuster mutates the report in place (e.g. the approach's
# cross-cycle consistency check) before trust selection reads it. Its
# third argument is the fit's own SSE-at-theta probe (survivor set +
# shared scaling - the objective the fit minimized), None when no fit
# ran; the cross-cycle arbitration uses it to settle flagged jumps on
# evidence.
SseFn = Callable[[Dict[str, float]], float]
ReportAdjuster = Callable[
    [FitResult, Dict[str, Dict[str, Any]], Optional[SseFn]], None]


@dataclass
class SysIdOutcome:
    """Everything a caller needs from one rollout sysID run.

    ``report`` and ``applied`` are empty when no segment survived
    trimming (no fit ran; ``fit_result`` is pinned at the declared
    inits and ``post_sse`` is nan) - callers must then leave the
    planner's physical params untouched.
    """

    fit_result: FitResult
    report: Dict[str, Dict[str, Any]]
    fitted: Dict[str, float]
    applied: Dict[str, float]
    num_segments: int
    num_survivors: int
    traj_rms: List[float]
    pre_sse: float
    post_sse: float
    hull_candidates: List[Dict[str, float]] = field(default_factory=list)
    from_cache: bool = False


@dataclass
class _FitComputation:
    """The cacheable (adjuster- and selection-independent) fit core."""

    fit_result: FitResult
    report: Dict[str, Dict[str, Any]]
    num_survivors: int
    traj_rms: List[float] = field(default_factory=list)
    hull_candidates: List[Dict[str, float]] = field(default_factory=list)
    pre_sse: float = float("nan")
    post_sse: float = float("nan")
    # SSE of an arbitrary joint theta on the fit's surviving segments
    # with the fit's own scaling (the closure identifiability_report
    # consumed); None when no fit ran. Cache-safe: it closes over the
    # env FACTORY and the survivor list, both stable per cache key.
    sse_fn: Optional[SseFn] = None


def run_rollout_sysid(
    fit_env: Any,
    rollouts: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    *,
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    anchors: Optional[Dict[str, float]] = None,
    rms_cache: Optional[Dict[Tuple, Tuple[List[float],
                                          List[Dict[str, float]]]]] = None,
    fit_cache: Optional[Dict[Tuple, _FitComputation]] = None,
    fit_cache_key: Optional[Any] = None,
    report_adjuster: Optional[ReportAdjuster] = None,
    held: Optional[Dict[str, float]] = None,
    config: Optional[SysIdConfig] = None,
) -> SysIdOutcome:
    """Run the full rollout sysID flow on recorded trajectories.

    ``fit_cache``/``fit_cache_key`` memoize the fit core: the key is
    ``fit_cache_key`` (an opaque artifact identity, e.g. the snapshot
    version tag - the RULES' identity, which the data signature cannot
    see) combined with the same declaration/data signature the
    explainability cache uses. Both must be provided for caching;
    callers fitting on data SUBSETS must pass a key that reflects the
    subset (or no cache), for the same reason the explainability cache
    excludes them. On a hit, the fit core is reused and only the
    (cheap, caller-dependent) adjuster + trust selection re-run - on a
    deep copy of the report, so one caller's adjustments never leak
    into another's.

    ``report_adjuster`` runs after the report is built and before trust
    selection (the approach's cross-cycle consistency check plugs in
    here); ``held`` is the currently-deployed value per param, consumed
    by the INCONSISTENT hold policy in
    :func:`select_trustworthy_params`.
    """
    config = config or SysIdConfig.from_cfg()
    anchors = anchors or {}
    t0 = time.monotonic()
    n0 = num_rollouts_run()
    physical_names = [s.name for s in physical_specs]
    init_params = {
        s.name: s.init_value
        for s in list(physical_specs) + list(rule_specs)
    }

    cache_key: Optional[Tuple] = None
    core: Optional[_FitComputation] = None
    if fit_cache is not None and fit_cache_key is not None:
        scaling_for_key = compute_residual_scaling(rollouts,
                                                   residual_features,
                                                   config=config)
        cache_key = (fit_cache_key,
                     _explainability_cache_key(physical_specs, rule_specs,
                                               rollouts, anchors,
                                               scaling_for_key,
                                               config.grid_seed_points))
        core = fit_cache.get(cache_key)
        if core is not None:
            logger.info("Rollout sysID: reusing the cached fit for this "
                        "artifact/data signature (no rollouts spent).")

    from_cache = core is not None
    if core is None:
        core = _compute_fit(fit_env, rollouts, physical_specs,
                            residual_features, rules, rule_specs, latent_init,
                            anchors, rms_cache, config)
        if fit_cache is not None and cache_key is not None:
            fit_cache[cache_key] = core

    fitted = dict(core.fit_result.point_estimate)
    report = copy.deepcopy(core.report)
    applied: Dict[str, float] = {}
    if core.num_survivors > 0:
        if report_adjuster is not None:
            report_adjuster(core.fit_result, report, core.sse_fn)
        applied = select_trustworthy_params(fitted,
                                            init_params,
                                            physical_names,
                                            report,
                                            anchors,
                                            held=held)
        _log_data_health(report, physical_specs)
    logger.info("Rollout sysID orchestration total: %d rollouts in %.1fs%s.",
                num_rollouts_run() - n0,
                time.monotonic() - t0,
                " (fit cache hit)" if from_cache else "")
    return SysIdOutcome(fit_result=core.fit_result,
                        report=report,
                        fitted=fitted,
                        applied=applied,
                        num_segments=len(rollouts),
                        num_survivors=core.num_survivors,
                        traj_rms=list(core.traj_rms),
                        pre_sse=core.pre_sse,
                        post_sse=core.post_sse,
                        hull_candidates=list(core.hull_candidates),
                        from_cache=from_cache)


def _log_data_health(report: Dict[str, Dict[str, Any]],
                     physical_specs: Sequence[ParamSpec]) -> None:
    """Warn when candidate fits disagree far beyond the posterior width.

    The disagreement hull (per-segment fits recorded by the consistency
    loop, plus any cross-cycle candidates an adjuster appended) is the
    honest uncertainty of this fit. When its half-span dwarfs the
    reported posterior_std, the recordings prefer mutually-incompatible
    explanations - typically context-corrupted data (replans,
    reset-reconstruction deltas, warm-env micro-state) rather than a
    genuinely sharp posterior - and the point estimate should not be
    trusted to posterior_std precision. The margin sweep already covers
    the hull (see :func:`identifiability.physics_sigma_points`); this
    log line makes the condition visible to run audits.
    """
    spec_by_name = {s.name: s for s in physical_specs}
    for name, entry in report.items():
        spec = spec_by_name.get(name)
        cands = entry.get("candidate_values")
        if spec is None or not cands:
            continue
        width = float(entry.get("posterior_std", float("nan")))
        zs = [scalar_to_fit_space(spec, float(v)) for v in cands]
        half_span = (max(zs) - min(zs)) / 2.0
        if np.isfinite(width) and width > 0 and half_span > 2.0 * width:
            logger.warning(
                "Rollout sysID data-health: %s candidate fits span "
                "[%.4g, %.4g] (half-span %.3g in fit space vs "
                "posterior_std %.3g) - the recordings prefer "
                "mutually-incompatible explanations; the physics-margin "
                "sweep covers the whole hull.", name, min(cands), max(cands),
                half_span, width)


def _compute_fit(
    fit_env: Any,
    rollouts: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    rules: Sequence[Any],
    rule_specs: Sequence[ParamSpec],
    latent_init: Any,
    anchors: Dict[str, float],
    rms_cache: Optional[Dict[Tuple, Tuple[List[float], List[Dict[str,
                                                                 float]]]]],
    config: SysIdConfig,
) -> _FitComputation:
    """The cacheable fit core: trim + fit + report on the survivors."""
    physical_names = [s.name for s in physical_specs]
    init_params = {
        s.name: s.init_value
        for s in list(physical_specs) + list(rule_specs)
    }
    # One scaling object per fit: every SSE/RMS below must share it or
    # their values are incomparable.
    scaling = compute_residual_scaling(rollouts,
                                       residual_features,
                                       config=config)
    pre_sse = compute_rollout_sse(fit_env, rollouts, init_params,
                                  residual_features, physical_names, rules,
                                  latent_init, scaling)
    logger.info("Rollout sysID - pre-SSE: %.6f", pre_sse)
    result, survivors, rms, hull_candidates = fit_params_rollout_trimmed(
        fit_env,
        rollouts,
        physical_specs,
        residual_features,
        rules=rules,
        rule_specs=rule_specs,
        latent_init=latent_init,
        scaling=scaling,
        anchors=anchors,
        rms_cache=rms_cache,
        config=config)
    if not survivors:
        # NO fit ran (result is pinned at the declared inits). The
        # report and applied set stay empty so nothing computed on zero
        # surviving data can leak to the planner (chaos makes any
        # identifiability probe report "identified" for everything).
        return _FitComputation(fit_result=result,
                               report={},
                               num_survivors=0,
                               traj_rms=list(rms),
                               pre_sse=pre_sse,
                               post_sse=float("nan"))
    # Post-SSE and the identifiability report are computed on the
    # SURVIVING trajectories: a trimmed (unexplainable) recording would
    # otherwise re-poison the verdicts with its noise.
    fitted = result.point_estimate
    post_sse = compute_rollout_sse(fit_env, survivors, fitted,
                                   residual_features, physical_names, rules,
                                   latent_init, scaling)
    logger.info("Rollout sysID - post-SSE: %.6f", post_sse)

    def rollout_sse_fn(params: Dict[str, float]) -> float:
        return compute_rollout_sse(fit_env, survivors, params,
                                   residual_features, physical_names, rules,
                                   latent_init, scaling)

    report = identifiability_report(
        result,
        rollout_sse_fn,
        list(physical_specs) + list(rule_specs),
        num_explainable=len(survivors),
        min_posterior_width=(config.min_posterior_width))
    # The consistency loop's disagreement hull rides on the report so
    # every consumer of the fit (margin sweep, diagnostics, agents
    # reading sim.fit output) sees the same uncertainty evidence.
    for name in physical_names:
        vals = sorted({float(c[name])
                       for c in hull_candidates if name in c}
                      | {float(fitted[name])})
        if len(vals) > 1 and name in report:
            report[name]["candidate_values"] = vals
    return _FitComputation(fit_result=result,
                           report=report,
                           num_survivors=len(survivors),
                           traj_rms=list(rms),
                           hull_candidates=list(hull_candidates),
                           pre_sse=pre_sse,
                           post_sse=post_sse,
                           sse_fn=rollout_sse_fn)
