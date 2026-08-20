"""System identification of PyBullet physical parameters by rollout matching.

The residual-rule fitting in :mod:`predicators.code_sim_learning.fitting` is
*teacher-forced and single-step*: it resets the base sim to each observed
``s_t`` and predicts one step. That is correct for slow residual features
(heating, filling) but wrong for **momentum-driven** dynamics such as a domino
cascade: the :class:`~predicators.structs.State` carries pose but no velocity,
so resetting to a mid-cascade state discards the angular momentum that produced
the next step, and the one-step prediction systematically under-rotates. No
friction/restitution value can repair that mismatch — physical parameters are
*invisible* to the teacher-forced objective.

This stack fits such parameters by matching a **free-running rollout**
instead: reset *once* to a trajectory's at-rest initial state, roll the base
sim forward under the recorded action sequence (so momentum accrues in-sim),
and compare the full per-step pose trajectory. Because the base sim and the
real environment are the same engine differing only in the fitted scalars, the
sum-of-squared-errors is ~0 at the true parameters, giving a well-specified
identification problem.

Design notes (mirroring MuJoCo's official ``mujoco.sysid`` toolbox):

* The agent declares a **sparse subset** of the parameters the env reveals
  (``env.get_physical_param_info()``) as ``PHYSICAL_PARAMS`` — never "all
  params by default". Undeclared parameters keep the env's built-in values.
* Physical parameters and learned-rule parameters are fit **jointly** in one
  posterior (one theta vector, one fit) so rules cannot silently absorb
  physics error and vice versa. With no rules the fit degenerates to pure
  physical identification; rule-only artifacts keep using the (cheaper)
  teacher-forced / recurrent objectives in ``fitting.py``.
* Non-identifiability is *reported*, not regularized away: the posterior
  contraction per parameter (:func:`identifiability_report`) is surfaced to
  the agent so it can drop null parameters from its declaration.

The interface deliberately mirrors :func:`training.fit_params` (declare an
initialization via :class:`ParamSpec`, a Gaussian prior around it, a Gaussian
likelihood) so the agent-facing flow is unchanged: pick an init, let the
solver refine it.

The stack is split across subsystem modules; this module keeps the fit
orchestrators and re-exports the public API so existing importers keep
working:

* :mod:`.config` - :class:`SysIdConfig`, the frozen snapshot of the
  ``code_sim_learning_*`` flags (resolved from ``CFG`` at entry time).
* :mod:`.rollout_env` - env-facing rollout plumbing:
  ``RolloutTrajectory``, :func:`rollout_states`, :func:`dispose_env`,
  :func:`physical_param_anchors`.
* :mod:`.trajectory_prep` - :func:`truncate_settled_tail`,
  :func:`split_at_rest_points`, :class:`ResidualScaling`,
  :func:`compute_residual_scaling`.
* :mod:`.rollout_objective` - :func:`compute_rollout_sse`,
  :func:`compute_rollout_residuals`, :func:`per_trajectory_rms`,
  :func:`fit_map_lm_rollout`.
* :mod:`.grid_seed` - the coordinate grid sweeps (LM-seed relocation and
  the :func:`min_explainable_rms` explainability sweep).
* :mod:`.identifiability` - :func:`identifiability_report`,
  :func:`select_trustworthy_params`, :func:`format_identifiability`.
* This module - :func:`fit_params_rollout` (grid seed + LM MAP + anchor
  ablation) and :func:`fit_params_rollout_trimmed` (explainability
  trimming + consistency loop).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    prior_widths, to_fit_space
from predicators.code_sim_learning.grid_seed import \
    _grid_seed_physical_specs, min_explainable_fits, min_explainable_rms
from predicators.code_sim_learning.identifiability import NOISE_FLOOR_EVALS, \
    format_identifiability, identifiability_report, \
    select_trustworthy_params
from predicators.code_sim_learning.lm import log_hessian_identifiability
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    dispose_env, num_rollouts_run, physical_param_anchors, rollout_states
from predicators.code_sim_learning.rollout_objective import \
    compute_rollout_residuals, compute_rollout_sse, fit_map_lm_rollout, \
    per_trajectory_rms
from predicators.code_sim_learning.trajectory_prep import ResidualScaling, \
    compute_residual_scaling, split_at_rest_points, truncate_settled_tail

logger = logging.getLogger(__name__)

# Re-exported public API of the split modules (see the module map in the
# docstring), so existing importers of this module keep working.
__all__ = [
    "RolloutTrajectory",
    "ResidualScaling",
    "SysIdConfig",
    "compute_residual_scaling",
    "compute_rollout_residuals",
    "compute_rollout_sse",
    "dispose_env",
    "fit_map_lm_rollout",
    "fit_params_rollout",
    "fit_params_rollout_trimmed",
    "format_identifiability",
    "identifiability_report",
    "min_explainable_fits",
    "min_explainable_rms",
    "per_trajectory_rms",
    "physical_param_anchors",
    "rollout_states",
    "select_trustworthy_params",
    "split_at_rest_points",
    "truncate_settled_tail",
]

# Prior width as a fraction of each param's box; shared by the rollout
# fit default and the pinned-at-init fallback result so the two cannot
# silently diverge.
_ROLLOUT_PRIOR_SIGMA_SCALE = 0.75

# Absolute RMS slack in the trim-consistency test, so exact ties and
# floating-point jitter never count as violations.
_CONSISTENCY_RMS_EPS = 1e-3


def fit_params_rollout(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    noise_sigma: float = 0.05,
    prior_sigma_scale: float = _ROLLOUT_PRIOR_SIGMA_SCALE,
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
    config: Optional[SysIdConfig] = None,
) -> FitResult:
    """Jointly identify physical + rule params against rollout SSE.

    The fit is a grid-seeded, prior-folded Levenberg-Marquardt MAP
    point fit - always. (A rollout emcee branch existed historically
    behind ``code_sim_learning_rollout_num_mcmc_steps`` but was never
    enabled by any experiment; it was removed 2026-07 rather than
    carried as dead scaffolding. The identifiability verdicts come from
    the grid landscape and the curvature probe, and the info-seeking
    explorer's calibrated ensemble comes from the Laplace bundle at the
    LM MAP.) The forward model is a base-sim rollout
    (:func:`compute_rollout_sse`) rather than the per-step process
    rules, and theta concatenates ``physical_specs`` with
    ``rule_specs``.

    The Gaussian prior is centered on each param's ``anchors`` entry
    (the env-registry baseline, see :func:`physical_param_anchors`)
    when given, else on the declared init, and - unlike the historical
    likelihood-only LM - is folded into the LM objective itself
    (:func:`fit_map_lm_rollout`), so data-flat directions stay at their
    anchors instead of drifting on rollout noise.

    When the grid sweep ran, each physical param's sweep diagnostics
    (SSE span, same-theta noise floor from 3 repeated evaluations at
    the anchor point, and the data-equivalent ``flat_interval``) are
    attached as ``FitResult.sensitivity``. With
    ``code_sim_learning_rollout_sensitivity_factor`` > 0 they also
    drive the pre-fit sensitivity screen: a param whose span does not
    clear the noise floor is "insensitive" and its fitted value is
    withheld by :func:`select_trustworthy_params`.

    Serial only: a shared ``base_env`` instance is mutated per
    evaluation (and a factory-built fresh env lives inside one
    :func:`rollout_states` call), so no multiprocessing pool may be
    used.
    """
    config = config or SysIdConfig.from_cfg()
    all_specs = list(physical_specs) + list(rule_specs)
    assert all_specs, "fit_params_rollout needs at least one ParamSpec."
    physical_names = [s.name for s in physical_specs]
    names = [s.name for s in all_specs]
    scales = [getattr(s, "scale", "linear") for s in all_specs]
    anchors = anchors or {}
    center_values = np.array(
        [anchors.get(s.name, s.init_value) for s in all_specs], dtype=float)
    # Prior widths are computed at the CENTER values (the anchor spec),
    # so a linear param's width scales with its anchor, not with
    # whatever init the agent declared this call.
    center_specs = [
        ParamSpec(s.name,
                  float(c),
                  lo=s.lo,
                  hi=s.hi,
                  scale=getattr(s, "scale", "linear"))
        for s, c in zip(all_specs, center_values)
    ]
    center_int = to_fit_space(all_specs, center_values)
    prior_sigma = prior_widths(center_specs, prior_sigma_scale)

    fit_t0 = time.monotonic()
    n_start = num_rollouts_run()

    # Coarse grid sweep to place the LM start in the right basin (see
    # _grid_seed_physical_specs for why LM alone can stall). Also
    # yields the per-param SSE spans and data-equivalent flat intervals
    # the sensitivity screen and the identifiability report consume.
    lm_physical_specs = list(physical_specs)
    sensitivity: Optional[Dict[str, Dict[str, Any]]] = None
    noise_floor: Optional[float] = None
    if (config.grid_seed_points > 0 and trajectories):
        # Same-theta noise floor at the anchor point (3 repeated evals).
        # Fresh-env rollouts are deterministic so this is normally 0.0;
        # it bounds the sweep's flat tolerance and the sensitivity
        # screen from below if nondeterminism ever returns.
        anchor_point = {
            s.name: anchors.get(s.name, s.init_value)
            for s in all_specs
        }
        floor_evals = [
            compute_rollout_sse(base_env, trajectories, anchor_point,
                                residual_features, physical_names, rules,
                                latent_init, scaling)
            for _ in range(NOISE_FLOOR_EVALS)
        ]
        noise_floor = float(np.max(floor_evals) - np.min(floor_evals))
        lm_physical_specs, sweep_info = _grid_seed_physical_specs(
            base_env,
            trajectories,
            physical_specs,
            residual_features,
            rules,
            rule_specs,
            latent_init,
            scaling=scaling,
            anchors=anchors,
            noise_floor=noise_floor,
            config=config)
        sens_factor = config.sensitivity_factor
        sensitivity = {}
        for name, info in sweep_info.items():
            entry: Dict[str, Any] = {
                "sse_span": info["span"],
                "noise_floor": noise_floor,
                "flat_interval": info["flat_interval"],
                "resolved_interval": info["resolved_interval"],
            }
            if sens_factor > 0:
                entry["sensitive"] = (info["span"] >
                                      sens_factor * max(noise_floor, 1e-12))
            sensitivity[name] = entry
        insensitive = sorted(n for n, d in sensitivity.items()
                             if not d.get("sensitive", True))
        if insensitive:
            logger.info(
                "Rollout sysID sensitivity screen: rollouts do not "
                "respond to %s anywhere in their boxes on this data "
                "(SSE span vs %.1f x noise floor %.4g); their fitted "
                "values will not be applied.", insensitive, sens_factor,
                noise_floor)

    n_grid = num_rollouts_run() - n_start

    # The grid-seeded, prior-folded LM MAP IS the fit. The Jacobian at
    # the MAP is kept on the result as the Laplace bundle for the
    # info-seeking explorer's calibrated ensemble.
    lm_theta, lm_jac = fit_map_lm_rollout(base_env,
                                          trajectories,
                                          lm_physical_specs,
                                          residual_features,
                                          rules,
                                          rule_specs,
                                          latent_init,
                                          scaling=scaling,
                                          prior_centers=center_int,
                                          prior_sigmas=prior_sigma,
                                          noise_sigma=noise_sigma)
    if (config.log_hessian_identifiability and lm_jac is not None
            and lm_jac.size > 0):
        log_hessian_identifiability(lm_jac, names, noise_sigma, prior_sigma)
    result = FitResult(names=names,
                       samples=np.asarray(lm_theta, dtype=float)[None, :],
                       log_probs=np.zeros(1),
                       jacobian=lm_jac,
                       noise_sigma=noise_sigma,
                       prior_sigma=prior_sigma,
                       scales=scales,
                       sensitivity=sensitivity)
    n_lm = num_rollouts_run() - n_start - n_grid
    if (config.anchor_ablation and config.grid_flat_frac > 0 and trajectories):
        result = _anchor_backward_elimination(
            base_env,
            trajectories,
            physical_specs,
            rule_specs,
            residual_features,
            rules,
            latent_init,
            scaling,
            anchors,
            noise_sigma,
            prior_sigma_scale,
            result,
            noise_floor,
            config,
        )
    n_total = num_rollouts_run() - n_start
    if trajectories:
        logger.info(
            "Rollout sysID fit cost: %d rollouts (grid+floor %d, LM %d, "
            "ablation %d) in %.1fs.", n_total, n_grid, n_lm,
            n_total - n_grid - n_lm,
            time.monotonic() - fit_t0)
    return result


# A MAP within this fit-space distance of its anchor counts as unmoved
# (the prior-folded LM keeps data-flat directions exactly at the anchor,
# so this only absorbs float noise).
_ANCHOR_ABLATION_EPS = 1e-9


def _anchor_backward_elimination(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    rule_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    rules: Sequence[Any],
    latent_init: Any,
    scaling: Optional[ResidualScaling],
    anchors: Dict[str, float],
    noise_sigma: float,
    prior_sigma_scale: float,
    result: FitResult,
    noise_floor: Optional[float],
    config: SysIdConfig,
) -> FitResult:
    """Revert compensatory physical-param moves via anchor-pinned refits.

    The curvature probe measures LOCAL precision at the joint MAP: a
    co-adapted parameter set has real curvature in every direction, so
    a param that only moved to compensate another's overshoot still
    reads "identified" (run_20260721_205821 seed1: the coordinate sweep
    overshot lateral_friction past its true value, then restitution and
    spinning_friction moved 26x / 23x off their true values to
    compensate - all three "identified", and the resulting belief sim
    invalidated every downstream plan). The global test the probe
    cannot make: is there a DATA-EQUIVALENT solution with this param at
    its env-registry anchor? Greedy backward elimination answers it -
    for each moved param, refit the remaining params (warm-started at
    the current MAP, priors still centered on the anchors) with that
    param pinned at its anchor; accept a refit only when it is BOTH
    data-equivalent (SSE within the grid flat set's tolerance,
    ``max(noise_floor, grid_flat_frac * SSE)``) AND strictly closer to
    the standing belief (lower total fit-space prior cost) - the grid's
    anchor-nearest flat-set principle applied jointly, which also stops
    a symmetric ridge from merely swapping WHICH param carries the
    move. Among acceptable refits the one nearest the belief wins;
    repeat on the reduced set. A genuinely identified param is never
    touched: pinning it destroys the SSE by construction. Pinned params
    re-enter the returned result AT their anchors, listed in
    ``FitResult.anchor_ablation`` so the identifiability report renders
    them "anchored" instead of "identified".

    The returned result drops the Jacobian when anything was pinned
    (the reduced refit's columns no longer match the full param list);
    Laplace-ensemble consumers already handle ``jacobian=None``.
    """
    all_specs = list(physical_specs) + list(rule_specs)
    orig_by_name = {s.name: s for s in all_specs}
    point = dict(result.point_estimate)
    insensitive = {
        n
        for n, d in (result.sensitivity or {}).items()
        if not d.get("sensitive", True)
    }

    def sse_at(params: Dict[str, float], declared: List[str]) -> float:
        return compute_rollout_sse(base_env, trajectories, params,
                                   residual_features, declared, rules,
                                   latent_init, scaling)

    assert list(result.names) == [s.name for s in all_specs]
    assert result.prior_sigma is not None
    prior_sig = result.prior_sigma
    belief_values = np.array(
        [anchors.get(s.name, s.init_value) for s in all_specs], dtype=float)
    belief_z = to_fit_space(all_specs, belief_values)

    def prior_cost(pt: Dict[str, float]) -> float:
        """Total fit-space prior cost of a full param point (all specs)."""
        z = to_fit_space(all_specs, [pt[s.name] for s in all_specs])
        return float(np.sum(((z - belief_z) / prior_sig)**2))

    def refit_pinned(surviving: List[ParamSpec],
                     pins: Dict[str, float]) -> Dict[str, float]:
        """LM refit of the surviving physical + all rule params.

        Warm-started at the current point, with ``pins`` (the ablated
        params) held EXPLICITLY at their anchor values throughout the
        fit - so the SSE that justifies a revert is measured at exactly
        the value recorded and applied, whatever the env registry's
        defaults.
        """
        if not surviving and not rule_specs:
            # Nothing left to refit: the candidate is the pins alone.
            # (Calling LM with zero specs raises on the empty theta -
            # observed as 'zero-size array to reduction operation' on
            # run_20260722_123949 seed2 when 4 of 5 params ablated.)
            return {}
        warm_physical = [
            ParamSpec(s.name,
                      float(point[s.name]),
                      lo=s.lo,
                      hi=s.hi,
                      scale=getattr(s, "scale", "linear")) for s in surviving
        ]
        warm_rules = [
            ParamSpec(s.name,
                      float(point[s.name]),
                      lo=s.lo,
                      hi=s.hi,
                      scale=getattr(s, "scale", "linear")) for s in rule_specs
        ]
        specs = warm_physical + warm_rules
        # Prior centers stay at the anchors (declared inits for rule
        # params), NOT the warm starts, mirroring the main fit.
        center_values = np.array([
            anchors.get(s.name, orig_by_name[s.name].init_value) for s in specs
        ],
                                 dtype=float)
        center_specs = [
            ParamSpec(s.name,
                      float(c),
                      lo=s.lo,
                      hi=s.hi,
                      scale=getattr(s, "scale", "linear"))
            for s, c in zip(specs, center_values)
        ]
        theta, _ = fit_map_lm_rollout(
            base_env,
            trajectories,
            warm_physical,
            residual_features,
            rules,
            warm_rules,
            latent_init,
            scaling=scaling,
            prior_centers=to_fit_space(specs, center_values),
            prior_sigmas=prior_widths(center_specs, prior_sigma_scale),
            noise_sigma=noise_sigma,
            fixed_physical=pins)
        return {s.name: float(v) for s, v in zip(specs, theta)}

    surviving = list(physical_specs)
    declared = [s.name for s in surviving]
    sse_curr = sse_at(point, declared)
    if noise_floor is None:
        floor_evals = [
            sse_at(point, declared) for _ in range(NOISE_FLOOR_EVALS - 1)
        ] + [sse_curr]
        noise_floor = float(np.max(floor_evals) - np.min(floor_evals))

    pinned: Dict[str, Dict[str, float]] = {}
    pinned_values: Dict[str, float] = {}
    while True:
        movable = []
        for s in surviving:
            if s.name in insensitive or s.name not in anchors:
                # Insensitive values are withheld anyway; a param
                # without a declared anchor has no baseline to pin to.
                continue
            dist = abs(
                to_fit_space([s], [point[s.name]])[0] -
                to_fit_space([s], [anchors[s.name]])[0])
            if dist > _ANCHOR_ABLATION_EPS:
                movable.append(s)
        if not movable:
            break
        tol = max(noise_floor, config.grid_flat_frac * sse_curr)
        cost_curr = prior_cost(point)
        best: Optional[Tuple[float, ParamSpec, Dict[str, float]]] = None
        best_cost = float("inf")
        for s in movable:
            reduced = [t for t in surviving if t is not s]
            pins = dict(pinned_values)
            pins[s.name] = float(anchors[s.name])
            declared = [t.name for t in reduced] + list(pins)
            # Cheap pre-test first: pin the param with everything else
            # UNCHANGED (one SSE eval). When the move was a small
            # compensatory drift - the common case, e.g. spinning
            # 0.4993 -> 0.5 on run_20260722_123949 seed2 - this alone
            # is data-equivalent and the LM refit (~a dozen full-
            # trajectory evals) is skipped. The refit still runs when
            # the cheap test fails, because data-equivalence may only
            # emerge after the OTHER params re-adjust.
            cheap_fit = {t.name: float(point[t.name]) for t in reduced}
            cheap_fit.update(
                {r.name: float(point[r.name])
                 for r in rule_specs})
            cand_sse = sse_at({**cheap_fit, **pins}, declared)
            if cand_sse <= sse_curr + tol:
                cand_fit = cheap_fit
            else:
                cand_fit = refit_pinned(reduced, pins)
                cand_sse = sse_at({**cand_fit, **pins}, declared)
            if cand_sse > sse_curr + tol:
                continue
            cand_point = dict(point)
            cand_point.update(cand_fit)
            cand_point[s.name] = float(anchors[s.name])
            cand_cost = prior_cost(cand_point)
            # Data-equivalent alone is not enough: on a symmetric ridge
            # it would merely swap WHICH param carries the move. Require
            # the refit to be strictly closer to the standing belief.
            if cand_cost + 1e-9 >= cost_curr:
                continue
            if cand_cost < best_cost:
                best = (cand_sse, s, cand_point)
                best_cost = cand_cost
        if best is None:
            break
        cand_sse, spec, cand_point = best
        anchor = anchors[spec.name]
        logger.info(
            "Rollout sysID anchor ablation: %s %.4g -> baseline %.4g is "
            "data-equivalent (SSE %.4g vs %.4g at the joint MAP, tol "
            "%.4g) and nearer the standing belief - the move was "
            "compensatory, reverting it.", spec.name, point[spec.name], anchor,
            cand_sse, sse_curr, tol)
        pinned[spec.name] = {
            "anchor": float(anchor),
            "fitted": float(point[spec.name]),
            "sse_map": float(sse_curr),
            "sse_pinned": float(cand_sse),
            "tol": float(tol),
        }
        pinned_values[spec.name] = float(anchor)
        surviving = [t for t in surviving if t is not spec]
        point = cand_point
        # The baseline stays at the running minimum ON PURPOSE: each
        # accepted revert may cost up to tol, and re-baselining to the
        # reduced point would let successive reverts drift the SSE
        # arbitrarily far in tol-sized steps. Anchoring to the minimum
        # bounds the CUMULATIVE degradation to ~tol of the original MAP
        # (conservative: a borderline second revert may be kept fitted).
        sse_curr = min(sse_curr, cand_sse)
    if not pinned:
        return result
    values = np.array([point[n] for n in result.names], dtype=float)
    return FitResult(names=list(result.names),
                     samples=values[None, :],
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=result.noise_sigma,
                     prior_sigma=result.prior_sigma,
                     scales=result.scales,
                     sensitivity=result.sensitivity,
                     anchor_ablation=pinned)


def _init_point_fit_result(all_specs: Sequence[ParamSpec],
                           noise_sigma: float) -> FitResult:
    """A single-sample FitResult pinned at the declared init values.

    Returned when trimming rejects every trajectory: no explainable data
    exists, so no fitted value — not even the pooled fit's — may leak to
    the caller. The identifiability probe on chaotic data can falsely
    report "identified" (chaos responds to everything at prior scale),
    so the guard alone is not sufficient protection; pinning the point
    estimate at the inits makes application a no-op regardless of the
    verdicts.
    """
    init_values = np.array([s.init_value for s in all_specs], dtype=float)
    prior_sigma = prior_widths(list(all_specs), _ROLLOUT_PRIOR_SIGMA_SCALE)
    return FitResult(names=[s.name for s in all_specs],
                     samples=init_values[None, :],
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=noise_sigma,
                     prior_sigma=prior_sigma,
                     scales=[getattr(s, "scale", "linear") for s in all_specs])


def _explainability_cache_key(
    physical_specs: Sequence[ParamSpec],
    rule_specs: Sequence[ParamSpec],
    trajectories: List[RolloutTrajectory],
    anchors: Dict[str, float],
    scaling: Optional[ResidualScaling],
    grid_seed_points: int,
) -> Tuple:
    """Cache key for :func:`min_explainable_rms` verdicts within one phase.

    Everything the candidate grid and the scored data depend on: spec
    boxes/scales, anchors, grid resolution, the residual scaling, and
    the segment shapes. Rule-spec INIT values matter (rule params are
    not swept, so candidates hold them at their anchor-or-init).
    Trajectory identity is approximated by per-segment lengths - exact
    within one learn phase ONLY when every cache-sharing caller fits on
    the same full recording set (fixed recordings, deterministic
    segmentation). Callers fitting on data subsets (e.g. exploratory
    ``sim.fit(traj_idxs=...)``) must NOT share the cache: different
    subsets with equal-length segments collide.
    """
    return (
        tuple((s.name, s.lo, s.hi, getattr(s, "scale", "linear"))
              for s in physical_specs),
        tuple((s.name, s.init_value) for s in rule_specs),
        tuple(sorted(anchors.items())),
        grid_seed_points,
        scaling.signature() if scaling is not None else None,
        tuple(len(actions) for _states, actions in trajectories),
    )


def fit_params_rollout_trimmed(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    noise_sigma: float = 0.05,
    scaling: Optional[ResidualScaling] = None,
    anchors: Optional[Dict[str, float]] = None,
    rms_cache: Optional[Dict[Tuple, Tuple[List[float],
                                          List[Dict[str, float]]]]] = None,
    config: Optional[SysIdConfig] = None,
) -> Tuple[FitResult, List[RolloutTrajectory], List[float], List[Dict[str,
                                                                      float]]]:
    """Drop trajectories no parameters can explain, fit on the rest.

    Goodness-of-fit trimming: a chaotic recording (e.g. a center-height
    scraping push whose 500-step outcome is contact-chaos) has a large
    rollout residual at EVERY parameter value — it is unexplainable, and
    pooling it with clean recordings lets its parameter-independent
    noise outvote their signal (measured on run_20260706_111805: one
    such trajectory dragged the pooled friction fit to 0.34 vs the true
    0.1 that the clean trajectory alone recovers). Each trajectory's
    best-achievable RMS over a candidate param grid
    (:func:`min_explainable_rms` — judged against its own best params,
    NOT a pooled fit, which chaos poisons) is compared against
    ``noise_sigma`` scaled by
    ``CFG.code_sim_learning_rollout_trim_rms_factor``; unexplainable
    recordings are dropped before the fit ever sees them.

    ``scaling`` defaults to :func:`compute_residual_scaling` over the
    input trajectories, so the trimming threshold compares
    dimensionless RMS values. ``rms_cache`` (a caller-owned dict, e.g.
    per learn phase) memoizes the explainability sweep - the most
    expensive part of repeated ``sim.fit`` calls - under a
    key that captures everything the verdict depends on, which both
    saves rollouts and pins the verdict for identical inputs.

    Returns ``(fit_result, surviving_trajectories, per_trajectory_rms,
    hull_candidates)`` — callers must compute their post-fit SSE /
    identifiability probe on the SURVIVORS, or a rejected trajectory's
    noise re-poisons the verdicts. If nothing survives, the survivor
    list is empty and the returned fit is pinned at the declared inits
    (see :func:`_init_point_fit_result`), so applying it is a no-op.

    ``hull_candidates`` records the disagreement the consistency loop
    resolved: whenever a survivor is dropped, the pre-drop joint fit
    and the dropped segment's own-best (grid-argmin) physical values
    are appended. The drop still anchors the POINT estimate on the
    cleanest data (dropping is right when a chaotic recording is
    accidentally explainable at wrong params - measured on
    run_20260706_111805), but the disagreement must survive as
    UNCERTAINTY: run_20260724_232411 seed1 dropped the solved cascade
    segment and shipped lateral_friction 1.0358 at the 0.1 sigma floor
    when the segment fits spanned [~0.27, ~1.04] around the true 0.5.
    Consumers fold these candidates into the physics-margin sweep (see
    :func:`identifiability.physics_sigma_points`), so what the drop
    removes from the mean reappears in the variance.
    """
    config = config or SysIdConfig.from_cfg()
    factor = config.trim_rms_factor
    all_specs = list(physical_specs) + list(rule_specs)
    if scaling is None:
        scaling = compute_residual_scaling(trajectories,
                                           residual_features,
                                           config=config)
    anchors = anchors or {}
    if not trajectories or factor <= 0:
        result = fit_params_rollout(base_env,
                                    trajectories,
                                    physical_specs,
                                    residual_features,
                                    rules=rules,
                                    rule_specs=rule_specs,
                                    latent_init=latent_init,
                                    noise_sigma=noise_sigma,
                                    scaling=scaling,
                                    anchors=anchors,
                                    config=config)
        return result, list(trajectories), [], []
    cache_key: Optional[Tuple] = None
    cached: Optional[Tuple[List[float], List[Dict[str, float]]]] = None
    if rms_cache is not None:
        cache_key = _explainability_cache_key(physical_specs, rule_specs,
                                              trajectories, anchors, scaling,
                                              config.grid_seed_points)
        cached = rms_cache.get(cache_key)
        if cached is not None:
            logger.info(
                "Rollout sysID trimming: reusing cached explainability "
                "verdicts for this declaration/data signature.")
    sweep_t0 = time.monotonic()
    sweep_n0 = num_rollouts_run()
    if cached is None:
        cached = min_explainable_fits(base_env,
                                      trajectories,
                                      physical_specs,
                                      residual_features,
                                      rules=rules,
                                      rule_specs=rule_specs,
                                      latent_init=latent_init,
                                      scaling=scaling,
                                      anchors=anchors,
                                      config=config)
        if rms_cache is not None and cache_key is not None:
            rms_cache[cache_key] = cached
        logger.info(
            "Rollout sysID explainability sweep cost: %d rollouts in "
            "%.1fs.",
            num_rollouts_run() - sweep_n0,
            time.monotonic() - sweep_t0)
    rms, argmins = cached
    threshold = factor * noise_sigma
    survivors = [t for t, r in zip(trajectories, rms) if r <= threshold]
    surv_argmins = [a for a, r in zip(argmins, rms) if r <= threshold]
    if len(survivors) < len(trajectories):
        logger.info(
            "Rollout sysID trimming: per-trajectory best RMS %s vs "
            "threshold %.4f (%g x noise %.3f) — dropping %d of %d "
            "unexplainable trajectories.", [f"{r:.4g}" for r in rms],
            threshold, factor, noise_sigma,
            len(trajectories) - len(survivors), len(trajectories))
    if not survivors:
        logger.warning(
            "Rollout sysID trimming: NO trajectory is explainable at any "
            "candidate params; skipping the fit and pinning the result at "
            "the declared inits.")
        return _init_point_fit_result(all_specs, noise_sigma), [], rms, []

    # Consistency loop. Explainability alone is not enough: a chaotic
    # recording can be ACCIDENTALLY explainable at wrong params (measured:
    # a quiet center-height shove reached best-RMS 0.034 at friction
    # ~1.34 while the clean topple's best sits at friction ~0.1), and
    # pooling two explainable-but-disagreeing recordings drags the fit
    # to a compromise nobody supports. Invariant on exit: every survivor
    # fits the FINAL params nearly as well as its own best params. When
    # violated, the least trustworthy survivor (largest best-achievable
    # RMS) is dropped and the fit reruns — anchoring the answer on the
    # cleanest data rather than the loudest.
    consistency = config.consistency_factor
    physical_names = [s.name for s in physical_specs]
    best = [r for r in rms if r <= threshold]
    hull_candidates: List[Dict[str, float]] = []
    while True:
        result = fit_params_rollout(base_env,
                                    survivors,
                                    physical_specs,
                                    residual_features,
                                    rules=rules,
                                    rule_specs=rule_specs,
                                    latent_init=latent_init,
                                    noise_sigma=noise_sigma,
                                    scaling=scaling,
                                    anchors=anchors,
                                    config=config)
        if len(survivors) <= 1 or consistency <= 0:
            break
        fit_rms = per_trajectory_rms(base_env, survivors,
                                     result.point_estimate, residual_features,
                                     physical_names, rules, latent_init,
                                     scaling)
        violated = [
            i for i in range(len(survivors))
            if fit_rms[i] > consistency * best[i] + _CONSISTENCY_RMS_EPS
        ]
        if not violated:
            break
        drop_idx = max(range(len(survivors)), key=lambda i: best[i])
        # The disagreement is real information about parameter
        # uncertainty even though the drop is right for the point
        # estimate: keep the pre-drop joint fit and the dropped
        # segment's own-best values as hull candidates for the margin
        # sweep.
        hull_candidates.append(
            {n: float(result.point_estimate[n])
             for n in physical_names})
        hull_candidates.append(dict(surv_argmins[drop_idx]))
        logger.info(
            "Rollout sysID consistency: survivors disagree (RMS at joint "
            "fit %s vs own best %s); dropping the least trustworthy "
            "(index %d, best RMS %.4g, own-best params %s) and refitting; "
            "its preferred explanation stays in the uncertainty hull.",
            [f"{r:.4g}" for r in fit_rms], [f"{r:.4g}" for r in best],
            drop_idx, best[drop_idx],
            {k: f"{v:.4g}"
             for k, v in surv_argmins[drop_idx].items()})
        survivors.pop(drop_idx)
        best.pop(drop_idx)
        surv_argmins.pop(drop_idx)
    return result, survivors, rms, hull_candidates
