"""Parameter fitting for the sim-learning approach.

The default path is a Levenberg-Marquardt point fit (MAP under a
Gaussian prior); emcee (affine-invariant ensemble MCMC) is the opt-in
posterior-sampling path behind ``code_sim_learning_num_mcmc_steps``.
The ``ParamSpec``/``FitResult`` types and fit-space transforms live in
:mod:`fit_space`; the LM core lives in :mod:`lm`; this module owns the
objectives (SSE / residual vectors) and the fit entry points.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    fit_space_bounds, from_fit_space, prior_widths, rows_from_fit_space, \
    to_fit_space
from predicators.code_sim_learning.lm import lm_point_fit_result, lm_prefit, \
    solve_lm
from predicators.settings import CFG
from predicators.structs import Action, State

logger = logging.getLogger(__name__)

# Step-level simulator: (State, Action, params_dict) -> {Object: {feat: val}}
StepSimulatorFn = Callable[[State, Action, Dict[str, float]], Dict]

# Per-trajectory list of (base_state, action, next_obs) triples.
# `base_state` is the base sim applied to the previous *real* observation,
# matching the shape used by `compute_sse` but grouped by trajectory so
# the latent block can be threaded across steps within each one.
TrajectoryTriples = List[Tuple[State, Action, State]]


def iter_step_terms(
    state: State,
    updates: Dict,
    next_obs: State,
    residual_features: Dict[str, List[str]],
) -> Any:
    """Yield one ``(obj, feat, predicted, observed, rule_fired)`` term per
    (state object x allowed feature) for a single step.

    The single source of truth behind the SSE / residual-vector /
    breakdown computations, in deterministic object x feature order: a
    fixed vector length and position across theta perturbations is a
    hard requirement of the finite-difference Jacobians LM builds, even
    when a hard gate flips which rule fires. A feature the simulator did
    not update is predicted as "no change" (the pre-step value).
    Predicted features for objects absent from ``state`` are ignored -
    they have no observation to compare against.
    """
    for obj in state:
        type_name = obj.type.name
        for feat_name in residual_features.get(type_name, []):
            fired = obj in updates and feat_name in updates[obj]
            if fired:
                raw = updates[obj][feat_name]
                pred = raw.item() if hasattr(raw, 'item') else float(raw)
            else:
                pred = float(state.get(obj, feat_name))
            obs = float(next_obs.get(obj, feat_name))
            yield obj, feat_name, float(pred), obs, fired


def compute_sse(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    params: Dict[str, float],
    residual_features: Dict[str, List[str]],
) -> float:
    """Sum of squared errors between predicted and observed residual features.

    Returns the total (un-normalized) SSE so that the Gaussian
    log-likelihood ``-0.5 * SSE / noise_sigma**2`` is the correct
    iid-observation form. Dividing by count would silently rescale the
    per-observation noise by sqrt(count), making the chain insensitive
    to parameter changes.
    """
    total_se = 0.0
    for s_t, action, s_next_obs in transitions:
        updates = simulator_fn(s_t, action, params)
        for _obj, _feat, pred, obs, _fired in iter_step_terms(
                s_t, updates, s_next_obs, residual_features):
            total_se += (pred - obs)**2
    return total_se


def compute_sse_recurrent(
    rules: List,
    trajectories: List[TrajectoryTriples],
    params: Dict[str, float],
    latent_init: Any,
    residual_features: Dict[str, List[str]],
) -> float:
    """SSE on observables, with the ``latent`` block threaded per trajectory.

    Counterpart to :func:`compute_sse` for the recurrent
    (partially-observable) approach. Each input trajectory is a list
    of ``(base_state, action, next_obs)`` triples — the same shape
    individual transitions take in :func:`compute_sse`, but grouped
    so the latent block can carry across steps within a trajectory.

    For each trajectory:

    * Build an initial ``latent`` dict from ``latent_init`` (constants
      and any ``ParamSpec``-valued entries resolve from ``params``).
    * Roll forward step-by-step: call
      :func:`apply_rules_with_latent` with the running latent and the
      history prefix; merge the predicted observable feature updates;
      compare to the real next-step observation.
    * The "filter" step is implicit — ``base_state`` is the base sim
      applied to the *real* previous observation, so we re-ground
      observables each step automatically. Only ``latent`` propagates
      across step boundaries within a trajectory.

    Returns the total un-normalised SSE so the Gaussian log-likelihood
    ``-0.5 * SSE / noise_sigma**2`` is the correct iid form.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import apply_rules_with_latent, \
        init_latent

    # pylint: enable=import-outside-toplevel

    total_se = 0.0
    for traj in trajectories:
        latent: Dict[str, Any] = init_latent(latent_init, params)
        history: List[Tuple[State, Optional[Action]]] = []
        for state_base, action, state_obs in traj:
            history.append((state_base, action))
            updates = apply_rules_with_latent(state_base, latent, history,
                                              rules, params)

            for _obj, _feat, pred, obs, _fired in iter_step_terms(
                    state_base, updates, state_obs, residual_features):
                total_se += (pred - obs)**2

    return total_se


def fit_params_recurrent(
    rules: List,
    trajectories: List[TrajectoryTriples],
    param_specs: List[ParamSpec],
    latent_init: Any,
    residual_features: Dict[str, List[str]],
    num_walkers: int = 32,
    num_steps: Optional[int] = None,
    burn_in: int = 200,
    noise_sigma: float = 0.05,
    prior_sigma_scale: float = 1.0,
) -> FitResult:
    """Fit recurrent-sim parameters via emcee MCMC.

    Mirror of :func:`fit_params` for the recurrent (latent-threaded)
    rollout used by the partial-observability approach. Differences
    from :func:`fit_params`:

    * Likelihood = :func:`compute_sse_recurrent` (per-trajectory
      rollout with latent carry) instead of per-transition
      :func:`compute_sse`.
    * Uses a recurrent LM warm-start / Hessian diagnostic / Laplace
      bundle (:func:`fit_map_lm_recurrent`, built on the rollout residual
      vector :func:`compute_residuals_recurrent`) under the same CFG flags
      as the FO path, in place of the per-transition :func:`fit_map_lm`.
      The Jacobian at the MAP is attached to the returned ``FitResult`` so
      callers can build the Laplace ensemble (see
      ``active_experiment.laplace_ensemble``).
    """
    names = [s.name for s in param_specs]
    scales = [getattr(s, "scale", "linear") for s in param_specs]
    init_values = np.array([s.init_value for s in param_specs])
    if num_steps is None:
        num_steps = CFG.code_sim_learning_num_mcmc_steps
    if num_steps < 0:
        raise ValueError("code_sim_learning_num_mcmc_steps must be "
                         "non-negative.")
    lo, hi = fit_space_bounds(param_specs)
    init_int = to_fit_space(param_specs, init_values)
    prior_sigma = prior_widths(param_specs, prior_sigma_scale)

    # Optional one-shot recurrent LM fit (see lm_prefit for its three
    # uses). Each residual eval here is a full set of per-trajectory
    # rollouts, so it is only paid when one of the gating flags is set.
    walker_center, lm_theta, lm_jac = lm_prefit(
        lambda: fit_map_lm_recurrent(rules, trajectories, param_specs,
                                     latent_init, residual_features),
        lambda p: compute_sse_recurrent(rules, trajectories, p, latent_init,
                                        residual_features), names, init_values,
        noise_sigma, prior_sigma, "recurrent")

    if num_steps == 0:
        return lm_point_fit_result(walker_center,
                                   lm_theta,
                                   lm_jac,
                                   names,
                                   noise_sigma,
                                   prior_sigma,
                                   "recurrent",
                                   scales=scales)

    logger.info("Running emcee (recurrent): %d walkers, %d steps, %d burn-in.",
                max(num_walkers, 2 * len(param_specs) + 2), num_steps,
                min(burn_in, max(num_steps - 1, 0)))
    samples, log_probs = run_emcee_posterior(
        param_specs,
        lambda p: compute_sse_recurrent(rules, trajectories, p, latent_init,
                                        residual_features),
        walker_center,
        init_int,
        prior_sigma,
        lo,
        hi,
        noise_sigma,
        num_walkers,
        num_steps,
        burn_in,
        label="recurrent")
    result = FitResult(names=names,
                       samples=samples,
                       log_probs=log_probs,
                       jacobian=lm_jac,
                       noise_sigma=noise_sigma,
                       prior_sigma=prior_sigma,
                       scales=scales)
    logger.info("emcee (recurrent) done. Posterior mean: %s",
                {k: f"{v:.4f}"
                 for k, v in result.point_estimate.items()})
    return result


def compute_residuals(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    params: Dict[str, float],
    residual_features: Dict[str, List[str]],
) -> np.ndarray:
    """Per-feature residuals (predicted - observed) as a flat vector.

    Used by Levenberg-Marquardt, which needs the residual *vector*
    rather than scalar SSE so it can build J = dr/dtheta. Iteration
    order is deterministic so the same theta produces the same vector
    across calls (required for finite-difference Jacobians).
    """
    residuals: List[float] = []
    for s_t, action, s_next_obs in transitions:
        updates = simulator_fn(s_t, action, params)
        residuals.extend(pred - obs
                         for _obj, _feat, pred, obs, _fired in iter_step_terms(
                             s_t, updates, s_next_obs, residual_features))
    return np.asarray(residuals, dtype=float)


def compute_residuals_recurrent(
    rules: List,
    trajectories: List[TrajectoryTriples],
    params: Dict[str, float],
    latent_init: Any,
    residual_features: Dict[str, List[str]],
) -> np.ndarray:
    """Per-feature residuals (predicted - observed) for the recurrent rollout.

    Vector counterpart to :func:`compute_sse_recurrent`; both draw their
    terms from :func:`iter_step_terms`, so the flat vector keeps a fixed
    length and position across theta perturbations even when a hard gate
    flips which rule fires -- required for the finite-difference
    Jacobian LM builds. By construction
    ``sum(compute_residuals_recurrent(...)**2)`` equals
    ``compute_sse_recurrent(...)``, so minimizing ``0.5 * ||r||^2`` with LM
    targets the same MAP the recurrent MCMC samples around, and yields the
    Jacobian for the Hessian diagnostic and the Laplace ensemble.

    Each call is a full set of per-trajectory rollouts, so an LM
    finite-difference Jacobian costs ``O(num_params)`` of these.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import apply_rules_with_latent, \
        init_latent

    # pylint: enable=import-outside-toplevel

    residuals: List[float] = []
    for traj in trajectories:
        latent: Dict[str, Any] = init_latent(latent_init, params)
        history: List[Tuple[State, Optional[Action]]] = []
        for state_base, action, state_obs in traj:
            history.append((state_base, action))
            updates = apply_rules_with_latent(state_base, latent, history,
                                              rules, params)
            residuals.extend(
                pred - obs
                for _obj, _feat, pred, obs, _fired in iter_step_terms(
                    state_base, updates, state_obs, residual_features))
    return np.asarray(residuals, dtype=float)


def log_sse_breakdown(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    params: Dict[str, float],
    residual_features: Dict[str, List[str]],
    label: str = "",
) -> None:
    """Log per-(type, feature) SSE so we can see which features dominate.

    Splits each feature's residual into two buckets:
      * ``pred``    — transitions where the rule produced an update
                      (residual is sim's prediction error)
      * ``no_pred`` — transitions where no rule fired
                      (residual is whatever the env changed on its own;
                      large values here mean the model is missing a
                      process for this feature)
    """
    bucket: Dict[Tuple[str, str], Dict[str, float]] = {}

    def _slot(key: Tuple[str, str]) -> Dict[str, float]:
        if key not in bucket:
            bucket[key] = {
                "sse_pred": 0.0,
                "n_pred": 0,
                "sse_no_pred": 0.0,
                "n_no_pred": 0,
                "max_abs_err": 0.0,
            }
        return bucket[key]

    for s_t, action, s_next_obs in transitions:
        updates = simulator_fn(s_t, action, params)
        for obj, feat_name, pred, obs, fired in iter_step_terms(
                s_t, updates, s_next_obs, residual_features):
            err = pred - obs
            slot = _slot((obj.type.name, feat_name))
            if fired:
                slot["sse_pred"] += err * err
                slot["n_pred"] += 1
            else:
                slot["sse_no_pred"] += err * err
                slot["n_no_pred"] += 1
            slot["max_abs_err"] = max(slot["max_abs_err"], abs(err))

    if not bucket:
        return

    total = sum(s["sse_pred"] + s["sse_no_pred"] for s in bucket.values())
    header = f"SSE breakdown{(' — ' + label) if label else ''} " \
             f"(total {total:.4f}):"
    logger.info(header)
    logger.info("  %-22s  %10s  %6s  %10s  %6s  %10s", "type.feature",
                "sse_pred", "n_pred", "sse_no_pred", "n_nop", "max|err|")
    rows = sorted(
        bucket.items(),
        key=lambda kv: -(kv[1]["sse_pred"] + kv[1]["sse_no_pred"]),
    )
    for (type_name, feat_name), s in rows:
        logger.info(
            "  %-22s  %10.4f  %6d  %10.4f  %6d  %10.4f",
            f"{type_name}.{feat_name}",
            s["sse_pred"],
            int(s["n_pred"]),
            s["sse_no_pred"],
            int(s["n_no_pred"]),
            s["max_abs_err"],
        )


def run_emcee_posterior(
    param_specs: List[ParamSpec],
    sse_fn: Callable[[Dict[str, float]], float],
    walker_center: np.ndarray,
    prior_center_int: np.ndarray,
    prior_sigma: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    noise_sigma: float,
    num_walkers: int,
    num_steps: int,
    burn_in: int,
    label: str,
    report_interval: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared emcee run behind the three fit entry points.

    Returns ``(samples_external, log_probs)`` with burn-in discarded and
    chains flattened. All coordinates are in the FIT space internally
    (``lo``/``hi``/``prior_center_int``/``prior_sigma``); the returned
    samples are mapped back to EXTERNAL units.

    RNG discipline: the walker init is the single ``np.random`` draw
    (then emcee's own internal draws); the call sequence is preserved
    exactly from the three formerly-duplicated blocks so fixed-seed
    chains are bit-identical across the consolidation.
    """
    import emcee  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    names = [s.name for s in param_specs]
    ndim = len(param_specs)
    num_walkers = max(num_walkers, 2 * ndim + 2)
    burn_in = min(burn_in, max(num_steps - 1, 0))

    def log_posterior(theta: np.ndarray) -> float:
        # theta lives in the FIT space (log for log-scale params).
        # Reject samples outside the per-parameter [lo, hi] box.
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        ext = from_fit_space(param_specs, theta)
        params = {n: float(ext[i]) for i, n in enumerate(names)}
        # Broad Gaussian prior centered on the prior center.
        log_prior = -0.5 * np.sum(
            ((theta - prior_center_int) / prior_sigma)**2)
        sse = sse_fn(params)
        return float(log_prior - 0.5 * sse / (noise_sigma**2))

    # Initialize walkers across the prior support (sigma = half the prior
    # width). A tight ball around init traps the chain on flat plateaus
    # of the likelihood (e.g., when threshold-based rules don't fire),
    # because emcee stretch moves scale with the swarm's spread.
    p0 = to_fit_space(param_specs, walker_center) + \
        0.5 * prior_sigma * np.random.randn(num_walkers, ndim)
    p0 = np.clip(p0, lo, hi)

    sampler = emcee.EnsembleSampler(num_walkers, ndim, log_posterior)

    # Run with periodic progress reports (flushed so long fits stay
    # observable in the experiment logs).
    for i, _result in enumerate(sampler.sample(p0, iterations=num_steps),
                                start=1):
        if i % report_interval == 0 or i == num_steps:
            best_lp = sampler.get_log_prob()[:i].max()
            logger.info("  %s emcee step %d/%d  (best log-prob: %.2f)", label,
                        i, num_steps, best_lp)
            for h in logger.handlers + logging.getLogger().handlers:
                h.flush()

    # Discard burn-in, flatten chains (back to external units).
    samples = rows_from_fit_space(
        param_specs, sampler.get_chain(discard=burn_in, flat=True))
    log_probs = sampler.get_log_prob(discard=burn_in, flat=True)
    return samples, log_probs


def fit_map_lm(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    param_specs: List[ParamSpec],
    residual_features: Dict[str, List[str]],
    max_nfev: int = 200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Find a MAP estimate via Levenberg-Marquardt (trust-region reflective).

    Returns ``(theta_map, jacobian_at_optimum)``. Jacobian is ``None``
    only if the residual vector is empty or LM raises; in those cases
    callers should treat the diagnostic as unavailable.

    How LM finds the MAP here:
      * ``compute_residuals`` returns r(theta) = (s_{t+1}_obs - sim(s_t, a;
        theta)) flattened over transitions and the features named in
        ``residual_features``. Minimizing 0.5 * ||r||^2 is exactly MLE
        under iid Gaussian observation noise; with the broad Gaussian
        prior used elsewhere in this module being effectively flat near
        init, the least-squares minimizer coincides with the MAP.
      * ``scipy.optimize.least_squares(method='trf')`` runs a
        Levenberg-Marquardt step inside a trust region with box
        constraints (``lo``/``hi`` from ``param_specs``). At each step
        it numerically estimates the Jacobian J = dr/dtheta, solves the
        damped normal equations (J^T J + lambda I) dtheta = -J^T r, and
        adapts lambda based on whether the step reduces SSE.
      * On exit, ``result.x`` is theta_map and ``result.jac`` is J at
        the optimum. J^T J / sigma^2 is the Gauss-Newton approximation
        to the negative log-likelihood Hessian — the input
        ``log_hessian_identifiability`` eigendecomposes to flag flat
        directions.

    Three uses of the result:
      * Hessian identifiability diagnostic — eigendecompose J^T J.
      * MCMC warm start — center emcee walkers on theta_map (and short-
        circuit to it directly when ``num_mcmc_steps == 0``).
      * Laplace ensemble — reuse J at the MAP for a calibrated posterior
        covariance (see ``active_experiment.laplace_ensemble``).
    """
    names = [s.name for s in param_specs]

    def residuals_fn(theta: np.ndarray) -> np.ndarray:
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        return compute_residuals(simulator_fn, transitions, params,
                                 residual_features)

    return solve_lm(residuals_fn, param_specs, max_nfev, "per-transition")


def fit_map_lm_recurrent(
    rules: List,
    trajectories: List[TrajectoryTriples],
    param_specs: List[ParamSpec],
    latent_init: Any,
    residual_features: Dict[str, List[str]],
    max_nfev: int = 200,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Levenberg-Marquardt MAP fit for the recurrent (latent-threaded) sim.

    Recurrent counterpart to :func:`fit_map_lm`, sharing the same
    :func:`solve_lm` core; only the residual vector differs — here it
    comes from :func:`compute_residuals_recurrent` (a full latent rollout
    per evaluation) rather than the per-transition residuals. Returns
    ``(theta_map, jacobian-at-optimum)`` under the ParamSpec ``[lo, hi]``
    box; the Jacobian is ``None`` when residuals are empty or LM raises.

    Same smoothness caveat as the FO path: the finite-difference Jacobian
    is only informative where the likelihood is smooth. A hard-gated
    parameter with no boundary-crossing data has a near-zero column in J,
    so LM leaves it at init -- but the Hessian diagnostic then surfaces it
    as a flat (unidentifiable) direction rather than a confident wrong
    value.

    Cost note: every residual evaluation is a full set of per-trajectory
    rollouts, so the finite-difference Jacobian costs ``O(num_params)``
    rollouts per LM iteration; for large param sets prefer MCMC. And
    because latent threading correlates residuals across steps, ``J^T J``
    ignores that coupling, making the recurrent Laplace covariance a
    slightly looser approximation than the per-transition one (MCMC at
    ``num_mcmc_steps > 0`` remains the gold path).
    """
    names = [s.name for s in param_specs]

    def residuals_fn(theta: np.ndarray) -> np.ndarray:
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        return compute_residuals_recurrent(rules, trajectories, params,
                                           latent_init, residual_features)

    return solve_lm(residuals_fn, param_specs, max_nfev, "recurrent")


def fit_params(
    simulator_fn: StepSimulatorFn,
    transitions: List[Tuple[State, Action, State]],
    param_specs: List[ParamSpec],
    residual_features: Dict[str, List[str]],
    num_walkers: int = 32,
    num_steps: Optional[int] = None,
    burn_in: int = 200,
    noise_sigma: float = 0.05,
    prior_sigma_scale: float = 1.0,
) -> FitResult:
    """Fit simulator parameters: LM point fit, optional emcee posterior.

    With ``code_sim_learning_num_mcmc_steps == 0`` (the experiment
    default) this returns the Levenberg-Marquardt MAP directly; with
    MCMC steps it runs emcee (gradient-free, so it tolerates
    non-smooth simulators), optionally warm-started from the LM fit.

    Args:
        simulator_fn: Simulator(state, action, params_dict) -> updates.
            Should run the base sim internally if needed.
        transitions: List of (s_t, action, s_{t+1}_obs) triples.
        param_specs: Parameter specifications (name, init_value).
        residual_features: {type_name: [feat_names]} to fit.
        num_walkers: Number of ensemble walkers (>= 2*ndim).
        num_steps: Total MCMC steps per walker. If None, defaults to
            CFG.code_sim_learning_num_mcmc_steps. If 0, skip training and
            use initial parameter values directly.
        burn_in: Steps to discard as burn-in.
        noise_sigma: Observation noise std dev for likelihood.
        prior_sigma_scale: Prior width as multiple of init_value.

    Returns:
        FitResult with posterior samples and log-probabilities.
    """
    names = [s.name for s in param_specs]
    scales = [getattr(s, "scale", "linear") for s in param_specs]
    init_values = np.array([s.init_value for s in param_specs])
    if num_steps is None:
        num_steps = CFG.code_sim_learning_num_mcmc_steps
    if num_steps < 0:
        raise ValueError("code_sim_learning_num_mcmc_steps must be "
                         "non-negative.")
    lo, hi = fit_space_bounds(param_specs)
    init_int = to_fit_space(param_specs, init_values)
    prior_sigma = prior_widths(param_specs, prior_sigma_scale)

    # Optional one-shot LM fit (see lm_prefit for its three uses).
    walker_center, lm_theta, lm_jac = lm_prefit(
        lambda: fit_map_lm(simulator_fn, transitions, param_specs,
                           residual_features),
        lambda p: compute_sse(simulator_fn, transitions, p, residual_features),
        names,
        init_values,
        noise_sigma,
        prior_sigma,
        "per-transition",
        warm_start_breakdown_fn=lambda p: log_sse_breakdown(simulator_fn,
                                                            transitions,
                                                            p,
                                                            residual_features,
                                                            label=
                                                            "lm-warm-start"))

    if num_steps == 0:
        return lm_point_fit_result(walker_center,
                                   lm_theta,
                                   lm_jac,
                                   names,
                                   noise_sigma,
                                   prior_sigma,
                                   "per-transition",
                                   scales=scales)

    logger.info("Running emcee: %d walkers, %d steps, %d burn-in.",
                max(num_walkers, 2 * len(param_specs) + 2), num_steps,
                min(burn_in, max(num_steps - 1, 0)))
    samples, log_probs = run_emcee_posterior(
        param_specs,
        lambda p: compute_sse(simulator_fn, transitions, p, residual_features),
        walker_center,
        init_int,
        prior_sigma,
        lo,
        hi,
        noise_sigma,
        num_walkers,
        num_steps,
        burn_in,
        label="per-transition")

    result = FitResult(names=names,
                       samples=samples,
                       log_probs=log_probs,
                       jacobian=lm_jac,
                       noise_sigma=noise_sigma,
                       prior_sigma=prior_sigma,
                       scales=scales)

    logger.info("emcee done. Posterior mean: %s",
                {k: f"{v:.4f}"
                 for k, v in result.point_estimate.items()})

    return result


# Observation-noise sigma shared by the rule-fit wrappers below and the
# approach's likelihood logging, so SSE -> log-likelihood conversions
# agree everywhere.
FIT_NOISE_SIGMA = 0.05


def log_param_changes(init_params: Dict[str, float],
                      fitted_params: Dict[str, float]) -> None:
    """Log each parameter's init -> fitted move (absolute and %)."""
    for name in sorted(fitted_params):
        init_val = init_params[name]
        fit_val = fitted_params[name]
        delta = fit_val - init_val
        pct = (delta / init_val * 100) if init_val != 0 else float("nan")
        logger.info("  %-30s  %.4f -> %.4f  (Δ=%.4f, %+.1f%%)", name, init_val,
                    fit_val, delta, pct)


def fit_rule_parameters(
    rules: List,
    specs: List[ParamSpec],
    base_pred_triples: List[Tuple[State, Action, State]],
    residual_features: Dict[str, List[str]],
    num_steps: Optional[int] = None,
) -> Tuple[FitResult, float]:
    """Fit parameters for synthesized residual rules (teacher-forced).

    ``base_pred_triples`` must already have the base step applied;
    precomputing avoids re-running it inside the MCMC inner loop.

    ``num_steps`` overrides the global MCMC budget for this fit
    (``None`` falls back to ``CFG.code_sim_learning_num_mcmc_steps``).

    Returns the full :class:`FitResult` (so callers can reach the
    posterior ``samples`` / Laplace ``jacobian`` for ensemble
    construction) alongside the post-fit SSE. Shared source of truth
    for the approach's engine fit and the synthesis tools' scoring, so
    the two cannot drift.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.utils import apply_rules

    def sim_fn(state: State, _action: Action, params: Dict[str,
                                                           float]) -> Dict:
        return apply_rules(state, rules, params)

    init_params = {s.name: s.init_value for s in specs}
    pre_sse = compute_sse(sim_fn, base_pred_triples, init_params,
                          residual_features)
    pre_ll = -0.5 * pre_sse / (FIT_NOISE_SIGMA**2)
    logger.info("Before fitting - SSE: %.6f  log-likelihood: %.2f", pre_sse,
                pre_ll)
    log_sse_breakdown(sim_fn,
                      base_pred_triples,
                      init_params,
                      residual_features,
                      label="before")

    result = fit_params(
        simulator_fn=sim_fn,
        transitions=base_pred_triples,
        param_specs=specs,
        residual_features=residual_features,
        num_steps=num_steps,
    )

    fitted_params = result.point_estimate
    post_sse = compute_sse(sim_fn, base_pred_triples, fitted_params,
                           residual_features)
    post_ll = -0.5 * post_sse / (FIT_NOISE_SIGMA**2)
    logger.info("After fitting  - SSE: %.6f  log-likelihood: %.2f", post_sse,
                post_ll)
    log_sse_breakdown(sim_fn,
                      base_pred_triples,
                      fitted_params,
                      residual_features,
                      label="after")
    log_param_changes(init_params, fitted_params)
    return result, post_sse


def fit_rule_parameters_latent(
    rules: List,
    specs: List[ParamSpec],
    groups: List[TrajectoryTriples],
    latent_init: Any,
    residual_features: Dict[str, List[str]],
    num_steps: Optional[int] = None,
) -> Tuple[FitResult, float]:
    """Recurrent MCMC fit over pre-grouped trajectories.

    Shared source of truth for the recurrent (latent-threaded) fit:
    the approach calls it with groups derived from its trajectory cache
    and latent init; the synthesis tools call it with groups they
    regroup and ``LATENT_INIT`` read fresh from ``simulator.py``. Both
    therefore score latent rules identically, with no tool/engine drift
    in the rule call convention.

    ``num_steps`` overrides the global MCMC budget (``None`` falls
    back to ``CFG.code_sim_learning_num_mcmc_steps``). The tools
    never pass it, so repeated tool calls stay at the fast global
    setting while the post-synthesis fit can run real MCMC.
    """
    init_params = {s.name: s.init_value for s in specs}
    pre_sse = compute_sse_recurrent(rules, groups, init_params, latent_init,
                                    residual_features)
    logger.info("Recurrent fit - pre-SSE: %.6f", pre_sse)

    result = fit_params_recurrent(
        rules=rules,
        trajectories=groups,
        param_specs=specs,
        latent_init=latent_init,
        residual_features=residual_features,
        num_steps=num_steps,
    )
    fitted_params = result.point_estimate
    post_sse = compute_sse_recurrent(rules, groups, fitted_params, latent_init,
                                     residual_features)
    logger.info("Recurrent fit - post-SSE: %.6f", post_sse)
    log_param_changes(init_params, fitted_params)
    return result, post_sse
