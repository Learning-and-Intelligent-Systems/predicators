"""Shared Levenberg-Marquardt core and its optional pre-fit wrappers.

``solve_lm`` minimizes a residual vector under a ``ParamSpec`` box in
the fit space; ``lm_prefit`` / ``lm_point_fit_result`` implement the
CFG-gated "LM before (or instead of) MCMC" flow shared by the three fit
entry points in :mod:`fitting` and :mod:`physical_sysid`;
``log_hessian_identifiability`` is the eigenanalysis diagnostic that
reuses the LM Jacobian.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    fit_space_bounds, from_fit_space, to_fit_space
from predicators.settings import CFG

logger = logging.getLogger(__name__)


def lm_prefit(
    lm_fit_fn: Callable[[], Tuple[np.ndarray, Optional[np.ndarray]]],
    sse_fn: Callable[[Dict[str, float]], float],
    names: List[str],
    init_values: np.ndarray,
    noise_sigma: float,
    prior_sigma: np.ndarray,
    label: str,
    warm_start_breakdown_fn: Optional[Callable[[Dict[str, float]],
                                               None]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Optional one-shot LM fit shared by the three fit entry points (per-
    transition, recurrent, and rollout).

    Three independent uses, each behind its own CFG flag (the fit runs
    once if any is set):

      * Hessian diagnostic — eigendecompose J^T J at the MAP
        (``code_sim_learning_log_hessian_identifiability``).
      * Warm start — center the MCMC walkers on theta_map
        (``code_sim_learning_warm_start_with_lm``).
      * Laplace ensemble — info-seeking exploration reuses J at the MAP
        for a calibrated posterior covariance, attached to the
        ``FitResult`` (``agent_explorer_info_seeking``).

    Returns ``(walker_center, lm_theta, lm_jac)``: the MCMC walker
    center (the LM MAP when warm-starting, else ``init_values``), the
    LM MAP itself, and the Jacobian at the MAP (the latter two ``None``
    when LM didn't run or failed). ``lm_fit_fn`` and ``sse_fn`` carry
    the per-transition vs recurrent specifics; the optional
    ``warm_start_breakdown_fn`` lets the per-transition caller add its
    ``log_sse_breakdown`` to the warm-start log.
    """
    walker_center = init_values
    lm_theta: Optional[np.ndarray] = None
    lm_jac: Optional[np.ndarray] = None
    if not (CFG.code_sim_learning_log_hessian_identifiability
            or CFG.code_sim_learning_warm_start_with_lm
            or CFG.agent_explorer_info_seeking):
        return walker_center, lm_theta, lm_jac
    theta_map, jac = lm_fit_fn()
    lm_theta = np.asarray(theta_map, dtype=float)
    if jac is not None and jac.size > 0:
        lm_jac = np.asarray(jac, dtype=float)
        if CFG.code_sim_learning_log_hessian_identifiability:
            log_hessian_identifiability(jac, names, noise_sigma, prior_sigma)
    if CFG.code_sim_learning_warm_start_with_lm:
        walker_center = lm_theta
        logger.info("Warm-starting %s MCMC walkers from LM MAP estimate.",
                    label)
        lm_params = {n: float(lm_theta[i]) for i, n in enumerate(names)}
        lm_sse = sse_fn(lm_params)
        logger.info(
            "After %s LM warm start — SSE: %.6f  log-likelihood: "
            "%.2f", label, lm_sse, -0.5 * lm_sse / (noise_sigma**2))
        if warm_start_breakdown_fn is not None:
            warm_start_breakdown_fn(lm_params)
    return walker_center, lm_theta, lm_jac


def lm_point_fit_result(
    walker_center: np.ndarray,
    lm_theta: Optional[np.ndarray],
    lm_jac: Optional[np.ndarray],
    names: List[str],
    noise_sigma: float,
    prior_sigma: np.ndarray,
    label: str,
    scales: Optional[List[str]] = None,
) -> FitResult:
    """Single-point ``FitResult`` for the ``num_steps == 0`` short-circuit.

    Picks the point estimate the skipped-emcee run reports: the LM MAP
    when one is available and either warm-start or info-seeking asked
    for it (so the Laplace covariance is anchored where the data places
    it, not at init), else the initial parameter values. Carries the
    Laplace bundle through.
    """
    point = walker_center
    if (not CFG.code_sim_learning_warm_start_with_lm
            and CFG.agent_explorer_info_seeking and lm_theta is not None):
        point = lm_theta
        logger.info("Skipping emcee; using %s LM MAP for Laplace ensemble.",
                    label)
    elif CFG.code_sim_learning_warm_start_with_lm and lm_theta is not None:
        logger.info("Skipping emcee; using %s LM warm-start parameters.",
                    label)
    else:
        logger.info("Skipping emcee; using initial parameter values.")
    return FitResult(names,
                     point[None, :],
                     np.zeros(1),
                     jacobian=lm_jac,
                     noise_sigma=noise_sigma,
                     prior_sigma=prior_sigma,
                     scales=scales)


def solve_lm(
    residuals_fn: Callable[[np.ndarray], np.ndarray],
    param_specs: List[ParamSpec],
    max_nfev: int,
    label: str,
    diff_step: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Shared Levenberg-Marquardt core for the per-transition, recurrent, and
    rollout (``physical_sysid``) MAP fits.

    Solves ``min_theta 0.5 * ||residuals_fn(theta)||^2`` with
    ``scipy.optimize.least_squares(method='trf')`` under the
    ``param_specs`` box, and returns ``(theta_map, jacobian_at_optimum)``.
    The Jacobian is ``None`` when the residual vector is empty or LM
    raises. ``label`` only tags the log lines (e.g. ``per-transition`` vs
    ``recurrent``). The single residual-vector seam is what lets the
    recurrent fit reuse this unchanged.

    ``diff_step`` overrides the relative finite-difference step for the
    numerical Jacobian (scipy's default ~sqrt(machine eps)). Residuals
    produced by a physics *simulation* need a much coarser step: a 1e-8
    relative perturbation of e.g. a friction coefficient is below the
    contact solver's sensitivity, the rollout comes back bitwise
    identical, and the Jacobian is identically zero — LM would stall at
    init. Analytic-formula residuals (the per-transition and recurrent
    fits) leave this ``None``.

    The optimizer runs in the FIT space (``z = log(theta)`` for
    log-scale params): ``residuals_fn`` still receives external theta,
    the returned MAP is external, but the returned Jacobian is
    ``dr/dz`` — consistent with the fit-space ``prior_sigma`` every
    downstream consumer (Hessian diagnostic, Laplace ensemble,
    identifiability probe) pairs it with. In fit space the relative
    ``diff_step`` also becomes a *multiplicative* theta perturbation
    for log params, so the finite-difference gradient stays equally
    informative across decades instead of vanishing at the low end.
    """
    from scipy.optimize import \
        least_squares  # pylint: disable=import-outside-toplevel

    names = [s.name for s in param_specs]
    init_ext = np.array([s.init_value for s in param_specs], dtype=float)
    init = to_fit_space(param_specs, init_ext)
    lo, hi = fit_space_bounds(param_specs)
    # Nudge init strictly into the interior so trf doesn't reject it.
    init = np.maximum(init, lo + 1e-9)
    safe_hi = np.where(np.isfinite(hi), hi - 1e-9, np.inf)
    init = np.minimum(init, safe_hi)

    def internal_residuals(z: np.ndarray) -> np.ndarray:
        return residuals_fn(from_fit_space(param_specs, z))

    init_residuals = internal_residuals(init)
    if init_residuals.size == 0:
        logger.warning(
            "No residuals to fit (empty residual_features); "
            "skipping %s LM fit.", label)
        return from_fit_space(param_specs, init), None

    sse_init = float(np.sum(init_residuals**2))

    try:
        result = least_squares(internal_residuals,
                               init,
                               method='trf',
                               bounds=(lo, hi),
                               diff_step=diff_step,
                               max_nfev=max_nfev)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("%s LM fit raised %s; skipping.", label, exc)
        return from_fit_space(param_specs, init), None

    sse_lm = float(2.0 * result.cost)
    x_ext = from_fit_space(param_specs, result.x)
    delta = {
        names[i]: float(x_ext[i] - init_ext[i])
        for i in range(len(names))
    }
    logger.info("%s LM fit: SSE %.4f -> %.4f in %d fn-evals (status=%d, %s).",
                label, sse_init, sse_lm, result.nfev, result.status,
                "converged" if result.success else "max-evals")
    logger.info("%s LM theta_map - init: %s", label,
                {k: f"{v:+.4f}"
                 for k, v in delta.items()})

    jac = np.asarray(result.jac, dtype=float)
    if jac.size == 0:
        return x_ext, None
    return x_ext, jac


def log_hessian_identifiability(
    jacobian: np.ndarray,
    param_names: List[str],
    noise_sigma: float,
    prior_sigma: np.ndarray,
    top_k: int = 3,
) -> None:
    """Eigendecompose the Hessian at the MAP and log identifiability.

    Under a Laplace approximation, the Hessian of the negative
    log-posterior is the inverse posterior covariance. Its eigenvectors
    are *combinations* of parameters (not individual params), and the
    eigenvalues say how tightly the data constrains each combination:

      * Large eigenvalue -> stiff direction: data pins this down.
      * Small eigenvalue -> sloppy direction: data is silent here.

    Sloppy directions point to parameter combinations no optimizer can
    recover from the current data — typically structural rule-pair
    degeneracy or under-excited input trajectories. The Gauss-Newton
    approximation H ~= J^T J / sigma^2 + diag(1/prior_sigma^2) reuses
    the LM Jacobian, so this analysis costs effectively nothing once
    LM has run.
    """
    H_data = jacobian.T @ jacobian / (noise_sigma**2)
    H_prior = np.diag(1.0 / prior_sigma**2)
    H = H_data + H_prior

    eigvals, eigvecs = np.linalg.eigh(H)  # ascending

    cond = float(eigvals[-1] / max(eigvals[0], 1e-30))
    logger.info("Hessian eigenanalysis (cond %.2e, %d params):", cond,
                len(param_names))

    def _format(vec: np.ndarray) -> str:
        order = np.argsort(-np.abs(vec))
        parts = []
        for j in order[:4]:
            if abs(vec[j]) < 0.05:
                break
            parts.append(f"{vec[j]:+.2f} {param_names[j]}")
        return "  ".join(parts) if parts else "(uniform)"

    n = len(eigvals)
    k = min(top_k, n)
    stiff_idx = list(range(n - 1, n - 1 - k, -1))
    stiff_set = set(stiff_idx)
    sloppy_idx = [i for i in range(k) if i not in stiff_set]

    logger.info("  Stiff (well-constrained):")
    for i in stiff_idx:
        logger.info("    lambda = %10.3e :  %s", eigvals[i],
                    _format(eigvecs[:, i]))

    if sloppy_idx:
        logger.info("  Sloppy (under-constrained):")
        for i in sloppy_idx:
            logger.info("    lambda = %10.3e :  %s", eigvals[i],
                        _format(eigvecs[:, i]))
