"""Active-experiment-design primitives for sim-learning exploration.

Pure, dependency-light helpers used to turn the explorer's refinement
from *feasibility-seeking* into *information-seeking*. Three pieces:

* :func:`perturbation_ensemble` — build a small ensemble of plausible
  parameter vectors around a point estimate (the MAP), by perturbing
  each parameter within its ``ParamSpec`` bounds. This is the universal
  fallback that works for both per-transition and recurrent simulators
  (neither a Jacobian nor MCMC samples are required).

* :func:`posterior_subsample_ensemble` / :func:`laplace_ensemble` — the
  *calibrated* upgrades, preferred when the fit supplies the inputs. The
  former subsamples real MCMC posterior draws (``num_mcmc_steps > 0``);
  the latter draws from the Laplace covariance ``(J^T J / sigma^2 +
  diag(1/prior^2))^-1`` at the MAP using the LM Jacobian — per-transition
  or recurrent — when MCMC was skipped (``num_mcmc_steps == 0``). Both
  let the ensemble spread reflect what the data actually leaves
  uncertain — per-parameter, with correlations — rather than uniform
  jitter, so disagreement concentrates on genuinely under-constrained
  parameters instead of merely sensitive ones.

* :func:`mean_bernoulli_entropy` — score how much an ensemble
  *disagrees* about a set of boolean atoms in a given state. High
  disagreement marks an experiment whose outcome is informative about
  the parameters (a state straddling a learned predicate's decision
  boundary), which is exactly what we want the refiner to seek instead
  of a robustly-feasible interior point.

The math here is deliberately framework-agnostic (numpy + plain dicts)
so it can be unit-tested without the planner, the agent SDK, or
PyBullet.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec, is_log

# Smallest representable disagreement; below this two members count as
# agreeing (guards against float dust in the entropy sum).
_ENTROPY_EPS = 1e-9


def _param_width(spec: ParamSpec) -> float:
    """A finite perturbation scale for one parameter, in its FIT space.

    Prefer the declared box width ``hi - lo`` (``log(hi) - log(lo)`` for
    a log-scale parameter, so the jitter is multiplicative and covers
    the box's decades evenly). When a bound is missing or non-finite,
    fall back to the magnitude of the init value (1.0 in log-space for
    log params, i.e. a factor of e) so every parameter gets a usable,
    strictly-positive scale.
    """
    lo, hi = spec.lo, spec.hi
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(
            hi) and hi > lo:
        if is_log(spec):
            return float(np.log(hi) - np.log(lo))
        return float(hi - lo)
    if is_log(spec):
        return 1.0
    mag = abs(float(spec.init_value))
    return mag if mag > 0 else 1.0


def _clip_to_spec(value: float, spec: ParamSpec) -> float:
    """Clip ``value`` into the parameter's ``[lo, hi]`` box."""
    if spec.lo is not None and np.isfinite(spec.lo):
        value = max(value, float(spec.lo))
    if spec.hi is not None and np.isfinite(spec.hi):
        value = min(value, float(spec.hi))
    return float(value)


def perturbation_ensemble(
    point: Dict[str, float],
    specs: Sequence[ParamSpec],
    num_members: int,
    perturb_frac: float,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """Build ``num_members`` parameter vectors around ``point``.

    Member 0 is always ``point`` itself (the ensemble anchor), unperturbed —
    so an ensemble of size 1 reduces to the point estimate and the
    caller's behavior is unchanged. Each remaining member perturbs every
    parameter by ``N(0, (perturb_frac * width)^2)``, clipped back into
    the parameter's box, where ``width`` is the ``ParamSpec`` bound width
    (see :func:`_param_width`).

    Parameters absent from ``point`` are skipped (the caller's point
    estimate is the source of truth for which params exist); parameters
    in ``point`` without a matching spec are carried through unperturbed.
    """
    if num_members < 1:
        raise ValueError("num_members must be >= 1")
    spec_by_name = {s.name: s for s in specs}
    anchor = {k: float(v) for k, v in point.items()}
    members: List[Dict[str, float]] = [dict(anchor)]
    for _ in range(num_members - 1):
        member = dict(anchor)
        for name, value in anchor.items():
            spec = spec_by_name.get(name)
            if spec is None:
                continue
            sigma = perturb_frac * _param_width(spec)
            if is_log(spec) and value > 0:
                perturbed = value * float(np.exp(rng.normal(0.0, sigma)))
            else:
                perturbed = value + rng.normal(0.0, sigma)
            member[name] = _clip_to_spec(perturbed, spec)
        members.append(member)
    return members


def posterior_subsample_ensemble(
    point: Dict[str, float],
    names: Sequence[str],
    samples: np.ndarray,
    num_members: int,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """Build an ensemble by subsampling MCMC posterior ``samples``.

    The calibrated counterpart to :func:`perturbation_ensemble` for the
    ``num_mcmc_steps > 0`` case: ``samples`` (shape ``(num_draws,
    len(names))``) already *is* the posterior, so each non-anchor member
    is a random posterior draw rather than synthetic jitter — the spread
    therefore reflects what the data actually leaves uncertain.

    Member 0 is always ``point`` (the ensemble anchor), so a size-1 ensemble
    reduces to the point estimate. Draws are without replacement when the
    pool is large enough, with replacement otherwise. Keys of ``point``
    not in ``names`` are carried through each member unperturbed.
    """
    if num_members < 1:
        raise ValueError("num_members must be >= 1")
    arr = np.asarray(samples, dtype=float)
    anchor = {k: float(v) for k, v in point.items()}
    members: List[Dict[str, float]] = [dict(anchor)]
    num_draws = arr.shape[0] if arr.ndim == 2 else 0
    if num_members == 1 or num_draws == 0:
        return members
    need = num_members - 1
    idx = rng.choice(num_draws, size=need, replace=num_draws < need)
    for i in idx:
        member = dict(anchor)
        for j, name in enumerate(names):
            member[name] = float(arr[int(i), j])
        members.append(member)
    return members


def laplace_ensemble(
    point: Dict[str, float],
    names: Sequence[str],
    specs: Sequence[ParamSpec],
    jacobian: np.ndarray,
    noise_sigma: float,
    prior_sigma: Union[Sequence[float], np.ndarray],
    num_members: int,
    rng: np.random.Generator,
) -> List[Dict[str, float]]:
    """Build an ensemble from the Laplace posterior at the MAP.

    The calibrated counterpart to :func:`perturbation_ensemble` for the
    ``num_mcmc_steps == 0`` case (the Jacobian comes from the
    per-transition or recurrent LM fit). Under a Laplace approximation
    the negative-log-posterior Hessian at the MAP is

        ``H = J^T J / sigma^2 + diag(1 / prior_sigma^2)``

    (Gauss-Newton, reusing the LM Jacobian ``J``), and the posterior
    covariance is ``Sigma = H^-1``. Each non-anchor member is
    ``MAP + N(0, Sigma)`` clipped to the parameter box. Unlike uniform
    jitter, the per-parameter spread and the *correlations* between
    parameters come from the data: stiff (well-constrained) directions
    barely move, sloppy (under-constrained) ones move a lot — so the
    disagreement signal concentrates where the model is genuinely unsure.

    ``names`` orders the columns of ``J`` (and ``prior_sigma``); ``specs``
    supplies the clipping box. Keys of ``point`` not in ``names`` are
    carried through unperturbed. Falls back to returning just the anchor
    if the covariance cannot be formed (degenerate/empty Jacobian).

    ``J`` and ``prior_sigma`` live in the FIT space (``z = log(theta)``
    for log-scale specs — see ``lm.solve_lm``), so the Laplace
    covariance and the draws are fit-space too; each draw is mapped
    back to external units (exponentiated for log params) before the
    box clip.
    """
    if num_members < 1:
        raise ValueError("num_members must be >= 1")
    name_list = list(names)
    spec_by_name = {s.name: s for s in specs}
    anchor = {k: float(v) for k, v in point.items()}
    members: List[Dict[str, float]] = [dict(anchor)]
    jac = np.asarray(jacobian, dtype=float)
    ndim = len(name_list)
    if num_members == 1 or jac.ndim != 2 or jac.shape[1] != ndim or ndim == 0:
        return members
    sigma = float(noise_sigma) if noise_sigma else 1.0
    prior = np.asarray(prior_sigma, dtype=float)
    if prior.shape != (ndim, ):
        prior = np.ones(ndim)
    # Gauss-Newton negative-log-posterior Hessian, then invert to covariance.
    hess = jac.T @ jac / (sigma**2) + np.diag(1.0 / np.square(prior))
    try:
        cov = np.linalg.inv(hess)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(hess)
    # Symmetrize and clip tiny negative eigenvalues (numerical dust) so the
    # covariance is a valid PSD sampler; sqrt-scale the eigenbasis.
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    scale = eigvecs * np.sqrt(eigvals)  # (ndim, ndim); scale @ z ~ N(0, cov)
    log_mask = [
        n in spec_by_name and is_log(spec_by_name[n])
        and anchor.get(n, 0.0) > 0 for n in name_list
    ]
    mean = np.array([
        np.log(anchor[n]) if log_mask[j] else anchor.get(n, 0.0)
        for j, n in enumerate(name_list)
    ],
                    dtype=float)
    for _ in range(num_members - 1):
        draw = mean + scale @ rng.standard_normal(ndim)
        member = dict(anchor)
        for j, name in enumerate(name_list):
            spec = spec_by_name.get(name)
            value = float(np.exp(draw[j])) if log_mask[j] else float(draw[j])
            member[name] = _clip_to_spec(value, spec) if spec else value
        members.append(member)
    return members


def _bernoulli_entropy(p: float) -> float:
    """Binary entropy in bits; 0 at p in {0, 1}, 1.0 at p = 0.5."""
    if p <= _ENTROPY_EPS or p >= 1.0 - _ENTROPY_EPS:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def mean_bernoulli_entropy(truth_matrix: np.ndarray) -> float:
    """Mean per-atom Bernoulli entropy over an ensemble.

    ``truth_matrix`` is a boolean ``(num_members, num_atoms)`` array:
    entry ``[k, m]`` is whether ensemble member ``k`` believes atom
    ``m`` holds in the candidate state. The score is the mean over atoms
    of the binary entropy of each atom's across-member truth fraction —
    0.0 when every member agrees on every atom (uninformative), up to
    1.0 when members are evenly split (maximally informative). Returns
    0.0 for an empty matrix.
    """
    arr = np.asarray(truth_matrix, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2:
        raise ValueError("truth_matrix must be 2D (members x atoms)")
    fracs = arr.mean(axis=0)  # P(atom holds) across members
    return float(np.mean([_bernoulli_entropy(p) for p in fracs]))
