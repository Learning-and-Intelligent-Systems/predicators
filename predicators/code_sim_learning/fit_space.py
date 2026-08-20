"""Fit-space primitives shared by every parameter-fitting path.

``ParamSpec`` declares a learnable parameter (bounds and linear/log
scale); ``FitResult`` is the common result bundle. The transform
helpers map between EXTERNAL units (linear, simulator-facing) and the
FIT space the optimizers run in (``z = log(theta)`` for log-scale
params). This module is a dependency-free leaf: it may import nothing
from the rest of the package, so ``fitting``, ``physical_sysid``,
``active_experiment``, ``utils``, and the approaches can all share it
without import cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Floor before taking logs of nonnegative values, to keep log(0) finite.
LOG_FLOOR = 1e-300


@dataclass
class ParamSpec:
    """Specification for a single learnable parameter.

    ``scale`` selects the fitting parameterization. ``"linear"`` (the
    default) fits theta directly. ``"log"`` fits ``z = log(theta)`` —
    the right choice for positive scale-like parameters (friction,
    mass) whose behavioral effect is multiplicative: the grid sweep
    becomes geometric (equal resolution per decade instead of piling
    every point at the high end), LM finite-difference steps become
    relative, and the Gaussian prior in z is a log-normal that treats
    "4x smaller" and "4x larger" as equally plausible. Everything
    simulator- and caller-facing stays in linear units; only the
    optimizer's internal coordinates change.
    """

    name: str
    init_value: float
    lo: Optional[float] = None
    hi: Optional[float] = None
    scale: str = "linear"

    def __post_init__(self) -> None:
        if self.scale not in ("linear", "log"):
            raise ValueError(f"ParamSpec scale must be 'linear' or 'log', "
                             f"got {self.scale!r} for {self.name!r}.")
        if self.scale == "log":
            if self.init_value <= 0:
                raise ValueError(f"log-scale param {self.name!r} needs a "
                                 f"positive init_value, got "
                                 f"{self.init_value}.")
            if self.lo is not None and self.lo <= 0:
                raise ValueError(f"log-scale param {self.name!r} needs a "
                                 f"positive lo bound, got {self.lo}.")


@dataclass
class FitResult:
    """Result of parameter fitting.

    The optional ``jacobian``/``noise_sigma``/``prior_sigma`` fields are a
    Laplace bundle, attached by :func:`fit_params`,
    :func:`fit_params_recurrent`, and
    :func:`physical_sysid.fit_params_rollout` whenever their
    Levenberg-Marquardt fit ran (info-seeking exploration or the
    Hessian/warm-start flags). They
    let a caller build a calibrated posterior covariance
    ``(J^T J / sigma^2 + diag(1/prior^2))^-1`` around the MAP without
    re-deriving it. They stay ``None`` when LM was skipped or failed —
    e.g. MCMC-only runs, where ``samples`` already carries the posterior.

    Space conventions: ``samples`` (and therefore ``point_estimate``)
    are always in EXTERNAL (linear, simulator-facing) units, while
    ``jacobian`` and ``prior_sigma`` live in the FIT space — for a
    log-scale parameter that means d(residual)/d(log theta) and a
    log-space prior width. ``scales`` records each column's ``ParamSpec
    .scale`` so consumers (identifiability probe, Laplace ensemble) can
    map between the two; ``None`` means all-linear (legacy results).
    """

    names: List[str]
    samples: np.ndarray  # (num_samples, num_params) — EXTERNAL units
    log_probs: np.ndarray  # (num_samples,)
    jacobian: Optional[np.ndarray] = None  # (num_residuals, num_params) at MAP
    noise_sigma: Optional[float] = None  # observation-noise sigma used in fit
    prior_sigma: Optional[
        np.ndarray] = None  # (num_params,) Gaussian-prior std, FIT space
    scales: Optional[List[str]] = None  # per-param ParamSpec.scale
    # Pre-fit sensitivity screen (physical_sysid): per-param
    # {"sse_span": ..., "noise_floor": ..., "sensitive": bool}. A param
    # whose grid-sweep SSE span stays within the same-theta noise floor
    # does not affect the rollouts at all on this data; its fitted value
    # is noise and must not be applied. None = screen not run.
    sensitivity: Optional[Dict[str, Dict[str, Any]]] = None
    # Post-fit anchor-ablation verdicts (physical_sysid): per reverted
    # param {"anchor": ..., "sse_map": ..., "sse_pinned": ..., "tol":
    # ...}. A listed param's MAP move was compensatory - a refit with it
    # pinned at its env-registry anchor is data-equivalent - so its
    # ``point_estimate`` entry IS the anchor and the identifiability
    # verdict reports it as anchored rather than identified. None =
    # ablation not run or nothing reverted.
    anchor_ablation: Optional[Dict[str, Dict[str, float]]] = None

    @property
    def point_estimate(self) -> Dict[str, float]:
        """MAP (sample with highest log-probability)."""
        best_idx = int(np.argmax(self.log_probs))
        return {
            n: float(self.samples[best_idx, i])
            for i, n in enumerate(self.names)
        }


def param_bounds(
        param_specs: List[ParamSpec]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-parameter (lo, hi) box from the ParamSpecs, in EXTERNAL units.

    An unspecified bound defaults to a small positive floor (lo) or +inf
    (hi). A parameter that declares a negative ``lo`` -- e.g. a signed
    local offset whose true value is negative -- is therefore fit over
    its real range, while a parameter that declares no bounds keeps the
    historical positivity assumption. Shared by the LM and emcee paths
    so they constrain to the same box.
    """
    lo = np.array([s.lo if s.lo is not None else 1e-6 for s in param_specs])
    hi = np.array([s.hi if s.hi is not None else np.inf for s in param_specs])
    return lo, hi


def is_log(spec: ParamSpec) -> bool:
    """Whether ``spec`` fits in log-space (tolerates legacy instances)."""
    return getattr(spec, "scale", "linear") == "log"


def to_fit_space(param_specs: List[ParamSpec], values: Any) -> np.ndarray:
    """Map external (linear) parameter values into the fit space."""
    return np.array([
        np.log(v) if is_log(s) else float(v)
        for s, v in zip(param_specs, values)
    ],
                    dtype=float)


def from_fit_space(param_specs: List[ParamSpec], values: Any) -> np.ndarray:
    """Map fit-space parameter values back to external (linear) units."""
    return np.array([
        np.exp(v) if is_log(s) else float(v)
        for s, v in zip(param_specs, values)
    ],
                    dtype=float)


def rows_from_fit_space(param_specs: List[ParamSpec],
                        arr: np.ndarray) -> np.ndarray:
    """Map a (num_rows, num_params) fit-space array to external units."""
    out = np.array(arr, dtype=float, copy=True)
    for j, spec in enumerate(param_specs):
        if is_log(spec):
            out[:, j] = np.exp(out[:, j])
    return out


def fit_space_bounds(
        param_specs: List[ParamSpec]) -> Tuple[np.ndarray, np.ndarray]:
    """The `param_bounds` box mapped into the fit space."""
    lo, hi = param_bounds(param_specs)
    lo_int = np.array(
        [np.log(l) if is_log(s) else l for s, l in zip(param_specs, lo)],
        dtype=float)
    hi_int = np.array(
        [np.log(h) if is_log(s) else h for s, h in zip(param_specs, hi)],
        dtype=float)
    return lo_int, hi_int


def prior_widths(param_specs: List[ParamSpec], scale: float) -> np.ndarray:
    """Positive Gaussian-prior width (sigma) per parameter, in FIT space.

    Linear parameters scale by ``|init|`` so a signed (negative-init)
    parameter gets a positive width, falling back to half the (finite)
    bound range when ``init`` is ~0 so a zero-centred parameter still
    gets a finite prior and walker spread instead of a degenerate zero-
    width one. Log parameters get a constant width of ``scale`` in log-
    space — a log-normal prior whose one-sigma band spans the same
    multiplicative factor (e.g. 0.75 => x/2.1 .. x2.1) at every init,
    matching how scale-like physics parameters actually behave.
    """
    lo, hi = param_bounds(param_specs)
    init_values = np.array([s.init_value for s in param_specs], dtype=float)
    sigma = np.abs(init_values) * scale
    finite = np.isfinite(lo) & np.isfinite(hi)
    fallback = np.where(finite, 0.5 * (hi - lo), 1.0)
    linear_sigma = np.where(sigma > 1e-9, sigma, fallback)
    log_mask = np.array([is_log(s) for s in param_specs], dtype=bool)
    return np.where(log_mask, float(scale), linear_sigma)


def scalar_to_fit_space(spec: ParamSpec, value: float) -> float:
    """``value`` in ``spec``'s fit space (log for log-scale params)."""
    if is_log(spec):
        return float(np.log(max(value, LOG_FLOOR)))
    return float(value)


def scalar_from_fit_space(spec: ParamSpec, z: float) -> float:
    """Inverse of :func:`scalar_to_fit_space`."""
    return float(np.exp(z)) if is_log(spec) else float(z)
