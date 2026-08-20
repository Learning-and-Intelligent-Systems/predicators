"""Post-fit identifiability verdicts for the rollout system-ID stack.

Posterior-vs-prior contraction reporting (MCMC widths or the noise-
aware SSE curvature probe), at-bound detection, trust selection of the
fitted values, and the human/agent-readable rendering. See the
:mod:`predicators.code_sim_learning.physical_sysid` module docstring
for why non-identifiability is reported rather than regularized away.
"""

from __future__ import annotations

import enum
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from predicators.code_sim_learning.fit_space import LOG_FLOOR, FitResult, \
    ParamSpec, is_log, param_bounds, scalar_from_fit_space, \
    scalar_to_fit_space

logger = logging.getLogger(__name__)


class Verdict(enum.Enum):
    """Per-parameter trust verdict of the rollout sysID fit.

    The enum is the single decision surface: every consumer (trust
    selection, explorer diagnostics, cross-cycle bookkeeping) branches
    on the member, never on rendered prose. The human/agent-facing
    explanation travels separately in the report entry's ``note`` field
    and is attached by :func:`format_identifiability` at render time.
    """

    IDENTIFIED = "identified"
    WEAKLY_IDENTIFIED = "weakly identified"
    NOT_IDENTIFIED = "NOT identified"
    # The grid-sweep SSE span never cleared the noise floor: rollouts do
    # not respond to this param anywhere in its box on this data.
    INSENSITIVE = "insensitive"
    # The MAP sits on its box edge: the data pushes the param out of its
    # physically-plausible range (usually model error being absorbed).
    AT_BOUND = "at bound"
    # Anchor ablation reverted the param: a refit with it at its
    # baseline is data-equivalent, so the fitted move was compensatory.
    ANCHORED = "anchored"
    # This cycle's confident fit jumped many combined sigmas from the
    # previous cycle's: the posterior is overconfident, and neither
    # value can be preferred on this evidence.
    INCONSISTENT = "INCONSISTENT across cycles"
    UNKNOWN = "unknown"

    @property
    def applies_fitted(self) -> bool:
        """Whether the fitted value is trustworthy enough to deploy."""
        return self in (Verdict.IDENTIFIED, Verdict.WEAKLY_IDENTIFIED)


# Posterior/prior-width ratio thresholds for the identifiability verdicts.
_IDENTIFIED_CONTRACTION = 0.3
_WEAK_CONTRACTION = 0.7

# Same-theta SSE evaluations used to estimate the nondeterminism noise
# floor (grid sweep and identifiability probe use the same count).
NOISE_FLOOR_EVALS = 3

# A fitted value within this fraction of the box width of a bound is
# reported "at bound" (untrustworthy: the optimizer hit the wall).
_AT_BOUND_BOX_FRAC = 1e-3


def identifiability_report(
    result: FitResult,
    sse_fn: Optional[Callable[[Dict[str, float]], float]] = None,
    param_specs: Optional[Sequence[ParamSpec]] = None,
    num_explainable: Optional[int] = None,
    min_posterior_width: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """Per-parameter posterior-vs-prior contraction from a rollout fit.

    ``contraction = posterior_std / prior_std``: ~1 means the data did not
    constrain the parameter at all (the posterior is just the prior — a
    *null* parameter whose MAP value is arbitrary and should not be
    trusted), while values well below 1 mean the trajectories pinned it
    down. This is the analogue of ``mujoco.sysid``'s post-fit confidence
    intervals: non-identifiability is diagnosed and reported, never
    regularized away silently.

    Posterior widths, in preference order per parameter:

    * The MCMC chain's marginal widths, when a chain ran.
    * The **grid landscape**: for a parameter the coordinate sweep
      covered, the width is the half-width (in FIT space) of its
      ``resolved_interval`` — the data-equivalent flat set widened to
      the midpoints toward the nearest evaluations the sweep actually
      REJECTED (``flat_interval`` as the legacy fallback). This makes
      the verdict agree with the landscape by construction: a
      plateau-wide interval reads NOT identified instead of the local
      curvature at an arbitrary plateau point reading "identified"
      (audited on run_20260722_123949, where the probe stamped every
      param identified while its own report noted "the fitted value is
      the edge of this interval ... not a unique optimum").
    * The prior-scale SSE **curvature probe** around the MAP (two
      rollout evals per parameter, see
      :func:`_probe_posterior_widths`) for parameters WITHOUT sweep
      coverage — rule params (never gridded) and fits run with the grid
      disabled — when ``sse_fn`` and ``param_specs`` are given. The
      Laplace covariance from the LM Jacobian is deliberately NOT used:
      finite-difference Jacobians of contact-rich rollouts are
      noise-dominated, and (measured on the domino smoke test) declare
      every parameter identified — the exact failure this report exists
      to catch.
    * Otherwise "unknown".

    ``min_posterior_width`` floors every landscape/probe width in FIT
    space (see ``code_sim_learning_rollout_min_posterior_width``): those
    widths measure local precision, while the replay objective's model
    bias — invisible to any local statistic — dominates the true error
    at the low end.

    Two verdict overrides guard the remaining blind spots:

    * ``num_explainable`` (how many trimmed-in segments back the fit):
      widths measure local *precision*, which a single clean recording
      can make arbitrarily sharp while the value is still biased;
      "identified" on n < 2 segments is downgraded to weakly identified
      with an explicit note.
    * ``result.sensitivity`` (pre-fit screen): a param whose grid-sweep
      SSE span sat inside the same-theta noise floor does not affect
      the rollouts at all on this data; its flat interval spans the box
      trivially, and the screen's dedicated verdict ("insensitive",
      fitted value not applied) says WHY.
    """
    scales = _result_scales(result, param_specs)
    sensitivity = result.sensitivity or {}
    ablation = getattr(result, "anchor_ablation", None) or {}
    if result.samples.shape[0] > 1:
        # Widths in FIT space (log for log-scale params), so the
        # contraction against the fit-space prior width is meaningful.
        arr = np.array(result.samples, dtype=float, copy=True)
        for j, scale in enumerate(scales):
            if scale == "log":
                arr[:, j] = np.log(np.maximum(arr[:, j], LOG_FLOOR))
        post_std = arr.std(axis=0)
    else:
        post_std = np.full(len(result.names), np.nan)
        swept = set()
        for i, name in enumerate(result.names):
            # The resolved interval (flat set widened to the midpoints
            # toward the nearest REJECTED evaluations) is the honest
            # landscape width: the raw flat interval can collapse to a
            # single point under flat-edge bisection (posterior_std = 0,
            # certainty the sweep's finite evaluations cannot support).
            sweep_entry = sensitivity.get(name) or {}
            interval = sweep_entry.get("resolved_interval") or sweep_entry.get(
                "flat_interval")
            if interval is not None:
                post_std[i] = _interval_half_width(interval, scales[i])
                swept.add(name)
        if sse_fn is not None and swept != set(result.names):
            probe_std = _probe_posterior_widths(result,
                                                sse_fn,
                                                param_specs,
                                                skip=swept)
            for i, name in enumerate(result.names):
                if name not in swept:
                    post_std[i] = probe_std[i]
        # Landscape/probe widths measure local precision only; the
        # free-running replay objective carries model bias no local
        # statistic can see (fits land 1-40% off truth on clean data),
        # so the reported width is floored at the configured minimum.
        # NaN (unknown) entries propagate through np.maximum untouched.
        if min_posterior_width > 0:
            post_std = np.maximum(post_std, min_posterior_width)
    at_bound = _params_at_bound(result, param_specs, scales)
    report: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(result.names):
        prior = (float(result.prior_sigma[i])
                 if result.prior_sigma is not None else float("nan"))
        post = float(post_std[i])
        contraction = (post / prior
                       if np.isfinite(prior) and prior > 0 else float("nan"))
        note = ""
        if np.isnan(contraction):
            verdict = Verdict.UNKNOWN
        elif contraction < _IDENTIFIED_CONTRACTION:
            verdict = Verdict.IDENTIFIED
        elif contraction < _WEAK_CONTRACTION:
            verdict = Verdict.WEAKLY_IDENTIFIED
        else:
            verdict = Verdict.NOT_IDENTIFIED
            note = "posterior ~= prior; MAP arbitrary"
        if (verdict is Verdict.IDENTIFIED and num_explainable is not None
                and num_explainable < 2):
            verdict = Verdict.WEAKLY_IDENTIFIED
            note = ("sharp posterior, but only "
                    f"{num_explainable} explainable segment(s) back it")
        if name in at_bound:
            # A MAP pinned at its box edge means the optimizer ran out
            # of box: the data pushes the parameter outside its
            # physically-plausible range (usually a weakly-informed
            # direction absorbing model error). The one-sided curvature
            # probe reads the wall as sharpness (measured: mass fit to
            # the 1.0 hi bound with posterior_std 5e-11 on
            # run_20260711_141026 replay data), so the probe verdict
            # cannot be trusted there.
            verdict = Verdict.AT_BOUND
            note = (f"box {at_bound[name]} edge; data pushes it out of its "
                    "plausible range; fitted value NOT applied")
        sens = sensitivity.get(name)
        if sens is not None and not sens.get("sensitive", True):
            verdict = Verdict.INSENSITIVE
            note = ("rollouts do not respond to this param anywhere in its "
                    "box on this data; fitted value is noise and was NOT "
                    "applied")
        abl = ablation.get(name)
        if abl is not None:
            # The probe's local curvature at a co-adapted MAP is real,
            # so it cannot see that the move was compensatory; the
            # ablation refit's global data-equivalence verdict wins.
            verdict = Verdict.ANCHORED
            note = ("a refit with this param at its baseline is "
                    "data-equivalent; the fitted move was compensatory and "
                    "the baseline is applied")
        report[name] = {
            "posterior_std": post,
            "prior_std": prior,
            "contraction": contraction,
            "verdict": verdict,
            "note": note,
        }
        if sens is not None:
            for key in ("sse_span", "noise_floor"):
                if key in sens:
                    report[name][key] = sens[key]
            interval = sens.get("flat_interval")
            if interval is not None:
                report[name]["flat_interval"] = interval
            resolved = sens.get("resolved_interval")
            if resolved is not None:
                report[name]["resolved_interval"] = resolved
        if abl is not None:
            report[name]["anchor_ablation"] = dict(abl)
    return report


def _interval_half_width(interval: Sequence[float], scale: str) -> float:
    """Half-width of a data-equivalent interval, in FIT space.

    The landscape-derived stand-in for a posterior std: the coordinate
    sweep could not distinguish any value inside ``interval`` on this
    data, so treating its half-width as the marginal uncertainty makes
    the contraction verdict agree with the landscape by construction. A
    degenerate interval (single flat member) reads as width 0 - the
    sweep resolved the value to within its own (bisected, sub-grid)
    resolution.
    """
    lo, hi = float(interval[0]), float(interval[1])
    if scale == "log":
        width = float(np.log(max(hi, LOG_FLOOR)) - np.log(max(lo, LOG_FLOOR)))
    else:
        width = hi - lo
    return 0.5 * max(width, 0.0)


def physics_sigma_points(applied: Dict[str, float],
                         report: Dict[str, Dict[str, Any]],
                         param_specs: Sequence[ParamSpec],
                         num_points: int = 2) -> List[Dict[str, float]]:
    """A grid of perturbations spanning +-1 posterior sigma of the fit.

    Consumed by the capture gate's physics-margin check
    (``agent_plan_validation_physics_margin``) and by the ``sim.run``
    physics sweep: validation rollouts AT the fitted values sample
    execution variability only, so a plan can pass them all while
    having zero margin to the fit's parameter error
    (run_20260723_091108: a capture validated 8/8 at fitted
    lateral_friction 0.5319 failed deterministically at true 0.5). Each
    param whose FITTED value was deployed (``Verdict.applies_fitted``)
    and whose reported ``posterior_std`` is finite and nonzero is swept
    in FIT space (multiplicative for log-scale params) over
    ``num_points`` evenly spaced offsets in [-1, +1] sigma, all
    perturbed params moving together along that diagonal, each value
    clipped to its box. ``num_points=2`` gives the two endpoints only -
    provably insufficient near a feasibility boundary, where success is
    a SPECKLED function of the params: run_20260724_140531's captured
    design passed both endpoints (0.4295/0.5246) and failed
    deterministically at the true value 0.5 inside them, because the
    k*-minimal task placed a ~0.015-wide failure hole exactly there. A
    dense grid (see agent_plan_validation_physics_margin_points) makes
    interior holes at grid resolution visible; rollouts are
    deterministic per point, so each point is a measurement, not a
    sample.

    The swept interval per param is the DISAGREEMENT HULL: the +-1
    sigma band around the applied value, widened to cover every
    ``candidate_values`` entry in the report (per-segment fits the
    consistency loop dropped, cross-cycle fits the INCONSISTENT
    adjuster recorded). The floored posterior_std alone is provably
    too narrow when data subsets disagree - run_20260724_232411 seed1
    shipped 1.0358 at the 0.1 floor while its segment fits spanned
    [~0.27, ~1.04] around the true 0.5, and only the hull contains the
    truth (an inflated sigma centered on the survivors' fit,
    [0.533, 2.013], still misses it). An INCONSISTENT param IS swept
    (its applied value is the held/trusted one and the hull covers
    both incompatible fits) - excluding it disarmed the gate exactly
    when uncertainty was largest (run_20260724_232411 seed2 cycle 2).
    Params kept at their anchor (NOT identified / insensitive /
    at-bound / anchored) are NOT perturbed: their reported width is
    prior-scale (the data never constrained them), and swinging the
    belief physics that far would reject every plan for uncertainty the
    fit was never asked to resolve. Returns the override dicts in
    ascending order along the hull, the exact fitted point and clipping
    duplicates dropped - or an empty list when nothing is perturbable,
    in which case the margin check is honestly vacuous (see the
    min-width floor in :func:`identifiability_report`, which exists
    precisely so a degenerate landscape width cannot silence it).
    """
    assert num_points >= 2, "need at least the two hull endpoints"
    spec_by_name = {s.name: s for s in param_specs}
    perturbable: Dict[str, Any] = {}
    for name, value in applied.items():
        spec = spec_by_name.get(name)
        entry = report.get(name)
        if spec is None or entry is None:
            continue
        verdict = entry.get("verdict")
        if (verdict is not None and not verdict.applies_fitted
                and verdict is not Verdict.INCONSISTENT):
            continue
        z_val = scalar_to_fit_space(spec, float(value))
        zs = [z_val]
        width = float(entry.get("posterior_std", float("nan")))
        if np.isfinite(width) and width > 0:
            zs.extend([z_val - width, z_val + width])
        for cand in entry.get("candidate_values", ()):
            if float(cand) > 0 or not is_log(spec):
                zs.append(scalar_to_fit_space(spec, float(cand)))
        z_lo, z_hi = min(zs), max(zs)
        if z_hi - z_lo <= 0:
            continue
        box_lo, box_hi = param_bounds([spec])
        perturbable[name] = (spec, z_lo, z_hi, box_lo[0], box_hi[0])
    if not perturbable:
        return []
    points: List[Dict[str, float]] = []
    for t in np.linspace(0.0, 1.0, num_points):
        point = {k: float(v) for k, v in applied.items()}
        moved = False
        for name, (p_spec, z_lo, z_hi, lo, hi) in perturbable.items():
            val = float(
                np.clip(
                    scalar_from_fit_space(p_spec, z_lo + t * (z_hi - z_lo)),
                    lo, hi))
            point[name] = val
            if val != float(applied[name]):
                moved = True
        if moved and point not in points:
            points.append(point)
    return points


def _params_at_bound(
    result: FitResult,
    param_specs: Optional[Sequence[ParamSpec]],
    scales: Sequence[str],
) -> Dict[str, str]:
    """Params whose MAP sits on its box edge, mapped to "lo" / "hi".

    Proximity is judged in FIT space (log for log-scale params) within
    0.1% of the box width, so "at the low end of a decades-spanning
    box" is measured multiplicatively. Without ``param_specs`` (no
    bounds known) nothing is flagged.
    """
    if not param_specs:
        return {}
    point = result.point_estimate
    lo, hi = param_bounds(list(param_specs))
    out: Dict[str, str] = {}
    for i, spec in enumerate(param_specs):
        name = spec.name
        if name not in point:
            continue
        x, lo_i, hi_i = float(point[name]), float(lo[i]), float(hi[i])
        if scales[i] == "log":
            x = float(np.log(max(x, LOG_FLOOR)))
            lo_i = float(np.log(lo_i)) if lo_i > 0 else -np.inf
            hi_i = float(np.log(hi_i)) if np.isfinite(hi_i) else np.inf
        if np.isfinite(lo_i) and np.isfinite(hi_i):
            tol = _AT_BOUND_BOX_FRAC * (hi_i - lo_i)
        else:
            tol = 1e-6
        if np.isfinite(lo_i) and x - lo_i <= tol:
            out[name] = "lo"
        elif np.isfinite(hi_i) and hi_i - x <= tol:
            out[name] = "hi"
    return out


def _result_scales(result: FitResult,
                   param_specs: Optional[Sequence[ParamSpec]]) -> List[str]:
    """Per-column fit scales for ``result``, from specs or the result."""
    if param_specs:
        return [getattr(s, "scale", "linear") for s in param_specs]
    if result.scales is not None:
        return list(result.scales)
    return ["linear"] * len(result.names)


def _probe_posterior_widths(
    result: FitResult,
    sse_fn: Callable[[Dict[str, float]], float],
    param_specs: Optional[Sequence[ParamSpec]],
    skip: Optional[set] = None,
) -> np.ndarray:
    """Posterior widths via a prior-scale SSE curvature probe at the MAP.

    For each parameter, perturb it by ±prior_sigma (clipped to its box)
    with all others held at the MAP, and turn the mean SSE increase into
    a quadratic-approximation posterior width:
    ``SSE(x±d) - SSE(x) ≈ c·d²`` ⇒ posterior precision ``c/noise²`` ⇒
    ``post_std = noise/sqrt(c)``. A flat direction (no SSE increase over
    the whole prior scale) yields ``inf`` ⇒ contraction ≥ 1 ⇒ "NOT
    identified".

    The probe is noise-aware: chaotic contact rollouts are not perfectly
    repeatable (residual sub-tolerance state between evaluations gets
    amplified), so the SSE at the MAP is evaluated three times and the
    observed same-theta spread is subtracted from every perturbation's
    SSE increase before it counts as curvature. Without this, the jitter
    itself reads as curvature and every parameter is declared identified
    — the failure mode of run_20260705_203314, where a ~5k prior-scale
    d_sse sat inside a ~±8k same-theta noise floor yet reported
    contraction 0.00 on all params.

    Parameters named in ``skip`` (those whose width the grid landscape
    already supplies) are not perturbed and get ``nan`` placeholders,
    saving two rollout evals each.
    """
    skip = skip or set()
    point = result.point_estimate
    noise = result.noise_sigma if result.noise_sigma else 0.05
    assert result.prior_sigma is not None
    scales = _result_scales(result, param_specs)
    if param_specs:
        lo, hi = param_bounds(list(param_specs))
        bounds = {
            s.name: (float(lo[i]), float(hi[i]))
            for i, s in enumerate(param_specs)
        }
    else:
        bounds = {}
    sse0_evals = [sse_fn(dict(point)) for _ in range(NOISE_FLOOR_EVALS)]
    sse0 = float(np.median(sse0_evals))
    noise_floor = float(np.max(sse0_evals) - np.min(sse0_evals))
    if noise_floor > 0:
        logger.info(
            "Identifiability probe: same-theta SSE noise floor %.4f "
            "(MAP evals: %s); curvature below it is discounted.", noise_floor,
            [f"{v:.4f}" for v in sse0_evals])
    widths: List[float] = []
    for i, name in enumerate(result.names):
        if name in skip:
            widths.append(float("nan"))
            continue
        sigma = float(result.prior_sigma[i])
        x = point[name]
        lo_i, hi_i = bounds.get(name, (-np.inf, np.inf))
        # Probe in the FIT space: for a log param the step is
        # multiplicative (x * e^±delta) and the curvature is measured
        # against the log-space delta, matching the log-space prior
        # width. This also stops a bound-hugging MAP from reading as
        # sharp curvature merely because the linear box ends there —
        # the flat low-friction basin of run_20260706_171526 spans a
        # 10x range that a linear probe sees as 4.5% of the box.
        if scales[i] == "log":
            x_fit = float(np.log(x))
            lo_fit = float(np.log(lo_i)) if lo_i > 0 else -np.inf
            hi_fit = float(np.log(hi_i)) if np.isfinite(hi_i) else np.inf
        else:
            x_fit, lo_fit, hi_fit = x, lo_i, hi_i
        curvatures: List[float] = []
        for sgn in (1.0, -1.0):
            room = (hi_fit - x_fit) if sgn > 0 else (x_fit - lo_fit)
            delta = min(sigma, room)
            if delta <= 1e-9:
                continue
            pert = dict(point)
            pert_fit = x_fit + sgn * delta
            pert[name] = (float(np.exp(pert_fit))
                          if scales[i] == "log" else pert_fit)
            d_sse = max(sse_fn(pert) - sse0 - noise_floor, 0.0)
            curvatures.append(d_sse / delta**2)
        c = float(np.mean(curvatures)) if curvatures else 0.0
        widths.append(noise / np.sqrt(c) if c > 0 else float("inf"))
    return np.asarray(widths, dtype=float)


def select_trustworthy_params(
    fitted: Dict[str, float],
    declared_inits: Dict[str, float],
    physical_names: Sequence[str],
    report: Dict[str, Dict[str, Any]],
    anchors: Optional[Dict[str, float]] = None,
    held: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Pick which fitted physical values are safe to apply to the planner.

    A parameter whose posterior did not contract (NOT_IDENTIFIED /
    UNKNOWN) or that failed the sensitivity screen (INSENSITIVE) has
    an arbitrary MAP — on uninformative data the grid seed and LM land
    wherever the rollout noise happened to be lowest — so applying it
    would move the planner's belief randomly, possibly further from the
    truth. For those, keep the env-registry ANCHOR when one is known
    (falling back to the declared init): the anchor is the env's
    standing baseline belief, whereas the declared init is this call's
    agent hypothesis, which unsupported data must not smuggle into the
    planner (observed: a declared restitution init of 0.15 surviving as
    "kept init" against a 0.02 baseline). Apply the fitted value only
    for parameters the data actually constrained
    (``Verdict.applies_fitted``).

    An INCONSISTENT parameter (this cycle's confident fit jumped many
    combined sigmas from the previous cycle's) keeps its entry in
    ``held`` — the value the planner is currently running with — when
    one exists: neither of the two mutually-incompatible fits can be
    preferred on this evidence, and hopping between them churned the
    belief env for whole runs (run_20260721_205821 seed1: restitution
    0.71 -> 0.52 -> 0.02 -> 0.32 -> 0.02 across cycles, every hop
    "identified"). Without a held value it falls back to the anchor
    like the other untrusted verdicts.
    """
    anchors = anchors or {}
    held = held or {}
    applied: Dict[str, float] = {}
    for name in physical_names:
        verdict = report.get(name, {}).get("verdict", Verdict.UNKNOWN)
        if verdict.applies_fitted:
            applied[name] = fitted[name]
            continue
        if verdict is Verdict.INCONSISTENT and name in held:
            applied[name] = held[name]
            logger.info(
                "Rollout sysID: NOT applying %s=%.4f (%s); holding the "
                "last trusted value %.4f (the margin sweep spans both).", name,
                fitted[name], verdict.value, held[name])
            continue
        fallback = anchors.get(name, declared_inits[name])
        applied[name] = fallback
        if fitted[name] != fallback:
            logger.info(
                "Rollout sysID: NOT applying %s=%.4f (verdict: %s); "
                "keeping the %s %.4f.", name, fitted[name], verdict.value,
                "registry anchor" if name in anchors else "declared init",
                fallback)
    return applied


def format_identifiability(report: Dict[str, Dict[str, Any]]) -> str:
    """Human/agent-readable rendering of :func:`identifiability_report`."""
    lines = []
    for name, info in report.items():
        verdict = info["verdict"]
        note = info.get("note", "")
        label = verdict.value + (f" ({note})" if note else "")
        lines.append(f"  {name:<28} posterior_std={info['posterior_std']:.4g}"
                     f"  prior_std={info['prior_std']:.4g}"
                     f"  contraction={info['contraction']:.2f}"
                     f"  -> {label}")
        interval = info.get("flat_interval")
        if interval is not None and interval[0] != interval[1]:
            note = (" - the fitted value is the edge of this interval "
                    "nearest the baseline belief, not a unique optimum"
                    if verdict in (Verdict.WEAKLY_IDENTIFIED,
                                   Verdict.NOT_IDENTIFIED) else "")
            lines.append(f"      data-equivalent over [{interval[0]:.4g}, "
                         f"{interval[1]:.4g}]{note}")
        cands = info.get("candidate_values", ())
        if len(cands) > 1:
            lines.append(
                f"      disagreement hull [{min(cands):.4g}, "
                f"{max(cands):.4g}]: data subsets preferred incompatible "
                "explanations; the physics-margin sweep spans this hull")
        abl = info.get("anchor_ablation")
        if abl is not None:
            fitted = (f" (reverted from {abl['fitted']:.4g})"
                      if "fitted" in abl else "")
            lines.append(f"      anchor ablation: SSE {abl['sse_pinned']:.4g} "
                         f"with it refit-pinned at baseline "
                         f"{abl['anchor']:.4g}{fitted} vs "
                         f"{abl['sse_map']:.4g} at the joint MAP "
                         f"(tol {abl['tol']:.4g})")
    return "\n".join(lines)
