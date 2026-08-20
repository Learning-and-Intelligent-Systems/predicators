"""Tests for log-scale parameter support in the sysID stack.

A log-scale ``ParamSpec`` fits ``z = log(theta)``: geometric grid
sweeps, multiplicative LM steps, a log-normal prior, and a log-space
identifiability probe. Regression anchor: run_20260706_171526, where a
linear parameterization over friction's [0.01, 2.0] box had no grid
candidate between 0.01 and 0.29, fitted 0.0114 for a true 0.1, and the
linear curvature probe blessed the bound-hugging MAP as "identified".
"""

import numpy as np
import pytest

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
from predicators.code_sim_learning.active_experiment import laplace_ensemble, \
    perturbation_ensemble
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec, \
    fit_space_bounds, from_fit_space, prior_widths, rows_from_fit_space, \
    to_fit_space
from predicators.code_sim_learning.grid_seed import grid_candidates
from predicators.code_sim_learning.identifiability import Verdict, \
    identifiability_report
from predicators.code_sim_learning.lm import solve_lm
from predicators.code_sim_learning.utils import stamp_physical_spec_scales

# ── ParamSpec validation ──────────────────────────────────────────


def test_paramspec_rejects_unknown_scale():
    """A scale that is neither "linear" nor "log" is rejected."""
    with pytest.raises(ValueError, match="linear.*log"):
        ParamSpec("friction", 0.5, lo=0.01, hi=2.0, scale="exp")


def test_paramspec_log_requires_positive_init_and_lo():
    """A log-scale spec requires a positive init_value and lower bound."""
    with pytest.raises(ValueError, match="positive init_value"):
        ParamSpec("friction", -0.5, lo=0.01, hi=2.0, scale="log")
    with pytest.raises(ValueError, match="positive lo"):
        ParamSpec("friction", 0.5, lo=0.0, hi=2.0, scale="log")


# ── Transforms, bounds, prior widths ──────────────────────────────


def _mixed_specs():
    return [
        ParamSpec("friction", 0.5, lo=0.01, hi=2.0, scale="log"),
        ParamSpec("offset", 0.0, lo=-0.3, hi=0.3),
    ]


def test_transform_round_trip():
    """log columns map through log/exp; linear columns pass untouched."""
    specs = _mixed_specs()
    ext = [0.1, -0.2]
    z = to_fit_space(specs, ext)
    assert np.isclose(z[0], np.log(0.1))
    assert np.isclose(z[1], -0.2)  # linear param untouched
    assert np.allclose(from_fit_space(specs, z), ext)


def test_internal_bounds_log_param():
    """Internal bounds are log-transformed for log params, raw for linear."""
    lo, hi = fit_space_bounds(_mixed_specs())
    assert np.isclose(lo[0], np.log(0.01))
    assert np.isclose(hi[0], np.log(2.0))
    assert np.isclose(lo[1], -0.3) and np.isclose(hi[1], 0.3)


def test_rows_to_external_only_touches_log_columns():
    """Batched internal->external conversion exponentiates only log columns."""
    specs = _mixed_specs()
    rows = np.array([[np.log(0.1), -0.2], [np.log(2.0), 0.1]])
    out = rows_from_fit_space(specs, rows)
    assert np.allclose(out, [[0.1, -0.2], [2.0, 0.1]])


def test_prior_widths_log_param_is_constant_in_log_space():
    """A log param's prior sigma is the scale itself, independent of init."""
    specs = _mixed_specs()
    widths = prior_widths(specs, 0.75)
    # Log param: sigma = the scale itself, independent of init — one
    # sigma spans the same multiplicative factor everywhere.
    assert np.isclose(widths[0], 0.75)
    # Linear param with init ~0 falls back to half the bound range.
    assert np.isclose(widths[1], 0.3)


# ── Grid candidates ───────────────────────────────────────────────


def test_grid_candidates_log_covers_low_decades():
    """geomspace puts a candidate near a true value of 0.1.

    linspace(0.01, 2.0, 8) has nothing between 0.01 and 0.29 — the gap
    that sent run_20260706_171526's fit to the 0.01 endpoint.
    """
    spec = ParamSpec("friction", 0.5, lo=0.01, hi=2.0, scale="log")
    cands = grid_candidates(spec, 8)
    assert min(abs(v - 0.1) for v in cands) < 0.01
    assert np.isclose(cands[0], 0.01) and np.isclose(cands[-1], 2.0)


def test_grid_candidates_linear_unchanged():
    """A linear param still gets an evenly spaced linspace grid."""
    spec = ParamSpec("offset", 0.0, lo=-0.3, hi=0.3)
    assert np.allclose(grid_candidates(spec, 7), np.linspace(-0.3, 0.3, 7))


# ── LM in log space ───────────────────────────────────────────────


def test_solve_lm_recovers_low_target_from_high_init():
    """LM walks 0.5 -> 0.02 (a 25x drop) in one fit, log-parameterized."""
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0, scale="log")]

    def resid(theta):
        return np.array([np.log(theta[0]) - np.log(0.02)] * 3)

    x, jac = solve_lm(resid, specs, 200, "test")
    assert abs(x[0] - 0.02) < 1e-4
    assert jac is not None and np.all(np.abs(jac) > 1e-6)


def test_solve_lm_jacobian_nonzero_at_unit_theta():
    """z = log(1) = 0 is the relative-diff-step edge case; the step must
    not collapse to nothing there."""
    specs = [ParamSpec("k", 1.0, lo=0.01, hi=10.0, scale="log")]

    def resid(theta):
        return np.array([theta[0] - 3.0])

    x, jac = solve_lm(resid, specs, 200, "test", diff_step=2e-2)
    assert abs(x[0] - 3.0) < 1e-3
    assert jac is not None and np.all(np.abs(jac) > 1e-6)


# ── Identifiability probe in log space ────────────────────────────


def _point_result(names, values, prior_sigma, scales):
    return FitResult(names=list(names),
                     samples=np.array([values], dtype=float),
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=0.05,
                     prior_sigma=np.asarray(prior_sigma, dtype=float),
                     scales=list(scales))


def test_probe_flat_low_basin_not_identified_in_log_space():
    """A bound-hugging MAP inside a flat multiplicative basin must read as NOT
    identified.

    SSE is flat for friction < 0.3 and the MAP sits at 0.0114 near the
    0.01 bound. The old linear probe stepped +prior_sigma=0.375 out of
    the basin, saw curvature, and declared "identified" (the false
    verdict of run_20260706_171526). The log probe's multiplicative
    steps (x2.1 up, bounded room down) stay inside the flat basin.
    """
    result = _point_result(["friction"], [0.0114], [0.75], ["log"])
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0, scale="log")]

    def sse_fn(params):
        return 0.0 if params["friction"] < 0.3 else 1e6

    report = identifiability_report(result, sse_fn, specs)
    assert report["friction"]["verdict"] is Verdict.NOT_IDENTIFIED


def test_probe_sharp_log_curvature_identified():
    """Real curvature at the MAP in log space is still identified."""
    result = _point_result(["friction"], [0.1], [0.75], ["log"])
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0, scale="log")]

    def sse_fn(params):
        return 1e4 * (np.log(params["friction"]) - np.log(0.1))**2

    report = identifiability_report(result, sse_fn, specs)
    assert report["friction"]["verdict"] is Verdict.IDENTIFIED


def test_mcmc_samples_contraction_measured_in_log_space():
    """Chain widths for log params compare log-space std to the log-space prior
    width."""
    rng = np.random.default_rng(0)
    # Tight multiplicatively (std 0.05 in log) vs prior-wide (std 0.75).
    tight = np.exp(rng.normal(np.log(0.1), 0.05, size=400))
    wide = np.exp(rng.normal(np.log(0.5), 0.75, size=400))
    result = FitResult(names=["friction", "mass"],
                       samples=np.column_stack([tight, wide]),
                       log_probs=np.zeros(400),
                       jacobian=None,
                       noise_sigma=0.05,
                       prior_sigma=np.array([0.75, 0.75]),
                       scales=["log", "log"])
    report = identifiability_report(result)
    assert report["friction"]["verdict"] is Verdict.IDENTIFIED
    assert report["mass"]["verdict"] is Verdict.NOT_IDENTIFIED


# ── Ensembles ─────────────────────────────────────────────────────


def test_perturbation_ensemble_log_param_stays_positive_and_boxed():
    """Multiplicative log-space perturbations stay positive and in-bounds."""
    specs = [ParamSpec("friction", 0.1, lo=0.01, hi=2.0, scale="log")]
    rng = np.random.default_rng(0)
    members = perturbation_ensemble({"friction": 0.1},
                                    specs,
                                    num_members=64,
                                    perturb_frac=0.25,
                                    rng=rng)
    values = [m["friction"] for m in members]
    assert values[0] == 0.1  # anchor
    assert all(0.01 <= v <= 2.0 for v in values)
    assert len(set(np.round(values, 6))) > 1  # actually perturbs


def test_laplace_ensemble_log_param_draws_are_multiplicative():
    """With a fit-space Jacobian/prior, draws exponentiate back to positive
    external values inside the box."""
    specs = [ParamSpec("friction", 0.1, lo=0.01, hi=2.0, scale="log")]
    rng = np.random.default_rng(0)
    # A weakly-informative Jacobian in z: posterior ~ prior (0.75 wide
    # in log) — draws should spread over decades but stay positive.
    jac = np.array([[1e-3]])
    members = laplace_ensemble({"friction": 0.1}, ["friction"],
                               specs,
                               jac,
                               noise_sigma=0.05,
                               prior_sigma=np.array([0.75]),
                               num_members=64,
                               rng=rng)
    values = np.array([m["friction"] for m in members])
    assert values[0] == 0.1  # anchor
    assert np.all(values >= 0.01) and np.all(values <= 2.0)
    ratios = values[1:] / 0.1
    # Multiplicative spread: both halving and doubling should occur.
    assert (ratios < 0.7).any() and (ratios > 1.4).any()


def test_laplace_ensemble_stiff_log_direction_barely_moves():
    """A stiff (informative) log direction gets a tiny posterior spread."""
    specs = [ParamSpec("friction", 0.1, lo=0.01, hi=2.0, scale="log")]
    rng = np.random.default_rng(0)
    # Very informative Jacobian in z -> tiny posterior width.
    jac = np.full((50, 1), 100.0)
    members = laplace_ensemble({"friction": 0.1}, ["friction"],
                               specs,
                               jac,
                               noise_sigma=0.05,
                               prior_sigma=np.array([0.75]),
                               num_members=32,
                               rng=rng)
    values = np.array([m["friction"] for m in members])
    assert np.all(np.abs(np.log(values / 0.1)) < 0.01)


# ── Registry stamping ─────────────────────────────────────────────


class _FakeEnv:

    def get_physical_param_info(self):
        """Return a registry with one log-scale and one linear param."""
        return {
            "friction": {
                "default": 0.5,
                "lo": 0.01,
                "hi": 2.0,
                "scale": "log",
                "description": "",
            },
            "restitution": {
                "default": 0.02,
                "lo": 0.0,
                "hi": 0.9,
                "description": "",
            },
        }


def test_stamp_scales_registry_wins_over_agent_default():
    """The env registry's scale overrides the agent's default per param."""
    agent_specs = [
        ParamSpec("friction", 0.5, lo=0.01, hi=2.0),  # agent omits scale
        ParamSpec("restitution", 0.02, lo=0.0, hi=0.9),
        ParamSpec("unlisted", 1.0, lo=0.1, hi=10.0, scale="log"),
    ]
    stamped = stamp_physical_spec_scales(agent_specs, _FakeEnv())
    by_name = {s.name: s for s in stamped}
    assert by_name["friction"].scale == "log"  # stamped from registry
    assert by_name["restitution"].scale == "linear"
    assert by_name["unlisted"].scale == "log"  # agent's choice kept
    # Bounds and inits survive the stamping untouched.
    assert by_name["friction"].lo == 0.01 and by_name["friction"].hi == 2.0


def test_stamp_scales_env_without_registry_is_identity():
    """An env lacking a param registry leaves the agent specs unchanged."""
    agent_specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0)]
    stamped = stamp_physical_spec_scales(agent_specs, object())
    assert stamped[0].scale == "linear"
    assert stamped[0].name == "friction"
