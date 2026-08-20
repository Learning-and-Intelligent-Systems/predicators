"""Tests for the rollout system-ID pure-Python pieces.

The rollout engine's physics (PyBullet resets, velocity zeroing) is
exercised end-to-end by the domino experiments; here we unit-test the
noise-aware curvature probe (given an ``sse_fn``), the settled-tail
trajectory truncation, and the fresh-env-per-rollout plumbing.
"""
# pylint: disable=protected-access,import-outside-toplevel

import numpy as np
import pybullet as p
import pytest

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
from predicators.code_sim_learning import grid_seed, physical_sysid, \
    rollout_env, rollout_objective, trajectory_prep
from predicators.code_sim_learning.fit_space import FitResult, ParamSpec
from predicators.code_sim_learning.identifiability import Verdict, \
    format_identifiability, identifiability_report, \
    select_trustworthy_params
from predicators.code_sim_learning.physical_sysid import \
    fit_params_rollout_trimmed
from predicators.code_sim_learning.trajectory_prep import truncate_settled_tail
from predicators.structs import Action, Object, State, Type

_DOMINO_TYPE = Type("domino", ["x"])
_ROBOT_TYPE = Type("robot", ["x"])
_RESIDUAL_FEATURES = {"domino": ["x"]}


def _trajectory(domino_xs, robot_xs=None):
    """Build a (states, actions) trajectory from per-step feature values."""
    domino = Object("d0", _DOMINO_TYPE)
    robot = Object("r0", _ROBOT_TYPE)
    if robot_xs is None:
        robot_xs = [0.0] * len(domino_xs)
    states = [
        State({
            domino: np.array([dx], dtype=float),
            robot: np.array([rx], dtype=float),
        }) for dx, rx in zip(domino_xs, robot_xs)
    ]
    actions = [
        Action(np.zeros(1, dtype=np.float32)) for _ in range(len(states) - 1)
    ]
    return states, actions


def test_truncate_cuts_static_tail():
    """Motion for 10 steps, static for 90 -> cut at last motion + margin."""
    xs = [0.01 * i for i in range(11)] + [0.1] * 90
    states, actions = truncate_settled_tail(_trajectory(xs),
                                            _RESIDUAL_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    # Last moving step is index 9 (states[9] -> states[10]); keep 9+1+5.
    assert len(actions) == 15
    assert len(states) == 16


def test_truncate_keeps_intermediate_pause():
    """Push -> pause -> second push: the cut anchors to the LAST motion."""
    xs = ([0.01 * i for i in range(11)] + [0.1] * 40 +
          [0.1 + 0.01 * i for i in range(10)] + [0.2] * 60)
    states, actions = truncate_settled_tail(_trajectory(xs),
                                            _RESIDUAL_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    # Last motion is the step onto the final plateau (index 60); both
    # pushes and the pause between them are retained.
    assert len(actions) == 66
    assert states[-1].get(Object("d0", _DOMINO_TYPE), "x") == 0.2


def test_truncate_noop_when_motion_never_stops():
    """Continuous motion to the last step -> nothing is cut."""
    xs = [0.01 * i for i in range(100)]
    traj = _trajectory(xs)
    states, actions = truncate_settled_tail(traj,
                                            _RESIDUAL_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    assert len(actions) == len(traj[1])
    assert len(states) == len(traj[0])


def test_truncate_static_trajectory_keeps_margin_prefix():
    """No scored feature ever moves -> keep only the margin prefix."""
    xs = [0.5] * 100
    states, actions = truncate_settled_tail(_trajectory(xs),
                                            _RESIDUAL_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    assert len(actions) == 5
    assert len(states) == 6


def test_truncate_ignores_unscored_features():
    """Robot motion is not scored, so it must not defer the cut."""
    domino_xs = [0.01 * i for i in range(11)] + [0.1] * 90
    robot_xs = [0.02 * i for i in range(101)]  # robot moves the whole time
    states, actions = truncate_settled_tail(_trajectory(domino_xs, robot_xs),
                                            _RESIDUAL_FEATURES,
                                            motion_tol=1e-3,
                                            margin=5)
    assert len(actions) == 15
    assert len(states) == 16


def _single_sample_result(names, values, prior_sigma):
    return FitResult(names=list(names),
                     samples=np.array([values], dtype=float),
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=0.05,
                     prior_sigma=np.asarray(prior_sigma, dtype=float))


def test_probe_identifies_curved_param_with_deterministic_sse():
    """A strongly curved direction is identified; a flat one is not."""
    result = _single_sample_result(["curved", "flat"], [0.5, 0.5],
                                   [0.25, 0.25])
    specs = [
        ParamSpec("curved", 0.5, lo=0.0, hi=1.0),
        ParamSpec("flat", 0.5, lo=0.0, hi=1.0),
    ]

    def sse_fn(params):
        return 1e4 * (params["curved"] - 0.5)**2

    report = identifiability_report(result, sse_fn, specs)
    assert report["curved"]["verdict"] is Verdict.IDENTIFIED
    assert report["flat"]["verdict"] is Verdict.NOT_IDENTIFIED


def test_probe_discounts_curvature_below_noise_floor():
    """Same-theta SSE jitter must not read as curvature.

    Regression for run_20260705_203314: a chaotic rollout objective
    jittered by thousands of SSE units at the SAME parameters, and the
    probe declared every parameter identified because the jitter looked
    like a large SSE increase under perturbation. With the noise-aware
    probe, a perturbation response inside the same-theta spread yields
    no curvature -> NOT identified.
    """
    result = _single_sample_result(["friction"], [0.5], [0.375])
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0)]

    # Deterministic pseudo-noise: SSE cycles through a large spread at
    # every call, independent of the parameters (pure jitter, no signal).
    values = [24748.0, 27359.0, 9671.0, 25406.0, 20000.0, 30000.0]
    calls = {"n": 0}

    def sse_fn(_params):
        v = values[calls["n"] % len(values)]
        calls["n"] += 1
        return v

    report = identifiability_report(result, sse_fn, specs)
    assert report["friction"]["verdict"] is Verdict.NOT_IDENTIFIED


def test_probe_survives_noise_floor_on_top_of_signal():
    """Real curvature well above the noise floor is still identified."""
    result = _single_sample_result(["friction"], [0.5], [0.375])
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2.0)]
    rng = np.random.default_rng(0)

    def sse_fn(params):
        # Strong quadratic signal + jitter two orders below the signal
        # at prior scale.
        signal = 1e6 * (params["friction"] - 0.1)**2
        return signal + float(rng.uniform(0.0, 100.0))

    report = identifiability_report(result, sse_fn, specs)
    assert report["friction"]["verdict"] is Verdict.IDENTIFIED


def test_resolved_interval_extends_collapsed_flat_set():
    """A single-point flat set resolves to the midpoints toward the nearest
    REJECTED evaluations, not to zero width.

    Regression for run_20260723_091108: flat-edge bisection lowered the
    best SSE until the flat set collapsed to the refined point, and the
    zero-width flat interval read as posterior_std=0 ("identified" with
    infinite confidence in a value +6.4% off truth).
    """
    spec = ParamSpec("friction", 0.5, lo=0.01, hi=2.0, scale="log")
    # Evaluated pool: rejected neighbors at 0.4 and 0.8, flat set {0.53}.
    pool = [(0.4, 10.0), (0.53, 1.0), (0.8, 10.0)]
    lo, hi = grid_seed._resolved_interval(spec, pool, [0.53])
    # Fit-space (log) midpoints toward the rejected neighbors.
    assert np.isclose(np.log(lo), 0.5 * (np.log(0.4) + np.log(0.53)))
    assert np.isclose(np.log(hi), 0.5 * (np.log(0.8) + np.log(0.53)))
    assert lo < 0.53 < hi


def test_resolved_interval_edge_without_rejected_neighbor():
    """A flat set reaching the last evaluated value keeps that edge."""
    spec = ParamSpec("friction", 0.5, lo=0.01, hi=2.0)
    # No rejected value above the flat set.
    pool = [(0.2, 10.0), (0.5, 1.0), (0.8, 1.0)]
    lo, hi = grid_seed._resolved_interval(spec, pool, [0.5, 0.8])
    assert np.isclose(lo, 0.5 * (0.2 + 0.5))
    assert np.isclose(hi, 0.8)


def _swept_result(sensitivity):
    return FitResult(names=["friction"],
                     samples=np.array([[0.53]], dtype=float),
                     log_probs=np.zeros(1),
                     jacobian=None,
                     noise_sigma=0.05,
                     prior_sigma=np.asarray([0.75], dtype=float),
                     scales=["log"],
                     sensitivity=sensitivity)


def test_report_uses_resolved_interval_and_floor():
    """Swept width = max(resolved-interval half-width, configured floor)."""
    specs = [ParamSpec("friction", 0.53, lo=0.01, hi=2.0, scale="log")]
    sens = {
        "friction": {
            "sse_span": 100.0,
            "noise_floor": 0.0,
            # Degenerate flat interval (bisection collapse); resolved
            # interval spans x1.5 in external units.
            "flat_interval": (0.53, 0.53),
            "resolved_interval": (0.53, 0.53 * 1.5),
        }
    }
    report = identifiability_report(_swept_result(sens),
                                    param_specs=specs,
                                    min_posterior_width=0.0)
    expected = 0.5 * np.log(1.5)
    assert np.isclose(report["friction"]["posterior_std"], expected)
    # A floor above the landscape width takes over.
    report = identifiability_report(_swept_result(sens),
                                    param_specs=specs,
                                    min_posterior_width=0.4)
    assert np.isclose(report["friction"]["posterior_std"], 0.4)
    assert report["friction"]["resolved_interval"] == (0.53, 0.53 * 1.5)


def test_report_floor_prevents_degenerate_zero_width():
    """A fully collapsed landscape width never reads as certainty."""
    specs = [ParamSpec("friction", 0.53, lo=0.01, hi=2.0, scale="log")]
    sens = {
        "friction": {
            "sse_span": 100.0,
            "noise_floor": 0.0,
            "flat_interval": (0.53, 0.53),
            "resolved_interval": (0.53, 0.53),
        }
    }
    report = identifiability_report(_swept_result(sens),
                                    param_specs=specs,
                                    min_posterior_width=0.1)
    assert np.isclose(report["friction"]["posterior_std"], 0.1)
    # Still identified (0.1 / 0.75 < 0.3) - the floor reports honest
    # width, it does not veto deployment.
    assert report["friction"]["verdict"] is Verdict.IDENTIFIED


def test_physics_sigma_points_log_scale_and_clipping():
    """+-1-sigma points are multiplicative for log params and box-clipped."""
    from predicators.code_sim_learning.identifiability import \
        physics_sigma_points
    specs = [
        ParamSpec("friction", 0.53, lo=0.01, hi=2.0, scale="log"),
        ParamSpec("restitution", 0.02, lo=0.0, hi=0.9),
    ]
    applied = {"friction": 0.53, "restitution": 0.02}
    report = {
        "friction": {
            "posterior_std": 0.1,
            "verdict": Verdict.IDENTIFIED
        },
        "restitution": {
            "posterior_std": 0.05,
            "verdict": Verdict.WEAKLY_IDENTIFIED
        },
    }
    pts = physics_sigma_points(applied, report, specs)
    assert len(pts) == 2
    lo_pt, hi_pt = pts[0], pts[1]
    assert np.isclose(lo_pt["friction"], 0.53 * np.exp(-0.1))
    assert np.isclose(hi_pt["friction"], 0.53 * np.exp(0.1))
    # Linear param moves additively; the low point clips at the box.
    assert np.isclose(lo_pt["restitution"], 0.0)
    assert np.isclose(hi_pt["restitution"], 0.07)


def test_physics_sigma_points_dense_grid():
    """num_points spreads a grid across the +-1-sigma range.

    Endpoints alone cannot see an interior failure hole
    (run_20260724_140531: a captured design passed both +-1-sigma
    endpoints and failed deterministically at the true value between
    them). The fitted point itself is dropped (validation rollouts
    already run there), as are duplicates from box clipping.
    """
    from predicators.code_sim_learning.identifiability import \
        physics_sigma_points
    specs = [ParamSpec("friction", 0.53, lo=0.01, hi=2.0, scale="log")]
    applied = {"friction": 0.4746}
    report = {
        "friction": {
            "posterior_std": 0.1,
            "verdict": Verdict.IDENTIFIED
        }
    }
    pts = physics_sigma_points(applied, report, specs, num_points=5)
    vals = [p["friction"] for p in pts]
    # t in {-1, -0.5, +0.5, +1}; t=0 (the fitted value) is dropped.
    assert np.allclose(vals,
                       [0.4746 * np.exp(t * 0.1) for t in (-1, -0.5, 0.5, 1)])
    # Clipping collapse: a tight box folds interior points onto the
    # bound; duplicates are dropped rather than re-measured.
    tight = [ParamSpec("friction", 0.53, lo=0.46, hi=0.49, scale="log")]
    pts_tight = physics_sigma_points(applied, report, tight, num_points=5)
    tight_vals = [p["friction"] for p in pts_tight]
    assert len(tight_vals) == len(set(tight_vals))
    assert all(0.46 <= v <= 0.49 for v in tight_vals)
    # num_points=2 keeps the historical endpoint behavior.
    endpoints = physics_sigma_points(applied, report, specs, num_points=2)
    assert np.isclose(endpoints[0]["friction"], 0.4746 * np.exp(-0.1))
    assert np.isclose(endpoints[1]["friction"], 0.4746 * np.exp(0.1))


def test_physics_sigma_points_empty_without_width():
    """Zero/NaN posterior width or a missing report entry perturbs nothing."""
    from predicators.code_sim_learning.identifiability import \
        physics_sigma_points
    specs = [ParamSpec("friction", 0.53, lo=0.01, hi=2.0, scale="log")]
    assert not physics_sigma_points({"friction": 0.53},
                                    {"friction": {
                                        "posterior_std": 0.0
                                    }}, specs)
    assert not physics_sigma_points(
        {"friction": 0.53}, {"friction": {
            "posterior_std": float("nan")
        }}, specs)
    assert not physics_sigma_points({"friction": 0.53}, {}, specs)


def test_physics_sigma_points_skip_untrusted_verdicts():
    """Params kept at their anchor (untrusted verdicts) are not perturbed.

    A NOT-identified param's reported width is prior-scale (the data
    never constrained it); swinging the belief physics that far would
    reject every plan for uncertainty the fit was never asked to
    resolve. Only the identified param moves.
    """
    from predicators.code_sim_learning.identifiability import \
        physics_sigma_points
    specs = [
        ParamSpec("friction", 0.53, lo=0.01, hi=2.0, scale="log"),
        ParamSpec("restitution", 0.02, lo=0.0, hi=0.9),
    ]
    applied = {"friction": 0.53, "restitution": 0.02}
    report = {
        "friction": {
            "posterior_std": 0.1,
            "verdict": Verdict.IDENTIFIED
        },
        "restitution": {
            "posterior_std": 0.75,
            "verdict": Verdict.NOT_IDENTIFIED
        },
    }
    pts = physics_sigma_points(applied, report, specs)
    assert len(pts) == 2
    lo_pt, hi_pt = pts[0], pts[1]
    assert lo_pt["restitution"] == 0.02
    assert hi_pt["restitution"] == 0.02
    assert lo_pt["friction"] < 0.53 < hi_pt["friction"]
    # All-untrusted -> nothing to perturb -> vacuous.
    report_none = {
        "friction": {
            "posterior_std": 0.75,
            "verdict": Verdict.NOT_IDENTIFIED
        }
    }
    assert not physics_sigma_points({"friction": 0.53}, report_none, specs[:1])


def test_physics_sigma_points_hull_widens_beyond_sigma():
    """candidate_values (segment/cycle disagreement) widen the sweep.

    Regression for run_20260724_232411 seed1: applied 1.0358 at the
    0.1 sigma floor swept only [0.937, 1.145] while the dropped
    segment's fits pointed at ~0.27; the hull sweep must span from the
    lowest candidate through the +1 sigma endpoint (true 0.5 then lies
    inside the swept interval).
    """
    from predicators.code_sim_learning.identifiability import \
        physics_sigma_points
    specs = [ParamSpec("friction", 1.0358, lo=0.01, hi=2.0, scale="log")]
    applied = {"friction": 1.0358}
    report = {
        "friction": {
            "posterior_std": 0.1,
            "verdict": Verdict.IDENTIFIED,
            "candidate_values": [0.2742, 1.0358],
        }
    }
    pts = physics_sigma_points(applied, report, specs, num_points=32)
    vals = [p["friction"] for p in pts]
    assert min(vals) == pytest.approx(0.2742, rel=1e-6)
    # The upper end is still the +1 sigma endpoint, not the raw fit.
    assert max(vals) == pytest.approx(1.0358 * np.exp(0.1), rel=1e-6)
    assert any(0.45 < v < 0.55 for v in vals)


def test_physics_sigma_points_inconsistent_is_swept():
    """An INCONSISTENT param is swept across both incompatible fits.

    Regression for run_20260724_232411 seed2 cycle 2: the INCONSISTENT
    verdict emptied the sweep entirely (capture ran with NO margin
    check) although the cross-cycle interval [0.3236, 0.6267] contained
    the true 0.5. The applied value is the held/trusted one; the sweep
    must cover the whole disagreement.
    """
    from predicators.code_sim_learning.identifiability import \
        physics_sigma_points
    specs = [ParamSpec("friction", 0.3236, lo=0.01, hi=2.0, scale="log")]
    applied = {"friction": 0.3236}  # held value
    report = {
        "friction": {
            "posterior_std": 0.1,
            "verdict": Verdict.INCONSISTENT,
            "candidate_values": [0.3236, 0.6267],
        }
    }
    pts = physics_sigma_points(applied, report, specs, num_points=16)
    vals = [p["friction"] for p in pts]
    assert min(vals) == pytest.approx(0.3236 * np.exp(-0.1), rel=1e-6)
    assert max(vals) == pytest.approx(0.6267, rel=1e-6)
    assert any(0.45 < v < 0.55 for v in vals)
    # Without candidates the +-1 sigma band still keeps the gate armed.
    report_plain = {
        "friction": {
            "posterior_std": 0.1,
            "verdict": Verdict.INCONSISTENT,
        }
    }
    pts_plain = physics_sigma_points(applied, report_plain, specs)
    assert pts_plain, "INCONSISTENT must not disarm the margin sweep"


def test_select_trustworthy_params_keeps_init_for_unidentified():
    """Un-contracted params keep the declared init; identified ones apply.

    On uninformative data the fitted value of a NOT-identified parameter
    is arbitrary (grid seed lands on a noise minimum), so applying it
    would move the planner's belief randomly.
    """
    fitted = {"friction": 1.337, "restitution": 0.44}
    inits = {"friction": 0.5, "restitution": 0.02}
    report = {
        "friction": {
            "verdict": Verdict.NOT_IDENTIFIED
        },
        "restitution": {
            "verdict": Verdict.IDENTIFIED
        },
    }
    applied = select_trustworthy_params(fitted, inits,
                                        ["friction", "restitution"], report)
    assert applied == {"friction": 0.5, "restitution": 0.44}


def test_select_trustworthy_params_weakly_identified_applies():
    """Weak contraction still counts as data-constrained -> apply."""
    fitted = {"friction": 0.12}
    inits = {"friction": 0.5}
    report = {"friction": {"verdict": Verdict.WEAKLY_IDENTIFIED}}
    applied = select_trustworthy_params(fitted, inits, ["friction"], report)
    assert applied == {"friction": 0.12}


def _patch_fit_and_rms(monkeypatch, fit_thetas, rms_by_count):
    """Stub the heavy PyBullet fit/residual calls for trimming tests.

    ``fit_thetas`` maps the number of trajectories passed to the fit to
    the friction value it "fits"; ``rms_by_count`` maps it to the per-
    trajectory RMS list returned at those params.
    """

    def fake_fit(_env, trajectories, *_args, **_kwargs):
        theta = fit_thetas[len(trajectories)]
        return FitResult(names=["friction"],
                         samples=np.array([[theta]]),
                         log_probs=np.zeros(1),
                         jacobian=None,
                         noise_sigma=0.05,
                         prior_sigma=np.array([0.375]))

    def fake_rms(_env, trajectories, *_args, **_kwargs):
        return rms_by_count[len(trajectories)]

    monkeypatch.setattr(physical_sysid, "fit_params_rollout", fake_fit)
    monkeypatch.setattr(physical_sysid, "per_trajectory_rms", fake_rms)
    # min_explainable_rms (called by the trimmed fit) evaluates its
    # candidate grid through grid_seed's namespace.
    monkeypatch.setattr(grid_seed, "per_trajectory_rms", fake_rms)
    # The stub trajectories are plain strings; skip the data-derived
    # residual scaling (exercised by its own tests).
    monkeypatch.setattr(physical_sysid, "compute_residual_scaling",
                        lambda *_a, **_k: None)


def test_trimming_drops_unexplainable_and_refits(monkeypatch):
    """A high-RMS trajectory is dropped and the refit's theta is returned."""
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    trajs = ["chaotic", "clean"]  # stand-ins; the stubs only count them
    # Pooled fit (2 trajs) lands at 0.34; refit on the 1 survivor at 0.1.
    _patch_fit_and_rms(monkeypatch, {
        2: 0.34,
        1: 0.1
    }, {
        2: [0.7, 0.001],
        1: [0.001]
    })
    result, survivors, rms, _hull = fit_params_rollout_trimmed(
        None, trajs, specs, {"domino": ["x"]})
    assert survivors == ["clean"]
    assert rms == [0.7, 0.001]
    assert result.point_estimate["friction"] == 0.1


def test_trimming_keeps_all_when_explainable(monkeypatch):
    """No trajectory above threshold -> single fit, nothing dropped."""
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    trajs = ["a", "b"]
    _patch_fit_and_rms(monkeypatch, {2: 0.1}, {2: [0.001, 0.002]})
    result, survivors, rms, _hull = fit_params_rollout_trimmed(
        None, trajs, specs, {"domino": ["x"]})
    assert survivors == ["a", "b"]
    assert result.point_estimate["friction"] == 0.1
    assert rms == [0.001, 0.002]


def test_consistency_loop_drops_disagreeing_survivor(monkeypatch):
    """Explainable-but-wrong data loses to cleaner data on disagreement.

    Regression for the measured failure: a quiet chaotic shove was
    explainable at friction ~1.34 (best RMS 0.034) while the clean
    topple's best is at ~0.1 (best RMS 0.01); the joint fit compromised
    where the clean trajectory fits far worse than its own best. The
    consistency loop must drop the higher-best-RMS survivor (the shove)
    and refit on the clean one.
    """
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    shove, clean = "shove", "clean"
    state = {"fit_called": False}

    def fake_fit(_env, trajectories, *_args, **_kwargs):
        state["fit_called"] = True
        theta = {
            ("shove", "clean"): 1.34,
            ("clean", ): 0.1
        }[tuple(trajectories)]
        return FitResult(names=["friction"],
                         samples=np.array([[theta]]),
                         log_probs=np.zeros(1),
                         jacobian=None,
                         noise_sigma=0.05,
                         prior_sigma=np.array([0.375]))

    def fake_rms(_env, trajectories, *_args, **_kwargs):
        if not state["fit_called"]:
            # min_explainable_rms sweep: both look explainable.
            return [0.034, 0.010]
        # Consistency check at the joint fit (friction 1.34): the shove
        # fits at its best, the clean trajectory fits 6x worse.
        assert tuple(trajectories) == ("shove", "clean")
        return [0.034, 0.060]

    monkeypatch.setattr(physical_sysid, "fit_params_rollout", fake_fit)
    monkeypatch.setattr(physical_sysid, "per_trajectory_rms", fake_rms)
    monkeypatch.setattr(grid_seed, "per_trajectory_rms", fake_rms)
    monkeypatch.setattr(physical_sysid, "compute_residual_scaling",
                        lambda *_a, **_k: None)
    result, survivors, _rms, hull = fit_params_rollout_trimmed(
        None, [shove, clean], specs, {"domino": ["x"]})
    assert survivors == ["clean"]
    assert result.point_estimate["friction"] == 0.1
    # The disagreement survives as hull candidates: the pre-drop joint
    # fit and the dropped segment's own-best (grid-argmin) values (the
    # sweep fake returns identical RMS at every candidate, so the
    # argmin stays at the base candidate, friction=0.5).
    assert hull == [{"friction": 1.34}, {"friction": 0.5}]


def test_trimming_all_dropped_pins_result_at_inits(monkeypatch):
    """Nothing explainable -> empty survivors, result PINNED at inits.

    The fit must not run at all (nothing to learn from), and the point
    estimate must be the declared init so that applying it is a no-op
    even if a downstream probe on chaotic data falsely says
    "identified".
    """
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    trajs = ["chaos1", "chaos2"]
    # No fit_thetas entries: any call to the (stubbed) fit would KeyError.
    _patch_fit_and_rms(monkeypatch, {}, {2: [0.8, 0.6]})
    result, survivors, _rms, _hull = fit_params_rollout_trimmed(
        None, trajs, specs, {"domino": ["x"]})
    assert survivors == []
    assert result.point_estimate["friction"] == 0.5


def test_mcmc_samples_bypass_probe():
    """With a real chain, widths come from the samples, not the probe."""
    rng = np.random.default_rng(0)
    samples = np.column_stack([
        rng.normal(0.1, 0.01, size=200),  # contracted -> identified
        rng.normal(0.5, 0.375, size=200),  # prior-wide -> NOT identified
    ])
    result = FitResult(names=["friction", "restitution"],
                       samples=samples,
                       log_probs=np.zeros(200),
                       jacobian=None,
                       noise_sigma=0.05,
                       prior_sigma=np.array([0.375, 0.375]))
    report = identifiability_report(result)
    assert report["friction"]["verdict"] is Verdict.IDENTIFIED
    assert report["restitution"]["verdict"] is Verdict.NOT_IDENTIFIED


# ── Stale-override hygiene ────────────────────────────────────────


class _FakeStickyEnv:
    """Env stub with the real API's sticky per-param merge semantics."""

    def __init__(self, info):
        self._info = info
        self.overrides = {}

    def get_physical_param_info(self):
        """Return the per-param registry this stub was built with."""
        return self._info

    def apply_physical_param_overrides(self, params):
        """Merge overrides in, rejecting any undeclared param name."""
        unknown = set(params) - set(self._info)
        assert not unknown, unknown
        self.overrides.update(params)


_REGISTRY = {
    "lateral_friction": {
        "default": 0.5,
        "lo": 0.01,
        "hi": 2.0,
        "scale": "log",
        "description": "",
    },
    "rolling_friction": {
        "default": 0.006,
        "lo": 0.0,
        "hi": 0.1,
        "description": "",
    },
}


def test_pin_all_physical_params_reverts_undeclared():
    """A fit declaring a subset must not inherit an earlier fit's values."""
    env = _FakeStickyEnv(_REGISTRY)
    # Earlier fit's eval left lateral_friction at 2.0 on the shared env.
    rollout_env._pin_all_physical_params(env, {"lateral_friction": 2.0})
    assert env.overrides["lateral_friction"] == 2.0
    # A later fit declares only rolling_friction: lateral_friction must be
    # pinned back to the env default, not stay at 2.0.
    rollout_env._pin_all_physical_params(env, {"rolling_friction": 0.02})
    assert env.overrides["lateral_friction"] == 0.5
    assert env.overrides["rolling_friction"] == 0.02


def test_pin_all_physical_params_env_without_registry():
    """Envs without a registry just get the params applied."""

    class _Bare:
        applied = None

        def apply_physical_param_overrides(self, params):
            """Record whatever params get applied."""
            self.applied = dict(params)

    env = _Bare()
    rollout_env._pin_all_physical_params(env, {"k": 1.0})
    assert env.applied == {"k": 1.0}


def test_apply_identified_reverts_params_dropped_from_declaration():
    """Regression for run_20260707_112310: an intermediate artifact applied
    lateral_friction 1.9993, the final artifact declared only rolling_friction,
    and the planner silently kept 1.9993."""
    from predicators.approaches.agent_sim_learning_approach import \
        AgentSimLearningApproach
    approach = AgentSimLearningApproach.__new__(AgentSimLearningApproach)
    env = _FakeStickyEnv(_REGISTRY)
    approach._base_env = env
    approach._identified_physical_params = {}
    apply = approach._apply_identified_physical_params

    apply({"lateral_friction": 1.9993, "rolling_friction": 0.007})
    assert env.overrides["lateral_friction"] == 1.9993

    apply({"rolling_friction": 0.006})
    assert env.overrides["lateral_friction"] == 0.5  # reverted to default
    assert env.overrides["rolling_friction"] == 0.006
    # The tracked dict mirrors the current declaration exactly, so env
    # recreation re-applies only what the final artifact declared.
    assert approach._identified_physical_params == {"rolling_friction": 0.006}


# ── Fresh-env-per-rollout plumbing ────────────────────────────────


class _FreshRolloutEnv:
    """Env stub owning a real DIRECT PyBullet client (empty world), so velocity
    zeroing and client disposal exercise the real API."""

    def __init__(self, log):
        self._physics_client_id = p.connect(p.DIRECT)
        log.append(self)
        self.reset_to = None
        self.steps = 0

    def apply_physical_param_overrides(self, params):
        """No-op: this stub carries no physical params."""
        del params

    def _set_state(self, state):
        self.reset_to = state

    def step(self, action):
        """Advance the step counter and return a marker of progress."""
        del action
        self.steps += 1
        return f"post_{self.steps}"

    @property
    def connected(self):
        """Whether this stub's PyBullet client is still connected."""
        info = p.getConnectionInfo(self._physics_client_id)
        return bool(info["isConnected"])


def test_rollout_states_factory_builds_and_disposes_fresh_env():
    """A factory ``base_env`` gets one fresh env per rollout, disposed after.

    Regression for run_20260708_213258: rollouts on a reused env are
    nondeterministic (same-theta SSE alternated 0.15/78).
    """
    states, actions = _trajectory([0.0, 0.1, 0.2])
    built = []

    def factory():
        return _FreshRolloutEnv(built)

    out1 = rollout_env.rollout_states(factory, states[0], actions, {"k": 1.0})
    out2 = rollout_env.rollout_states(factory, states[0], actions, {"k": 1.0})
    assert out1 == ["post_1", "post_2"] == out2
    assert len(built) == 2  # one fresh env per rollout
    assert all(not env.connected for env in built)  # both disposed
    assert all(env.reset_to is states[0] for env in built)


def test_rollout_states_factory_disposes_on_rollout_error():
    """The fresh env is disconnected even when a step raises."""
    states, actions = _trajectory([0.0, 0.1, 0.2])
    built = []

    class _ExplodingEnv(_FreshRolloutEnv):

        def step(self, action):
            raise RuntimeError("mid-rollout failure")

    def factory():
        return _ExplodingEnv(built)

    try:
        rollout_env.rollout_states(factory, states[0], actions, {})
        assert False, "expected the step error to propagate"
    except RuntimeError:
        pass
    assert len(built) == 1
    assert not built[0].connected


def test_rollout_states_env_instance_is_not_disposed():
    """Passing an env instance keeps the caller-owned-env behavior."""
    states, actions = _trajectory([0.0, 0.1, 0.2])
    built = []
    env = _FreshRolloutEnv(built)
    out = rollout_env.rollout_states(env, states[0], actions, {"k": 1.0})
    assert out == ["post_1", "post_2"]
    assert env.connected  # caller-owned env stays alive
    p.disconnect(env._physics_client_id)  # pylint: disable=protected-access


# ── Residual scaling (angle wrap + per-feature normalization) ──────

_ANGULAR_TYPE = Type("spinner", ["x", "yaw"], angular_features=["yaw"])


def _angular_trajectory(xs, yaws):
    obj = Object("s0", _ANGULAR_TYPE)
    states = [
        State({obj: np.array([x, y], dtype=float)}) for x, y in zip(xs, yaws)
    ]
    actions = [
        Action(np.zeros(1, dtype=np.float32)) for _ in range(len(states) - 1)
    ]
    return states, actions


def test_residual_scaling_wraps_angular_features():
    """-pi vs +pi is the same orientation, so the residual must be ~0."""
    scaling = trajectory_prep.ResidualScaling(angular=frozenset({("spinner",
                                                                  "yaw")}),
                                              scales={
                                                  ("spinner", "yaw"):
                                                  float(np.pi)
                                              })
    near_zero = scaling.residual("spinner", "yaw", -np.pi + 1e-6, np.pi)
    assert abs(near_zero) < 1e-5
    # A genuine quarter-turn error survives the wrap: pi/2 / pi = 0.5.
    quarter = scaling.residual("spinner", "yaw", np.pi / 2, 0.0)
    assert abs(quarter - 0.5) < 1e-9


def test_residual_scaling_normalizes_linear_features():
    """Linear residuals are divided by the feature's scale entry."""
    scaling = trajectory_prep.ResidualScaling(angular=frozenset(),
                                              scales={("spinner", "x"): 0.5})
    assert abs(scaling.residual("spinner", "x", 0.6, 0.5) - 0.2) < 1e-12
    # Unknown features fall back to scale 1 (raw difference).
    assert abs(scaling.residual("spinner", "q", 0.6, 0.5) - 0.1) < 1e-12


def test_compute_residual_scaling_reads_type_metadata():
    """Angular features come from the Type; linear scales from the span."""
    xs = [0.0, 0.1, 0.4]  # span 0.4 > floor
    yaws = [0.0, 1.0, 3.0]
    traj = _angular_trajectory(xs, yaws)
    scaling = trajectory_prep.compute_residual_scaling(
        [traj], {"spinner": ["x", "yaw"]})
    assert scaling is not None
    assert ("spinner", "yaw") in scaling.angular
    assert abs(scaling.scales[("spinner", "yaw")] - np.pi) < 1e-12
    assert abs(scaling.scales[("spinner", "x")] - 0.4) < 1e-12


def test_compute_residual_scaling_floors_static_features(monkeypatch):
    """A feature that never moves gets the floor, not a ~0 divisor."""
    from predicators.settings import CFG
    monkeypatch.setattr(CFG, "code_sim_learning_rollout_feature_scale_floor",
                        0.05)
    traj = _angular_trajectory([0.2, 0.2, 0.2], [0.0, 0.0, 0.0])
    scaling = trajectory_prep.compute_residual_scaling([traj],
                                                       {"spinner": ["x"]})
    assert scaling is not None
    assert abs(scaling.scales[("spinner", "x")] - 0.05) < 1e-12


def test_compute_residual_scaling_disabled(monkeypatch):
    """The CFG kill switch restores the raw (legacy) objective."""
    from predicators.settings import CFG
    monkeypatch.setattr(CFG, "code_sim_learning_rollout_scale_residuals",
                        False)
    traj = _angular_trajectory([0.0, 0.1], [0.0, 0.1])
    assert trajectory_prep.compute_residual_scaling(
        [traj], {"spinner": ["x", "yaw"]}) is None


# ── Rest-point segmentation (multiple shooting) ────────────────────


def test_split_at_rest_points_separates_phases():
    """Two motion phases with a long rest between -> two segments."""
    xs = ([0.01 * i for i in range(11)] + [0.1] * 40 +
          [0.1 + 0.01 * i for i in range(11)] + [0.2] * 40)
    segments = trajectory_prep.split_at_rest_points(_trajectory(xs),
                                                    _RESIDUAL_FEATURES,
                                                    motion_tol=1e-3,
                                                    min_rest_steps=10,
                                                    margin=5)
    assert len(segments) == 2
    for states, actions in segments:
        assert len(states) == len(actions) + 1
    # Each segment starts at an at-rest observed state and keeps a
    # settle margin after its last motion.
    first_states, first_actions = segments[0]
    # Last active step index 9 -> keep 9 + 1 + margin 5 = 15 steps,
    # matching the truncate_settled_tail convention.
    assert len(first_actions) == 15
    assert float(first_states[0].get(Object("d0", _DOMINO_TYPE), "x")) == 0.0
    second_states, _ = segments[1]
    assert abs(
        float(second_states[0].get(Object("d0", _DOMINO_TYPE), "x")) -
        0.1) < 1e-12


def test_split_at_rest_points_keeps_short_pauses_together():
    """A pause shorter than min_rest_steps must not split the segment."""
    xs = ([0.01 * i for i in range(11)] + [0.1] * 4 +
          [0.1 + 0.01 * i for i in range(11)] + [0.2] * 30)
    segments = trajectory_prep.split_at_rest_points(_trajectory(xs),
                                                    _RESIDUAL_FEATURES,
                                                    motion_tol=1e-3,
                                                    min_rest_steps=10,
                                                    margin=5)
    assert len(segments) == 1


def test_split_at_rest_points_static_trajectory_yields_nothing():
    """No scored motion -> no segments (no parameter signal at all)."""
    segments = trajectory_prep.split_at_rest_points(_trajectory([0.5] * 50),
                                                    _RESIDUAL_FEATURES,
                                                    motion_tol=1e-3,
                                                    min_rest_steps=10,
                                                    margin=5)
    assert not segments


# ── Verdict hardening ──────────────────────────────────────────────


def test_identified_downgraded_on_single_segment():
    """A sharp posterior from one explainable segment is only weak."""

    def sse_fn(params):
        return 1e4 * (params["friction"] - 0.1)**2

    specs = [ParamSpec("friction", 0.1, lo=0.01, hi=2.0)]
    result = _single_sample_result(["friction"], [0.1], [0.75])
    report = identifiability_report(result, sse_fn, specs, num_explainable=1)
    assert report["friction"]["verdict"] is Verdict.WEAKLY_IDENTIFIED
    # With two segments the same posterior keeps the full verdict.
    report2 = identifiability_report(result, sse_fn, specs, num_explainable=2)
    assert report2["friction"]["verdict"] is Verdict.IDENTIFIED


def test_insensitive_param_overrides_probe_and_is_not_applied():
    """The sensitivity screen wins over apparent probe curvature."""

    def sse_fn(params):
        return 1e4 * (params["mass"] - 0.05)**2  # chaos-fake curvature

    specs = [ParamSpec("mass", 0.05, lo=0.005, hi=1.0)]
    result = _single_sample_result(["mass"], [0.03], [0.75])
    result.sensitivity = {
        "mass": {
            "sse_span": 0.001,
            "noise_floor": 0.01,
            "sensitive": False
        }
    }
    report = identifiability_report(result, sse_fn, specs, num_explainable=3)
    assert report["mass"]["verdict"] is Verdict.INSENSITIVE
    applied = select_trustworthy_params({"mass": 0.03}, {"mass": 0.05},
                                        ["mass"],
                                        report,
                                        anchors={"mass": 0.05})
    assert applied == {"mass": 0.05}


def test_untrusted_param_falls_back_to_anchor_not_declared_init():
    """Unsupported agent hypotheses must not reach the planner."""
    report = {"restitution": {"verdict": Verdict.UNKNOWN}}
    applied = select_trustworthy_params({"restitution": 0.13},
                                        {"restitution": 0.15}, ["restitution"],
                                        report,
                                        anchors={"restitution": 0.02})
    assert applied == {"restitution": 0.02}
    # Without an anchor the declared init remains the fallback.
    applied2 = select_trustworthy_params({"restitution": 0.13},
                                         {"restitution": 0.15},
                                         ["restitution"], report)
    assert applied2 == {"restitution": 0.15}


def test_annotated_weak_verdicts_still_apply():
    """Downgraded-but-supported verdicts keep applying the fitted value."""
    report = {
        "friction": {
            "verdict":
            Verdict.WEAKLY_IDENTIFIED,
            "note": ("sharp posterior, but only 1 explainable "
                     "segment(s) back it)")
        }
    }
    applied = select_trustworthy_params({"friction": 0.09}, {"friction": 0.2},
                                        ["friction"],
                                        report,
                                        anchors={"friction": 0.5})
    assert applied == {"friction": 0.09}


def test_inconsistent_param_holds_currently_applied_value():
    """INCONSISTENT holds the deployed value, not the new fit or anchor.

    Two mutually-incompatible confident fits cannot be arbitrated on
    this evidence; hopping between them churned the belief env for whole
    runs (run_20260721_205821 seed1).
    """
    report = {
        "restitution": {
            "verdict": Verdict.INCONSISTENT,
            "note": "0.5138 -> 0.3219 is 97.6 combined sigmas",
        }
    }
    applied = select_trustworthy_params({"restitution": 0.3219},
                                        {"restitution": 0.02}, ["restitution"],
                                        report,
                                        anchors={"restitution": 0.02},
                                        held={"restitution": 0.5138})
    assert applied == {"restitution": 0.5138}
    # Without a held value the anchor is the fallback, like the other
    # untrusted verdicts.
    applied2 = select_trustworthy_params({"restitution": 0.3219},
                                         {"restitution": 0.02},
                                         ["restitution"],
                                         report,
                                         anchors={"restitution": 0.02})
    assert applied2 == {"restitution": 0.02}


def test_trimming_uses_rms_cache(monkeypatch):
    """A second call with the same signature reuses the cached sweep."""
    specs = [ParamSpec("friction", 0.5, lo=0.01, hi=2)]
    trajs = [_trajectory([0.0, 0.1, 0.2]), _trajectory([0.0, 0.05, 0.1])]
    calls = {"sweep": 0}

    def fake_min_fits(_env, trajectories, *_args, **_kwargs):
        calls["sweep"] += 1
        return ([0.001] * len(trajectories), [{
            "friction": 0.1
        } for _ in trajectories])

    def fake_fit(_env, _trajectories, *_args, **_kwargs):
        return FitResult(names=["friction"],
                         samples=np.array([[0.1]]),
                         log_probs=np.zeros(1),
                         jacobian=None,
                         noise_sigma=0.05,
                         prior_sigma=np.array([0.375]))

    monkeypatch.setattr(physical_sysid, "min_explainable_fits", fake_min_fits)
    monkeypatch.setattr(physical_sysid, "fit_params_rollout", fake_fit)
    monkeypatch.setattr(physical_sysid, "per_trajectory_rms",
                        lambda *_a, **_k: [0.001, 0.001])
    cache = {}
    for _ in range(2):
        _result, survivors, _rms, _hull = fit_params_rollout_trimmed(
            None,
            trajs,
            specs, {"domino": ["x"]},
            anchors={"friction": 0.5},
            rms_cache=cache)
        assert len(survivors) == 2
    assert calls["sweep"] == 1


def test_map_at_box_bound_is_not_trusted():
    """A MAP pinned at its box edge must not be applied, however sharp.

    Regression for the replay of run_20260711_141026: mass fit to its
    1.0 hi bound with posterior_std 5e-11 ("identified" per the
    one-sided curvature probe, which reads the box wall as sharpness)
    while the true value was 0.1.
    """

    def sse_fn(params):
        return 1e6 * (1.0 - params["mass"])**2  # optimum on the box wall

    specs = [ParamSpec("mass", 0.1, lo=0.005, hi=1.0)]
    result = _single_sample_result(["mass"], [1.0], [0.75])
    report = identifiability_report(result, sse_fn, specs, num_explainable=5)
    assert report["mass"]["verdict"] is Verdict.AT_BOUND
    assert "hi" in report["mass"]["note"]
    applied = select_trustworthy_params({"mass": 1.0}, {"mass": 0.05},
                                        ["mass"],
                                        report,
                                        anchors={"mass": 0.1})
    assert applied == {"mass": 0.1}
    # An interior MAP with the same curvature stays identified.
    result_interior = _single_sample_result(["mass"], [0.5], [0.75])
    report2 = identifiability_report(result_interior,
                                     lambda p: 1e6 * (p["mass"] - 0.5)**2,
                                     specs,
                                     num_explainable=5)
    assert report2["mass"]["verdict"] is Verdict.IDENTIFIED


# ── Grid sweep: flat set, multi-pass, flat-edge refinement ─────────


def _run_sweep(monkeypatch, sse_fn, specs, anchors, **flags):
    """Run the grid seeding against a synthetic SSE landscape."""
    from predicators.settings import CFG
    for flag, value in flags.items():
        monkeypatch.setattr(CFG, flag, value)

    def fake_sse(_env, _trajs, params, *_args, **_kwargs):
        return sse_fn(params)

    monkeypatch.setattr(grid_seed, "compute_rollout_sse", fake_sse)
    seeded, info = grid_seed._grid_seed_physical_specs(None, ["traj"],
                                                       specs,
                                                       {"domino": ["x"]}, [],
                                                       [],
                                                       None,
                                                       anchors=anchors)
    return {s.name: s.init_value for s in seeded}, info


def _saturating_sse(params):
    """The measured run_20260711_224624 shape: lateral friction saturates
    above ~0.45 (topple reach), and low spinning friction buys a token
    1.6% SSE gain by compensating lateral's quantization error."""
    lat = params["lateral_friction"]
    spin = params["spinning_friction"]
    base = 60.0 if lat >= 0.45 else 60.0 + 300.0 * (0.45 - lat) / 0.45
    return base - (1.0 if spin < 0.05 else 0.0)


_HIGH_FRICTION_SPECS = [
    ParamSpec("lateral_friction", 0.1, lo=0.01, hi=2.0, scale="log"),
    ParamSpec("spinning_friction", 0.5, lo=0.01, hi=2.0, scale="log"),
]
_HIGH_FRICTION_ANCHORS = {"lateral_friction": 0.1, "spinning_friction": 0.5}


def test_grid_sweep_keeps_anchor_for_insignificant_gain(monkeypatch):
    """A compensating param must not move off its anchor for a ~1.6% gain.

    Regression for run_20260711_224624: with lateral_friction parked
    high, the argmin sweep dragged spinning_friction 0.5 -> 0.024 (true
    0.5) for a 1.6% SSE improvement - well inside the flat tolerance.
    """
    seeds, _info = _run_sweep(monkeypatch, _saturating_sse,
                              _HIGH_FRICTION_SPECS, _HIGH_FRICTION_ANCHORS)
    assert seeds["spinning_friction"] == 0.5


def test_grid_sweep_refines_flat_edge_toward_anchor(monkeypatch):
    """A true value mid-grid-gap becomes expressible via edge bisection.

    On run_20260711_224624 the 7-point log grid had no candidate between
    0.342 and 0.827, so a saturated landscape (flat above ~0.45) was
    always reported as the 0.827 grid point against a true 0.5. The
    flat-edge bisection must land near the saturation onset instead,
    and the flat interval must report the full data-equivalent range.
    """
    seeds, info = _run_sweep(monkeypatch, _saturating_sse,
                             _HIGH_FRICTION_SPECS, _HIGH_FRICTION_ANCHORS)
    assert 0.44 <= seeds["lateral_friction"] <= 0.54
    lo, hi = info["lateral_friction"]["flat_interval"]
    assert lo <= 0.46
    assert hi == 2.0
    # Without refinement the seed is quantized to the grid point.
    seeds_coarse, _info2 = _run_sweep(
        monkeypatch,
        _saturating_sse,
        _HIGH_FRICTION_SPECS,
        _HIGH_FRICTION_ANCHORS,
        code_sim_learning_rollout_grid_refine_evals=0)
    assert abs(seeds_coarse["lateral_friction"] - 0.827) < 1e-2


def test_grid_sweep_second_pass_relaxes_early_param(monkeypatch):
    """A param swept before its neighbor moved is re-examined next pass."""

    def coupled_sse(params):
        a, b = params["a"], params["b"]
        return 50.0 * (a * b - 6.0)**2 + 50.0 * (b - 3.0)**2

    specs = [
        ParamSpec("a", 1.0, lo=0.0, hi=4.0),
        ParamSpec("b", 1.0, lo=0.0, hi=4.0),
    ]
    anchors = {"a": 1.0, "b": 1.0}
    # Pass 1 greedily sends a to the 4.0 grid edge (with b still at 1);
    # after b moves to 2, pass 2 must relax a back to 3.
    seeds, _info = _run_sweep(monkeypatch,
                              coupled_sse,
                              specs,
                              anchors,
                              code_sim_learning_rollout_grid_seed_points=5,
                              code_sim_learning_rollout_grid_refine_evals=0)
    assert seeds == {"a": 3.0, "b": 2.0}
    seeds_one_pass, _info2 = _run_sweep(
        monkeypatch,
        coupled_sse,
        specs,
        anchors,
        code_sim_learning_rollout_grid_seed_points=5,
        code_sim_learning_rollout_grid_refine_evals=0,
        code_sim_learning_rollout_grid_sweep_passes=1)
    assert seeds_one_pass == {"a": 4.0, "b": 2.0}


def test_grid_sweep_sharp_basin_not_dragged_to_anchor(monkeypatch):
    """Refinement must respect a sharp basin's own extent.

    The relative flat tolerance shrinks with the best SSE, so on clean
    data (deep narrow basin, e.g. the low-friction-arm landscape) the
    anchor-ward bisection stops at the basin edge instead of dragging
    the seed toward the anchor.
    """

    def basin_sse(params):
        v = params["lateral_friction"]
        return 0.05 if 0.12 <= v <= 0.16 else 80.0

    specs = [ParamSpec("lateral_friction", 0.5, lo=0.01, hi=2.0, scale="log")]
    seeds, info = _run_sweep(monkeypatch, basin_sse, specs,
                             {"lateral_friction": 0.5})
    assert 0.12 <= seeds["lateral_friction"] <= 0.165
    lo, hi = info["lateral_friction"]["flat_interval"]
    assert 0.12 <= lo and hi <= 0.165


def test_flat_interval_drives_verdict_over_local_curvature():
    """The landscape's data-equivalent interval IS the posterior width.

    Regression for run_20260722_123949: the local curvature probe
    stamped every parameter "identified" while its own report noted the
    fitted value was merely "the edge of this interval ... not a unique
    optimum". A plateau-wide interval must read NOT identified no
    matter how sharp the local curvature at the chosen edge is - and
    the probe must not even be consulted (or spend rollouts) for a
    swept parameter.
    """
    specs = [ParamSpec("friction", 0.6, lo=0.01, hi=2.0, scale="log")]
    result = _single_sample_result(["friction"], [0.6], [0.75])
    result.sensitivity = {
        "friction": {
            "sse_span": 100.0,
            "noise_floor": 0.0,
            "sensitive": True,
            "flat_interval": (0.5, 2.0),
        }
    }
    probe_calls = {"n": 0}

    def sharp_sse(params):
        probe_calls["n"] += 1
        return 1e3 * (np.log(params["friction"]) - np.log(0.6))**2

    report = identifiability_report(result,
                                    sharp_sse,
                                    specs,
                                    num_explainable=3)
    entry = report["friction"]
    # log half-width ln(2.0/0.5)/2 ~= 0.693 vs prior 0.75 -> ~0.92.
    assert entry["verdict"] is Verdict.NOT_IDENTIFIED
    assert entry["flat_interval"] == (0.5, 2.0)
    assert probe_calls["n"] == 0
    text = format_identifiability(report)
    assert "data-equivalent over [0.5, 2]" in text
    assert "not a unique optimum" in text
    # A degenerate interval means the sweep resolved the value: width 0,
    # contraction 0, identified - and nothing rendered.
    result.sensitivity["friction"]["flat_interval"] = (0.6, 0.6)
    report2 = identifiability_report(result,
                                     sharp_sse,
                                     specs,
                                     num_explainable=3)
    assert report2["friction"]["verdict"] is Verdict.IDENTIFIED
    assert probe_calls["n"] == 0
    assert "data-equivalent" not in format_identifiability(report2)


# ── Anchor-ablation backward elimination ──────────────────────────


class _CompensatingEnv:
    """DIRECT-client rollout env whose per-step motion mixes two params.

    ``x`` advances by ``0.01*gain_a + 0.5*(gain_b - 0.02)`` per step, so
    the observed motion constrains only the combination: a co-adapted
    (gain_a=1.0, gain_b=0.04) reproduces data generated at the true
    (gain_a=2.0, gain_b=0.02) exactly - the compensation ridge the
    ablation must resolve toward the standing belief.
    """

    def __init__(self):
        self._physics_client_id = p.connect(p.DIRECT)
        self._params = {"gain_a": 1.0, "gain_b": 0.02}
        self._x = 0.0

    def get_physical_param_info(self):
        """Registry with a wide log param and a tight linear param."""
        return {
            "gain_a": {
                "default": 1.0,
                "lo": 0.1,
                "hi": 10.0,
                "scale": "log",
                "description": "",
            },
            "gain_b": {
                "default": 0.02,
                "lo": 0.0,
                "hi": 0.9,
                "description": "",
            },
        }

    def apply_physical_param_overrides(self, params):
        """Sticky per-param merge, like the real env API."""
        self._params.update(params)

    def _set_state(self, state):
        self._x = float(state.get(Object("d0", _DOMINO_TYPE), "x"))

    def step(self, action):
        """Advance ``x`` by the params' combined per-step motion."""
        del action
        self._x += (0.01 * self._params["gain_a"] + 0.5 *
                    (self._params["gain_b"] - 0.02))
        domino = Object("d0", _DOMINO_TYPE)
        robot = Object("r0", _ROBOT_TYPE)
        return State({
            domino: np.array([self._x], dtype=float),
            robot: np.array([0.0], dtype=float),
        })


def _ablation_fixtures():
    import dataclasses

    from predicators.code_sim_learning.config import SysIdConfig
    from predicators.code_sim_learning.fit_space import prior_widths
    spec_a = ParamSpec("gain_a", 1.0, lo=0.1, hi=10.0, scale="log")
    spec_b = ParamSpec("gain_b", 0.02, lo=0.0, hi=0.9)
    anchors = {"gain_a": 1.0, "gain_b": 0.02}
    prior_sigma = prior_widths([spec_a, spec_b], 0.75)
    config = dataclasses.replace(SysIdConfig.from_cfg(),
                                 anchor_ablation=True,
                                 grid_flat_frac=0.05)
    # Observed data generated at the TRUE params (gain_a=2, gain_b=0.02):
    # dx = 0.02 per step.
    traj = _trajectory([0.02 * t for t in range(11)])
    return spec_a, spec_b, anchors, prior_sigma, config, traj


def test_anchor_ablation_reverts_compensatory_param():
    """A co-adapted MAP on the compensation ridge is resolved to the anchor-
    consistent basin: the tight-prior param reverts to its anchor and the wide-
    prior param refits to its true value."""
    spec_a, spec_b, anchors, prior_sigma, config, traj = _ablation_fixtures()
    env = _CompensatingEnv()
    # Co-adapted MAP: gain_a stuck at its anchor, gain_b compensating.
    result = FitResult(names=["gain_a", "gain_b"],
                       samples=np.array([[1.0, 0.04]]),
                       log_probs=np.zeros(1),
                       noise_sigma=0.05,
                       prior_sigma=prior_sigma,
                       scales=["log", "linear"])
    out = physical_sysid._anchor_backward_elimination(
        env, [traj], [spec_a, spec_b], [], _RESIDUAL_FEATURES, [], None, None,
        anchors, 0.05, 0.75, result, 0.01, config)
    assert out.anchor_ablation is not None
    assert set(out.anchor_ablation) == {"gain_b"}
    point = out.point_estimate
    assert point["gain_b"] == 0.02
    assert abs(point["gain_a"] - 2.0) < 0.15
    entry = out.anchor_ablation["gain_b"]
    assert entry["anchor"] == 0.02
    assert entry["sse_pinned"] <= entry["sse_map"] + entry["tol"]
    # The verdict pipeline renders it "anchored" and applies the anchor.
    report = identifiability_report(out)
    assert report["gain_b"]["verdict"] is Verdict.ANCHORED
    applied = select_trustworthy_params(point, {
        "gain_a": 1.0,
        "gain_b": 0.04
    }, ["gain_a", "gain_b"], report, anchors)
    assert applied["gain_b"] == 0.02
    text = format_identifiability(report)
    assert "anchor ablation" in text


def test_anchor_ablation_cheap_pretest_skips_lm_refit(monkeypatch):
    """A tiny drift that is data-equivalent when pinned alone must revert via
    the one-eval cheap pre-test, without any LM refit.

    The common case on real runs (run_20260722_123949 seed2: spinning
    0.4993 -> 0.5, restitution 0.02021 -> 0.02): the LM refits were the
    dominant ablation cost and are unnecessary when pinning the param
    with everything else unchanged already explains the data.
    """
    spec_a, spec_b, _anchors, prior_sigma, config, traj = _ablation_fixtures()
    env = _CompensatingEnv()
    # Standing belief already carries gain_a's true value, so gain_b is
    # the only movable param (a genuinely-moved co-declared param would
    # legitimately get - and fail - its own pinned refit attempt).
    anchors = {"gain_a": 2.0, "gain_b": 0.02}
    refits = {"n": 0}
    real_fit = physical_sysid.fit_map_lm_rollout

    def counting_fit(*args, **kwargs):
        refits["n"] += 1
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(physical_sysid, "fit_map_lm_rollout", counting_fit)
    # gain_b drifted slightly (0.021 vs anchor 0.02 - a per-step x
    # error of 5e-4, well inside the flat tolerance).
    result = FitResult(names=["gain_a", "gain_b"],
                       samples=np.array([[2.0, 0.021]]),
                       log_probs=np.zeros(1),
                       noise_sigma=0.05,
                       prior_sigma=prior_sigma,
                       scales=["log", "linear"])
    out = physical_sysid._anchor_backward_elimination(
        env, [traj], [spec_a, spec_b], [], _RESIDUAL_FEATURES, [], None, None,
        anchors, 0.05, 0.75, result, 0.01, config)
    assert out.anchor_ablation is not None
    assert set(out.anchor_ablation) == {"gain_b"}
    assert out.point_estimate["gain_b"] == 0.02
    assert abs(out.point_estimate["gain_a"] - 2.0) < 1e-9
    assert refits["n"] == 0


def test_anchor_ablation_keeps_genuinely_moved_param():
    """A MAP whose single move the data requires (and whose alternative is
    farther from the belief) is returned unchanged."""
    spec_a, spec_b, anchors, _prior_sigma, config, traj = _ablation_fixtures()
    env = _CompensatingEnv()
    from predicators.code_sim_learning.fit_space import prior_widths
    result = FitResult(names=["gain_a", "gain_b"],
                       samples=np.array([[2.0, 0.02]]),
                       log_probs=np.zeros(1),
                       noise_sigma=0.05,
                       prior_sigma=prior_widths([spec_a, spec_b], 0.75),
                       scales=["log", "linear"])
    out = physical_sysid._anchor_backward_elimination(
        env, [traj], [spec_a, spec_b], [], _RESIDUAL_FEATURES, [], None, None,
        anchors, 0.05, 0.75, result, 0.01, config)
    assert out is result
    assert out.anchor_ablation is None
    assert out.point_estimate == {"gain_a": 2.0, "gain_b": 0.02}


def test_huberize_caps_outliers():
    """Squared huberized residual follows the Huber loss; 0 disables."""
    from predicators.code_sim_learning.rollout_objective import _huberize
    assert _huberize(0.5, 1.0) == 0.5
    assert _huberize(-0.5, 1.0) == -0.5
    # Outside delta: r'^2 == 2*delta*|r| - delta^2 (linear growth).
    big = _huberize(10.0, 1.0)
    assert big**2 == pytest.approx(2 * 1.0 * 10.0 - 1.0)
    assert _huberize(-10.0, 1.0) == -big
    # delta <= 0 disables the cap.
    assert _huberize(10.0, 0.0) == 10.0


def test_rollout_sse_summary_residuals(monkeypatch):
    """Endpoint + onset summary terms are appended with the flag weight.

    The observed domino starts moving at step 1 and ends at x=0.3; the
    (faked) rollout never moves it. With summary_weight w: per-step
    residuals (huber-capped), plus w * endpoint residual squared per
    scored feature, plus w * ((sim_onset - obs_onset)/horizon)^2 with
    the never-moving sim onset clamped to the horizon.
    """
    from predicators.settings import CFG
    monkeypatch.setattr(CFG, "code_sim_learning_rollout_huber_delta", 0.0)
    states, actions = _trajectory([0.0, 0.1, 0.2, 0.3])
    sim_static = [states[0]] * len(actions)  # sim: domino never moves

    monkeypatch.setattr(rollout_objective, "rollout_states",
                        lambda *_a, **_k: sim_static)

    def sse(weight):
        monkeypatch.setattr(CFG, "code_sim_learning_rollout_summary_weight",
                            weight)
        return rollout_objective.compute_rollout_sse(None, [(states, actions)],
                                                     {"friction": 0.5},
                                                     _RESIDUAL_FEATURES,
                                                     ["friction"])

    base = sse(0.0)
    # Per-step residuals: sim x stays 0.0 vs obs 0.1, 0.2, 0.3.
    assert base == pytest.approx(0.01 + 0.04 + 0.09)
    with_summary = sse(4.0)
    # Endpoint: 4 * 0.3^2. Onset: obs moves at state 1, sim never
    # (clamped to horizon 3): 4 * ((3 - 1) / 3)^2.
    assert with_summary == pytest.approx(base + 4 * 0.09 + 4 * (2 / 3)**2)


def test_rollout_sse_huber_caps_spike(monkeypatch):
    """A qualitatively-diverged replay contributes linearly, not squared."""
    from predicators.settings import CFG
    monkeypatch.setattr(CFG, "code_sim_learning_rollout_summary_weight", 0.0)
    states, actions = _trajectory([0.0, 0.0, 100.0])
    sim_static = [states[0]] * len(actions)
    monkeypatch.setattr(rollout_objective, "rollout_states",
                        lambda *_a, **_k: sim_static)

    def sse(delta):
        monkeypatch.setattr(CFG, "code_sim_learning_rollout_huber_delta",
                            delta)
        return rollout_objective.compute_rollout_sse(None, [(states, actions)],
                                                     {"friction": 0.5},
                                                     _RESIDUAL_FEATURES,
                                                     ["friction"])

    assert sse(0.0) == pytest.approx(100.0**2)
    assert sse(1.0) == pytest.approx(2 * 100.0 - 1.0)
