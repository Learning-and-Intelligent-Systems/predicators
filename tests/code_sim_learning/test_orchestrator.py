"""Tests for the shared rollout-sysID orchestrator.

One flow behind both the approach's final-commit fit and the agent's
``sim.fit`` tool: end-to-end fit + report + trust selection, whole-fit
memoization per (artifact, data) signature, and caller-local report
adjustment on cache hits.
"""
# pylint: disable=protected-access

import numpy as np
import pybullet as p

import predicators.approaches  # noqa: F401  # pylint: disable=unused-import
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.identifiability import Verdict
from predicators.code_sim_learning.orchestrator import run_rollout_sysid
from predicators.code_sim_learning.rollout_env import num_rollouts_run
from predicators.structs import Action, Object, State, Type

_DOMINO_TYPE = Type("domino", ["x"])
_RESIDUAL_FEATURES = {"domino": ["x"]}
_DOMINO = Object("d0", _DOMINO_TYPE)


class _GainEnv:
    """DIRECT-client rollout env: ``x`` advances by ``0.01 * gain``/step."""

    def __init__(self):
        self._physics_client_id = p.connect(p.DIRECT)
        self._params = {"gain": 1.0}
        self._x = 0.0

    def get_physical_param_info(self):
        """Registry: one wide log-scale parameter, baseline 1.0."""
        return {
            "gain": {
                "default": 1.0,
                "lo": 0.1,
                "hi": 10.0,
                "scale": "log",
                "description": "",
            },
        }

    def apply_physical_param_overrides(self, params):
        """Sticky per-param merge, like the real env API."""
        self._params.update(params)

    def _set_state(self, state):
        self._x = float(state.get(_DOMINO, "x"))

    def step(self, action):
        """Advance ``x`` by the current gain's per-step motion."""
        del action
        self._x += 0.01 * self._params["gain"]
        return State({_DOMINO: np.array([self._x], dtype=float)})


def _trajectory(num_steps=10, gain=2.0):
    """Observed data generated at the true ``gain``."""
    states = [
        State({_DOMINO: np.array([0.01 * gain * t], dtype=float)})
        for t in range(num_steps + 1)
    ]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(num_steps)]
    return states, actions


def test_run_rollout_sysid_fit_cache_and_report_isolation():
    """The fit core is memoized per (artifact, data) key; adjusters and trust
    selection stay caller-local on deep-copied reports."""
    env = _GainEnv()
    spec = ParamSpec("gain", 1.0, lo=0.1, hi=10.0, scale="log")
    traj = _trajectory()
    anchors = {"gain": 1.0}
    fit_cache = {}

    outcome = run_rollout_sysid(env, [traj], [spec],
                                _RESIDUAL_FEATURES,
                                anchors=anchors,
                                rms_cache={},
                                fit_cache=fit_cache,
                                fit_cache_key="vers_001")
    assert not outcome.from_cache
    assert outcome.num_survivors == 1
    assert outcome.report["gain"]["verdict"].applies_fitted
    assert abs(outcome.applied["gain"] - 2.0) < 0.2
    assert outcome.post_sse < outcome.pre_sse
    assert len(fit_cache) == 1

    # Identical call: zero new rollouts, same applied values.
    n_before = num_rollouts_run()
    outcome2 = run_rollout_sysid(env, [traj], [spec],
                                 _RESIDUAL_FEATURES,
                                 anchors=anchors,
                                 rms_cache={},
                                 fit_cache=fit_cache,
                                 fit_cache_key="vers_001")
    assert outcome2.from_cache
    assert num_rollouts_run() == n_before
    assert outcome2.applied == outcome.applied

    # A cached call with a caller-local adjuster (the cross-cycle
    # INCONSISTENT demotion) changes ITS selection only - the cached
    # report and earlier outcomes are untouched (deep copy).
    def demote(_result, report, sse_fn):
        # The fit's own SSE probe rides along even on cache hits (the
        # cross-cycle arbitration needs it); sanity-check it is live.
        assert sse_fn is not None and np.isfinite(sse_fn({"gain": 1.0}))
        report["gain"]["verdict"] = Verdict.INCONSISTENT
        report["gain"]["note"] = "test demotion"

    outcome3 = run_rollout_sysid(env, [traj], [spec],
                                 _RESIDUAL_FEATURES,
                                 anchors=anchors,
                                 rms_cache={},
                                 fit_cache=fit_cache,
                                 fit_cache_key="vers_001",
                                 report_adjuster=demote,
                                 held={"gain": 1.5})
    assert outcome3.from_cache
    assert outcome3.applied == {"gain": 1.5}
    assert outcome.report["gain"]["verdict"].applies_fitted

    # A different artifact key recomputes.
    outcome4 = run_rollout_sysid(env, [traj], [spec],
                                 _RESIDUAL_FEATURES,
                                 anchors=anchors,
                                 rms_cache={},
                                 fit_cache=fit_cache,
                                 fit_cache_key="vers_002")
    assert not outcome4.from_cache
    assert len(fit_cache) == 2
