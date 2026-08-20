"""Tests for ``sim.residuals(rollout=True)`` - the open-loop report.

Regression for run_20260728_111805: the synthesis agent declined
PHYSICAL_PARAMS because teacher-forced residuals structurally cannot see
compounding divergence (they predict each step from the recorded state),
so "the base sim replicates the data" looked true at a friction that was
wrong by 5x open-loop. The rollout mode replays trajectories free-running;
``sweep_params`` (a list, or "all") opts into sweeping env-registry
physical parameters, surfacing "this data is explained Nx better at a
different value" before the declaration decision, and ``phys_params``
scores a single hypothesized point.
"""

from pathlib import Path
from typing import Callable, Optional, cast

import numpy as np
import pybullet as p

from predicators import utils
from predicators.agent_sdk.synthesis_backend import SynthesisBackend
from predicators.agent_sdk.tools.synthesis import SynthesisToolkit, \
    create_synthesis_tools
from predicators.structs import Action, LowLevelTrajectory, Object, State, Type

_BALL_TYPE = Type("ball", ["x"])
_TRUE_FRICTION = 1.0

_NOOP_SIMULATOR = """
def _noop_rule(state, updates, params):
    _ = params["dummy_k"]
    return updates

RESIDUAL_RULES = [_noop_rule]
PARAM_SPECS = [ParamSpec("dummy_k", 0.1, lo=0.0, hi=1.0)]
RESIDUAL_FEATURES = {"ball": []}
"""


class _LinearEnv:
    """Env stub whose only dynamics is ``x += friction`` per step.

    ``bounce`` is registered but inert, so its sweep must come out flat.
    Owns a real DIRECT client so velocity zeroing and disposal exercise
    the real PyBullet API.
    """

    def __init__(self) -> None:
        self._physics_client_id = p.connect(p.DIRECT)
        self._params = {"friction": 0.1, "bounce": 0.0}
        self._state: State = None  # type: ignore[assignment]

    def get_physical_param_info(self):
        """Registry: one effective log-scale param, one inert one."""
        return {
            "friction": {
                "default": 0.1,
                "lo": 0.01,
                "hi": 2.0,
                "scale": "log",
                "description": "per-step drift",
            },
            "bounce": {
                "default": 0.0,
                "lo": 0.0,
                "hi": 1.0,
                "description": "does nothing",
            },
        }

    def apply_physical_param_overrides(self, params):
        """Store the pinned parameter values."""
        self._params.update(params)

    def _set_state(self, state: State) -> None:
        self._state = state.copy()

    def step(self, action: Action) -> State:
        """Advance every ball by the current friction value."""
        del action
        nxt = self._state.copy()
        for obj in nxt:
            nxt.set(obj, "x", nxt.get(obj, "x") + self._params["friction"])
        self._state = nxt
        return nxt.copy()


class _FakeApproach:
    """Duck-typed stand-in exposing what the residuals runner uses."""

    def __init__(self, trajectories) -> None:
        self._fit_trajectories = trajectories
        self._base_env = _LinearEnv()

    def _rollout_fit_trajectories(self,
                                  residual_features=None,
                                  traj_idxs=None):
        # Real signature/semantics minus truncation config coupling.
        del residual_features, traj_idxs
        return [(list(t.states), list(t.actions))
                for t in self._fit_trajectories]

    def _get_rollout_fit_env(self):
        return _LinearEnv


def _observed_trajectory(num_steps: int = 6) -> LowLevelTrajectory:
    """A ball drifting at the TRUE friction, 1.0 per step."""
    ball = Object("ball0", _BALL_TYPE)
    states = [
        State({ball: np.array([_TRUE_FRICTION * t])})
        for t in range(num_steps + 1)
    ]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(num_steps)]
    return LowLevelTrajectory(states, actions)


def _make_toolkit(
        tmp_path: Path,
        budget_check: Optional[Callable[[], None]] = None) -> SynthesisToolkit:
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text(_NOOP_SIMULATOR, encoding="utf-8")
    return create_synthesis_tools(exec_ns={},
                                  base_pred_triples=[],
                                  inferred_residual_features={},
                                  simulator_file=str(sim_file),
                                  versions_dir=str(tmp_path / "versions"),
                                  approach=cast(
                                      SynthesisBackend,
                                      _FakeApproach([_observed_trajectory()])),
                                  budget_check=budget_check)


def test_rollout_residuals_flags_effective_param(tmp_path) -> None:
    """Sweep flags the mis-set effective param; the inert one reads flat.

    The artifact's RESIDUAL_FEATURES is empty (the exact 111805 shape),
    so the report's scope must come from observed motion, not from the
    declared rule scope.
    """
    utils.reset_config({"seed": 0})
    toolkit = _make_toolkit(tmp_path)
    report = toolkit.residuals_runner(rollout=True,
                                      sweep_num_points=6,
                                      sweep_params="all")
    assert "OPEN-LOOP ROLLOUT residual report" in report
    assert "ball: x" in report
    # friction: baseline 0.1 vs truth 1.0 - geomspace(0.01, 2, 6) has a
    # candidate at ~0.69 whose divergence is ~3x smaller per step, so
    # the SSE ratio clears the consistency bar.
    assert "friction (log scale, baseline 0.1)" in report
    assert "strong evidence FOR declaring" in report
    # bounce: inert, all candidates score identically.
    assert "flat across the range" in report
    assert "cannot constrain it" in report


def test_rollout_residuals_default_reports_without_sweep(tmp_path) -> None:
    """Without sweep_params the report is baseline-only plus the nudge.

    The sweep is opt-in (it costs one fresh-env rollout per candidate
    per segment), so the bare rollout report must still hand the agent
    the registry and point at sweep_params/phys_params.
    """
    utils.reset_config({"seed": 0})
    toolkit = _make_toolkit(tmp_path)
    report = toolkit.residuals_runner(rollout=True)
    assert "OPEN-LOOP ROLLOUT residual report" in report
    assert "Rollout SSE at current baselines" in report
    assert "No parameter probing requested" in report
    # The registry listing with baselines and boxes.
    assert "friction: baseline 0.1, box [0.01, 2] (log scale)" in report
    assert "bounce: baseline 0, box [0, 1] (linear scale)" in report
    # The nudge toward the PHYSICAL_PARAMS decision.
    assert "sweep_params" in report and "phys_params" in report
    assert "BEFORE deciding the PHYSICAL_PARAMS declaration" in report
    # And no sweep actually ran.
    assert "Physical-parameter sweep" not in report


def test_rollout_residuals_phys_params_point(tmp_path) -> None:
    """phys_params scores one point and reports the ratio vs baseline.

    At the true friction the replay reproduces the data, so the point
    must come back materially better explained than the 0.1 baseline.
    """
    utils.reset_config({"seed": 0})
    toolkit = _make_toolkit(tmp_path)
    report = toolkit.residuals_runner(rollout=True,
                                      phys_params={"friction": 1.0})
    assert "SSE at phys_params (friction=1)" in report
    assert "better explained at this point" in report
    assert "strong evidence FOR declaring" in report


def test_rollout_residuals_phys_params_worse_point(tmp_path) -> None:
    """A hypothesized point the baseline beats reads as evidence against.

    Data drifting at the 0.1 baseline is explained perfectly there, so
    scoring friction=1.0 must come back WORSE, not "cannot distinguish".
    """
    utils.reset_config({"seed": 0})
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text(_NOOP_SIMULATOR, encoding="utf-8")
    ball = Object("ball0", _BALL_TYPE)
    states = [State({ball: np.array([0.1 * t])}) for t in range(7)]
    actions = [Action(np.zeros(1, dtype=np.float32)) for _ in range(6)]
    toolkit = create_synthesis_tools(
        exec_ns={},
        base_pred_triples=[],
        inferred_residual_features={},
        simulator_file=str(sim_file),
        versions_dir=str(tmp_path / "versions"),
        approach=cast(SynthesisBackend,
                      _FakeApproach([LowLevelTrajectory(states, actions)])))
    worse = toolkit.residuals_runner(rollout=True,
                                     phys_params={"friction": 1.0})
    assert "WORSE explained at this point" in worse
    assert "evidence against this hypothesis" in worse


def test_rollout_residuals_param_arg_validation(tmp_path) -> None:
    """Bad sweep_params/phys_params inputs error with the registry."""
    utils.reset_config({"seed": 0})
    toolkit = _make_toolkit(tmp_path)
    out = toolkit.residuals_runner(rollout=True, sweep_params=["nope"])
    assert "not in the env's physical-param registry" in out
    assert "'bounce'" in out and "'friction'" in out
    out = toolkit.residuals_runner(rollout=True, sweep_params="everything")
    assert "must be a list of registry names or the string 'all'" in out
    out = toolkit.residuals_runner(rollout=True, phys_params={"nope": 1.0})
    assert "phys_params ['nope'] not in the env's physical-param " \
        "registry" in out
    out = toolkit.residuals_runner(rollout=True,
                                   phys_params={"friction": float("nan")})
    assert "must be finite numbers" in out
    out = toolkit.residuals_runner(rollout=True,
                                   sweep_params=["friction"],
                                   phys_params={"friction": 1.0})
    assert "mutually exclusive" in out
    out = toolkit.residuals_runner(sweep_params=["friction"])
    assert "apply to the open-loop report only" in out


def test_rollout_residuals_budget_stop_salvages_partial(tmp_path) -> None:
    """A budget stop mid-sweep returns the completed params, not nothing.

    The checkpoint fires between candidate rollouts; 'bounce' (first in
    sorted order) completes within the allowance, 'friction' is reported
    as unswept.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.belief_probe import ProbeBudgetExceeded
    utils.reset_config({"seed": 0})
    calls = {"n": 0}

    def _budget_check() -> None:
        calls["n"] += 1
        if calls["n"] > 6:  # bounce's 6 candidates pass, then stop.
            raise ProbeBudgetExceeded("time limit reached")

    toolkit = _make_toolkit(tmp_path, budget_check=_budget_check)
    report = toolkit.residuals_runner(rollout=True, sweep_params="all")
    assert "bounce (linear scale, baseline 0)" in report
    assert "SWEEP STOPPED EARLY after 1/2 parameters" in report
    assert "still unswept: ['friction']" in report


def test_rollout_residuals_contains_rule_crashes(tmp_path) -> None:
    """A crashing RESIDUAL_RULES function comes back as a report.

    Not as a raw traceback through the tool: rules run on every rolled-
    out step, so an agent's buggy rule is a routine input here.
    """
    utils.reset_config({"seed": 0})
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text(_NOOP_SIMULATOR.replace('params["dummy_k"]',
                                                'params["missing"]'),
                        encoding="utf-8")
    toolkit = create_synthesis_tools(
        exec_ns={},
        base_pred_triples=[],
        inferred_residual_features={},
        simulator_file=str(sim_file),
        versions_dir=str(tmp_path / "versions"),
        approach=cast(SynthesisBackend,
                      _FakeApproach([_observed_trajectory()])))
    out = toolkit.residuals_runner(rollout=True)
    assert "Error: open-loop rollout scoring failed" in out
    assert "RESIDUAL_RULES bug" in out


def test_rollout_residuals_requires_full_trajectories(tmp_path) -> None:
    """Without complete trajectories the report explains itself.

    (Rather than crashing: the rollout mode needs full (states, actions)
    sequences, not isolated transitions.)
    """
    utils.reset_config({"seed": 0})
    sim_file = tmp_path / "simulator.py"
    sim_file.write_text(_NOOP_SIMULATOR, encoding="utf-8")
    toolkit = create_synthesis_tools(exec_ns={},
                                     base_pred_triples=[],
                                     inferred_residual_features={},
                                     simulator_file=str(sim_file),
                                     versions_dir=str(tmp_path / "versions"),
                                     approach=cast(SynthesisBackend,
                                                   _FakeApproach([])))
    out = toolkit.residuals_runner(rollout=True)
    assert "needs full trajectories" in out
