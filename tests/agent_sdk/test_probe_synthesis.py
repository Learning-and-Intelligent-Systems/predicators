"""Tests for the synthesis-phase behavior of the explore_python probe.

Covers the ``ctx.probe_option_model_provider`` hook: model resolution
(candidate provider vs. the solve-phase ``ctx.option_model`` fallback),
the explicit-``task_idx`` guard on ``reset`` during synthesis, the
provider's load/cache glue, and the phase-appropriate tool description.
"""
# pylint: disable=protected-access
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe, \
    build_probe_namespace
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.fit_space import FitResult
from predicators.option_model import _OptionModelBase
from predicators.structs import Object, State, Task, Type


def _tiny_task() -> Task:
    obj_type = Type("thing", ["x"])
    obj = Object("thing0", obj_type)
    return Task(State({obj: np.array([0.0], dtype=np.float32)}), set())


def _fake_model() -> _OptionModelBase:
    """An opaque sentinel standing in for an option model."""
    return cast(_OptionModelBase, object())


def test_probe_model_resolution_prefers_provider() -> None:
    """With no provider the probe runs ctx.option_model (solve phase); with a
    provider installed it runs the provider's model and never touches
    ctx.option_model (synthesis phase)."""
    stale = _fake_model()
    candidate = _fake_model()
    ctx = ToolContext()
    ctx.option_model = stale
    sim = BeliefProbe(ctx)
    assert sim._option_model() is stale

    ctx.probe_option_model_provider = lambda: candidate
    assert sim._option_model() is candidate

    # Provider errors (e.g. no loadable simulator.py yet) surface to the
    # caller instead of falling back to the stale model.
    def _no_candidate() -> _OptionModelBase:
        raise RuntimeError("no candidate simulator yet")

    ctx.probe_option_model_provider = _no_candidate
    with pytest.raises(RuntimeError, match="no candidate simulator"):
        sim._option_model()


def test_probe_reset_requires_task_idx_during_synthesis() -> None:
    """During synthesis, reset() must not silently fall back to the (stale,
    solve-time) ctx.current_task: task_idx is required."""
    task = _tiny_task()
    ctx = ToolContext()
    ctx.train_tasks = [task]
    ctx.current_task = task

    # Solve phase: current-task fallback works.
    sim = BeliefProbe(ctx)
    sim.reset()
    assert sim._state is not None

    # Synthesis phase: explicit task_idx required...
    ctx.probe_option_model_provider = _fake_model
    sim = BeliefProbe(ctx)
    with pytest.raises(ValueError, match="task_idx explicitly"):
        sim.reset()
    # ...and accepted.
    sim.reset(task_idx=0)
    assert sim._state is not None


def test_candidate_probe_model_provider_glue(tmp_path, monkeypatch) -> None:
    """The provider gates on a loadable simulator.py, caches by content hash
    (no refit for an unchanged file), and rebuilds on change.

    Exercises the real ``_make_candidate_probe_model_provider`` and the
    real file loader; only the fit/build layer below
    ``build_candidate_option_model`` is stubbed (its body is the shared
    ``evaluate_plan_refinement`` path).
    """
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {}
    approach._latent_init = None
    fit_calls = {"n": 0}

    def _fake_fit(rules, specs, triples, features):
        del rules, triples, features
        fit_calls["n"] += 1
        names = [s.name for s in specs]
        return FitResult(names=names,
                         samples=np.array([[s.init_value for s in specs]]),
                         log_probs=np.array([0.0])), 0.0

    monkeypatch.setattr(
        "predicators.approaches.synthesis_validation.fit_rule_parameters",
        _fake_fit)
    setattr(approach, "_build_combined_simulator", lambda learned: learned)
    setattr(approach, "_build_option_model", lambda sim: ("model", sim))

    simulator_file = str(tmp_path / "simulator.py")
    provider = approach._make_candidate_probe_model_provider(
        simulator_file,
        trajectories=[],
        base_pred_triples=[],
        inferred_hint={"thing": ["x"]})

    # No file yet: hard error, never a fallback model.
    with pytest.raises(RuntimeError, match="no candidate simulator yet"):
        provider()

    # Broken file: hard error too.
    with open(simulator_file, "w", encoding="utf-8") as f:
        f.write("RESIDUAL_RULES = None\n")
    with pytest.raises(RuntimeError, match="failed to load"):
        provider()

    valid = ("def _rule(state, updates, params):\n"
             "    return updates\n"
             "RESIDUAL_RULES = [_rule]\n"
             "PARAM_SPECS = [ParamSpec('k', 1.0, lo=0.0, hi=2.0)]\n"
             "RESIDUAL_FEATURES = {'thing': ['x']}\n")
    with open(simulator_file, "w", encoding="utf-8") as f:
        f.write(valid)
    model = provider()
    assert fit_calls["n"] == 1
    assert approach._fitted_params == {"k": 1.0}

    # Unchanged content: cached, no refit.
    assert provider() is model
    assert fit_calls["n"] == 1

    # Changed content: rebuilt.
    with open(simulator_file, "w", encoding="utf-8") as f:
        f.write(valid.replace("1.0, lo", "1.5, lo"))
    assert provider() is not model
    assert fit_calls["n"] == 2
    assert approach._fitted_params == {"k": 1.5}


def test_probe_descriptions_follow_phase() -> None:
    """The probe surface follows the session: explore_python (solve-only)
    carries the belief-simulator + evaluate_option_plan wording, while in
    synthesis the probe rides inside run_python, whose description carries the
    candidate-simulator + evaluate_plan_refinement wording."""
    utils.reset_config({"agent_planner_use_explore_python": True})

    def _desc(ctx: ToolContext) -> str:
        (tool, ) = (t for t in create_mcp_tools(ctx, ["explore_python"])
                    if getattr(t, "name", "") == "explore_python")
        description = getattr(tool, "description", "")
        assert description
        return description

    solve_desc = _desc(ToolContext())
    assert "belief simulator" in solve_desc
    assert "evaluate_option_plan" in solve_desc
    # The solve namespace also carries the recorded real trajectories.
    assert "trajectories" in solve_desc
    assert "describe_trajectory" in solve_desc

    def _run_python_desc() -> str:
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.tools import create_synthesis_tools
        toolkit = create_synthesis_tools(exec_ns={},
                                         base_pred_triples=[],
                                         inferred_residual_features={},
                                         simulator_file="/dev/null",
                                         versions_dir="/dev/null")
        (run_python, ) = (t for t in toolkit.tools
                          if getattr(t, "name", "") == "run_python")
        description = getattr(run_python, "description", "")
        assert description
        return description

    synth_desc = _run_python_desc()
    assert "CANDIDATE simulator" in synth_desc
    assert "task_idx is required" in synth_desc
    assert "sim.fit" in synth_desc
    assert "sim.residuals" in synth_desc
    # The fit/refine/forward-run protocol replaced the old validation
    # tool, and the probe is unconditional in synthesis sessions.
    assert "evaluate_plan_refinement" not in synth_desc
    utils.reset_config({"agent_planner_use_explore_python": False})
    assert "CANDIDATE simulator" in _run_python_desc()


def test_probe_namespace_contract() -> None:
    """The solve exec namespace carries the probe, numpy, and the recorded.

    real trajectories (read-only evidence) - and deliberately nothing
    evaluator-shaped or authoring-shaped (those are synthesis-role; see
    build_probe_namespace).
    """
    utils.reset_config({})
    ctx = ToolContext()
    ns = build_probe_namespace(ctx)
    assert {"sim", "BeliefProbe", "np", "trajectories", "describe_trajectory"
            } <= set(ns)
    assert "evaluate_trajectory" not in ns
    assert "Predicate" not in ns
    assert "ParamSpec" not in ns

    # No data collected yet: the digest helper says so instead of
    # crashing cryptically.
    with pytest.raises(ValueError, match="No trajectories"):
        ns["describe_trajectory"](0)


def test_probe_task_digest() -> None:
    """sim.task() describes the current task in solve sessions, requires an
    explicit train-task index during synthesis (stale-current-task guard, same
    as reset), and only advertises the is_goal_state query in synthesis, whose
    exec namespace binds it."""
    utils.reset_config({})
    task = _tiny_task()
    ctx = ToolContext()
    ctx.train_tasks = [task]
    ctx.current_task = task
    sim = BeliefProbe(ctx)

    solve_digest = sim.task()
    assert "current solve task" in solve_digest
    assert "thing0:thing" in solve_digest
    assert "is_goal_state" not in solve_digest
    assert sim.task(0).startswith("Task 0:")
    with pytest.raises(ValueError, match="Invalid task_idx"):
        sim.task(3)

    ctx.probe_option_model_provider = _fake_model
    with pytest.raises(ValueError, match="task_idx explicitly"):
        sim.task()
    synth_digest = sim.task(0)
    assert "is_goal_state(state, 0)" in synth_digest


def test_probe_fit_gating_and_delegation() -> None:
    """sim.fit delegates to ctx.probe_fit_provider in synthesis sessions and
    raises in solve sessions (the deployed belief model is fixed there)."""
    utils.reset_config({})
    ctx = ToolContext()
    sim = BeliefProbe(ctx)
    with pytest.raises(RuntimeError, match="unavailable in this session"):
        sim.fit()

    calls: dict = {}

    def _provider(path=None, traj_idxs=None, fixed=None) -> str:
        calls.update(path=path, traj_idxs=traj_idxs, fixed=fixed)
        return "[cycle_000_vers_000] fit report"

    ctx.probe_fit_provider = _provider
    out = sim.fit(traj_idxs=[0], fixed={"k": 1.0})
    assert out == "[cycle_000_vers_000] fit report"
    assert calls == {"path": None, "traj_idxs": [0], "fixed": {"k": 1.0}}


def test_probe_residuals_gating_and_delegation() -> None:
    """sim.residuals delegates to ctx.probe_residuals_provider in synthesis
    sessions and raises in solve sessions (no candidate simulator to score)."""
    utils.reset_config({})
    ctx = ToolContext()
    sim = BeliefProbe(ctx)
    with pytest.raises(RuntimeError, match="unavailable in this session"):
        sim.residuals()

    calls: dict = {}

    def _provider(**kwargs) -> str:
        calls.update(kwargs)
        return "[cycle_000_vers_000] residual report"

    ctx.probe_residuals_provider = _provider
    out = sim.residuals(max_transitions=5, fit_params=True)
    assert out == "[cycle_000_vers_000] residual report"
    assert calls == {
        "max_transitions": 5,
        "abs_tol": 1e-4,
        "rel_tol": 1e-3,
        "num_worst_examples": 3,
        "fit_params": True,
        "path": None,
        "rollout": False,
        "sweep_num_points": 6,
        "sweep_params": None,
        "phys_params": None,
    }
    calls.clear()
    sim.residuals(rollout=True, sweep_params=["lateral_friction"])
    assert calls["rollout"] is True
    assert calls["sweep_params"] == ["lateral_friction"]
    calls.clear()
    sim.residuals(rollout=True, phys_params={"lateral_friction": 0.3})
    assert calls["phys_params"] == {"lateral_friction": 0.3}


def test_probe_run_reports_subgoal_divergence() -> None:
    """ProbeResult renders per-step SUBGOAL NOT REACHED lines, so a single
    continuous sim.run of a refined plan is the forward-validation pass."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.belief_probe import ProbeResult
    step = {
        "option": "Place(robot)[0.1]",
        "num_actions": 7,
        "failure": None,
        "added": [],
        "deleted": [],
        "subgoals_missing": ["WidgetAtFixture(widget0, fixture0)"],
        "image": None,
    }
    task = _tiny_task()
    rendered = repr(ProbeResult([step], False, [], task.init, []))
    assert "SUBGOAL NOT REACHED: {WidgetAtFixture(widget0, fixture0)}" \
        in rendered
