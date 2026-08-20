"""Tests for ``BeliefProbe.run(physics_sweep=True)``.

The sweep re-runs a plan once per identified-physical-parameter grid
point (the same points the capture gate's physics-margin check uses),
each on a fresh env at the base planner seed, so the agent can find
interior failure holes BEFORE submitting (run_20260724_140531: a capture
passed both +-1-sigma endpoints and failed deterministically at the true
value between them).
"""
# pylint: disable=protected-access
import contextlib
import time

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.belief_probe import BeliefProbe, ProbeBudgetExceeded
from predicators.agent_sdk.tools import ToolContext
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block = Object("block0", _block_type)
_ReachedHi = Predicate("ReachedHi", [_block_type],
                       lambda s, o: s.get(o[0], "x") >= 0.9)


def _noop_policy(_s, _m, _o, _p):
    return Action(np.zeros(1, dtype=np.float32))


_Move = ParameterizedOption(
    "Move",
    types=[_block_type],
    params_space=Box(low=np.array([0.0], dtype=np.float32),
                     high=np.array([1.0], dtype=np.float32)),
    policy=_noop_policy,
    initiable=lambda _s, _m, _o, _p: True,
    terminal=lambda _s, _m, _o, _p: False,
)


class _PhysicsHoleModel:
    """Fake option model with a failure hole in its physics parameter.

    Move applies its parameter unless ``friction`` sits inside the
    (0.49, 0.51) hole - emulating the speckled success band of
    run_20260724_140531, where a design passed both +-1-sigma endpoints
    and failed at the true value between them.
    """

    last_execution_failure = None

    def __init__(self):
        self.num_calls = 0
        self.friction = 0.4746
        self.last_trajectory = None

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        nxt = state.copy()
        if len(option.params) and not 0.49 < self.friction < 0.51:
            nxt.set(_block, "x", float(option.params[0]))
        self.last_trajectory = LowLevelTrajectory(
            [state, nxt], [Action(np.zeros(1, dtype=np.float32))])
        return nxt, 1


def _make_ctx(points):
    init = State({_block: np.array([0.0], dtype=np.float32)})
    goal = {GroundAtom(_ReachedHi, [_block])}
    task = Task(init, goal)
    model = _PhysicsHoleModel()
    ctx = ToolContext(
        types={_block_type},
        predicates={_ReachedHi},
        processes=set(),
        options={_Move},
        train_tasks=[task],
        example_state=init,
        option_model=model,
        current_task=task,
    )
    scope_overrides = []

    @contextlib.contextmanager
    def _scope(physical_overrides=None):
        scope_overrides.append(physical_overrides)
        prev = model.friction
        if physical_overrides:
            model.friction = physical_overrides["friction"]
        try:
            yield
        finally:
            model.friction = prev

    ctx.validation_env_scope = _scope
    ctx.physics_margin_provider = lambda: list(points)
    return ctx, model, scope_overrides


def test_physics_sweep_reports_interior_hole():
    """One rollout per point (fitted first), failures called out."""
    utils.reset_config({})
    points = [{"friction": mu} for mu in (0.43, 0.48, 0.5, 0.52)]
    ctx, _, scope_overrides = _make_ctx(points)
    sim = BeliefProbe(ctx)
    sim.reset()
    res = sim.run("Move(block0:block)[0.95]", render=False, physics_sweep=True)
    # Fitted reference point + one per provider point, in order.
    assert scope_overrides == [None] + points
    assert [p["params"] for p in res.points] == [None] + points
    assert res.successes == 4
    assert [p["goal_reached"] for p in res.points] == \
        [True, True, True, False, True]
    text = res.text
    assert "Physics sweep: 4/5 points reached the goal" in text
    assert "friction=0.5: goal NOT reached" in text
    # A failing point triggers the non-monotonicity guidance.
    assert "fails INSIDE the identified-parameter uncertainty range" in text
    # The sweep is a measurement, not navigation: state is unchanged.
    assert sim._require_state().get(_block, "x") == 0.0
    assert ctx.attempt_rollout_count == 5


def test_physics_sweep_all_pass_has_no_hole_guidance():
    """A clean sweep reports plainly, without the failure guidance."""
    utils.reset_config({})
    points = [{"friction": mu} for mu in (0.43, 0.52)]
    ctx, _, _ = _make_ctx(points)
    sim = BeliefProbe(ctx)
    sim.reset()
    res = sim.run("Move(block0:block)[0.95]", render=False, physics_sweep=True)
    assert res.successes == 3
    assert "fails INSIDE" not in res.text


def test_physics_sweep_mode_exclusivity():
    """physics_sweep varies the physics; trials/solved/contacts measure the.

    plan at the fitted values - combining them is refused loudly.
    """
    utils.reset_config({})
    ctx, _, _ = _make_ctx([{"friction": 0.43}])
    sim = BeliefProbe(ctx)
    sim.reset()
    for kwargs in ({"trials": 3}, {"solved": True}, {"contacts": True}):
        with pytest.raises(ValueError, match="its own mode"):
            sim.run("Move(block0:block)[0.95]",
                    render=False,
                    physics_sweep=True,
                    **kwargs)


def test_physics_sweep_requires_scope_and_points():
    """No fresh-env scope or no identified params -> honest refusal."""
    utils.reset_config({})
    ctx, _, _ = _make_ctx([{"friction": 0.43}])
    ctx.validation_env_scope = None
    sim = BeliefProbe(ctx)
    sim.reset()
    with pytest.raises(ValueError, match="fresh-env scope"):
        sim.run("Move(block0:block)[0.95]", render=False, physics_sweep=True)
    ctx2, _, _ = _make_ctx([])
    sim2 = BeliefProbe(ctx2)
    sim2.reset()
    with pytest.raises(ValueError, match="no identified physical"):
        sim2.run("Move(block0:block)[0.95]", render=False, physics_sweep=True)


def test_physics_sweep_returns_partial_on_mid_loop_budget_expiry():
    """A budget stop mid-sweep returns the completed points."""
    utils.reset_config({})
    points = [{"friction": mu} for mu in (0.43, 0.52)]
    ctx, model, _ = _make_ctx(points)
    ctx.attempt_deadline = time.monotonic() + 60.0
    orig = model.get_next_state_and_num_actions

    def _expire_after_rollout(state, option):
        result = orig(state, option)
        ctx.attempt_deadline = time.monotonic() - 1.0
        return result

    model.get_next_state_and_num_actions = _expire_after_rollout
    sim = BeliefProbe(ctx)
    sim.reset()
    res = sim.run("Move(block0:block)[0.95]", render=False, physics_sweep=True)
    assert len(res.points) == 1
    assert any("time budget expired after 1/3 sweep points" in n
               for n in res.notes)
    # Nothing completed -> nothing to salvage.
    ctx3, _, _ = _make_ctx(points)
    sim3 = BeliefProbe(ctx3)
    sim3.reset()
    ctx3.attempt_deadline = time.monotonic() - 1.0
    with pytest.raises(ProbeBudgetExceeded):
        sim3.run("Move(block0:block)[0.95]", render=False, physics_sweep=True)
