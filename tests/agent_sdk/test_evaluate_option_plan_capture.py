"""Capture-gating tests for the ``evaluate_option_plan`` tool.

Drives the real MCP tool handler with a fake option model (no PyBullet),
covering the two gates in front of ``ctx.solved_plan``:

* multi-rollout validation - the shared sim env is nondeterministic
  across repeats, so a goal-reaching plan is captured only after every
  one of ``CFG.agent_plan_validation_rollouts`` rollouts succeeds; a
  flaky plan is reported to the agent instead of captured;
* task-evaluator legitimacy - a goal-reaching but ``legitimate=False``
  rollout is refused as a reward hack (the real evaluator applies the
  same certificate). The refusal is internal: the agent-facing report
  speaks only in (terminated, reward) terms and never leaks the
  certificate's legitimacy bool or reason string.

Under ``capture_best_effort_plan`` (the final-submission nudge) neither
gate refuses: the submission is captured regardless - honest shortfall,
certificate-rejected rollout, or flaky repeat - but is marked as NOT a
validated solve (``solved_plan_reached_goal=False``), so it executes for
its honest reward without counting as a solve.
"""

import asyncio
import contextlib
from typing import Any

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.tools import ToolContext, create_mcp_tools
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, ParameterizedOption, Predicate, State, Task, TaskEvaluator, Type

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

_PLAN_TEXT = "Move(block0:block)[0.95] -> {ReachedHi(block0:block)}"
# A plan that lands short of the goal (x=0.5 < 0.9), so ReachedHi never
# holds: an honest shortfall, not a reward hack.
_SHORTFALL_PLAN_TEXT = "Move(block0:block)[0.5]"


class _Model:
    """Fake option model: Move sets block.x to its parameter value.

    ``succeed_first_n`` bounds how many calls apply the parameter; later
    calls leave the state unchanged, emulating a flaky plan whose repeat
    rollout misses the goal. Exposes ``last_trajectory`` so evaluator
    verdicts are NON-coarse (a coarse verdict never blocks capture).
    """

    last_execution_failure = None

    def __init__(self, succeed_first_n=10**9):
        self.num_calls = 0
        self._succeed_first_n = succeed_first_n
        self.last_trajectory = None

    def get_next_state_and_num_actions(self, state, option):
        """Roll the option forward one step, counting the call."""
        self.num_calls += 1
        nxt = state.copy()
        if self.num_calls <= self._succeed_first_n and len(option.params):
            nxt.set(_block, "x", float(option.params[0]))
        self.last_trajectory = LowLevelTrajectory(
            [state, nxt], [Action(np.zeros(1, dtype=np.float32))])
        return nxt, 1


class _StubEvaluator(TaskEvaluator):
    """Deterministic legitimacy verdict."""

    def __init__(self, goal, legit):
        super().__init__(goal)
        self._legit = legit

    def _certify(self, states, step_options, sim_env=None):
        if self._legit:
            return True, ""
        return False, "stub: the cascade was staged, not pushed"


def _make_ctx(model, evaluator=None, best_effort=False, goal_nl=None):
    init = State({_block: np.array([0.0], dtype=np.float32)})
    goal = {GroundAtom(_ReachedHi, [_block])}
    task = Task(init, goal, evaluator=evaluator, goal_nl=goal_nl)
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
    ctx.capture_goal_reaching_plans = True
    ctx.capture_best_effort_plan = best_effort
    return ctx


def _call_tool(ctx, plan_text=_PLAN_TEXT):
    """Invoke the real tool handler once against ``ctx``."""
    tools = {
        t.name: t.handler
        for t in create_mcp_tools(ctx, tool_names=["evaluate_option_plan"])
    }
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    result: Any = loop.run_until_complete(tools["evaluate_option_plan"]({
        "plan":
        plan_text
    }))
    return result["content"][0]["text"]


def _run_tool(model,
              evaluator=None,
              rollouts=3,
              plan_text=_PLAN_TEXT,
              best_effort=False,
              goal_nl=None):
    utils.reset_config({"agent_plan_validation_rollouts": rollouts})
    ctx = _make_ctx(model,
                    evaluator=evaluator,
                    best_effort=best_effort,
                    goal_nl=goal_nl)
    return _call_tool(ctx, plan_text), ctx


def test_robust_plan_is_captured_with_validation_note():
    """All rollouts succeed: captured, with the K/K validation note."""
    model = _Model()
    text, ctx = _run_tool(model, rollouts=3)
    assert "Captured as the current answer" in text
    assert "Validated 3/3 rollouts" in text
    assert ctx.solved_plan is not None
    # One reported rollout + two validation repeats.
    assert model.num_calls == 3


def test_flaky_plan_is_not_captured():
    """A repeat rollout that misses the goal blocks capture, loudly."""
    model = _Model(succeed_first_n=1)
    text, ctx = _run_tool(model, rollouts=3)
    assert "FLAKY (plan NOT captured)" in text
    assert "rollout 2/3 FAILED" in text
    assert "goal not reached" in text
    assert "ReachedHi" in text
    assert "Captured as the current answer" not in text
    assert ctx.solved_plan is None
    # The first rollout DID reach the goal - that is what makes it flaky.
    assert "Goal achieved: True" in text


def test_single_rollout_config_disables_repeats():
    """``agent_plan_validation_rollouts=1`` restores single-rollout capture."""
    model = _Model()
    text, ctx = _run_tool(model, rollouts=1)
    assert "Captured as the current answer" in text
    assert "Validated" not in text
    assert ctx.solved_plan is not None


def test_flaky_message_reports_all_rollout_outcomes():
    """The FLAKY report lists EVERY rollout's outcome and an estimated.

    reliability, instead of stopping at the first failure - the
    per-rollout list is what distinguishes failure modes.
    """
    model = _Model(succeed_first_n=1)
    text, _ = _run_tool(model, rollouts=3)
    assert "estimated reliability 1/3" in text
    assert "rollout 1: goal reached" in text
    assert "rollout 2: FAILED" in text
    assert "rollout 3: FAILED" in text
    # All three rollouts actually ran (no early break).
    assert model.num_calls == 3


def test_flaky_verdict_line_labeled_as_rollout_1():
    """When the submission is rejected as FLAKY, rollout 1's evaluator.

    verdict is labeled as such - unlabeled it read as a second,
    contradictory verdict in the same message.
    """
    model = _Model(succeed_first_n=1)
    evaluator = _StubEvaluator({GroundAtom(_ReachedHi, [_block])}, True)
    text, _ = _run_tool(model, evaluator=evaluator, rollouts=3)
    assert "FLAKY (plan NOT captured)" in text
    assert "[rollout 1 only - NOT the operative outcome" in text


def test_missing_goal_atoms_printed_even_with_goal_nl():
    """A goal-nl task still names the missing goal atoms on a shortfall - 'Goal
    achieved: False' alone left agents unable to tell a near-miss from a non-
    starter."""
    model = _Model()
    text, _ = _run_tool(model,
                        plan_text=_SHORTFALL_PLAN_TEXT,
                        goal_nl="topple the target")
    assert "Goal achieved: False" in text
    assert "Missing goal atoms" in text
    assert "ReachedHi" in text
    assert model.num_calls == 1


def test_illegitimate_plan_is_not_captured_and_skips_repeats():
    """A non-coarse ``legitimate=False`` verdict refuses capture before any
    validation repeats are spent, reported in reward terms only: the
    certificate's reason string never reaches the agent."""
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    text, ctx = _run_tool(model,
                          evaluator=_StubEvaluator(goal, legit=False),
                          rollouts=3)
    assert "NOT CAPTURED" in text
    assert "scores it as a non-solve" in text
    assert "stub: the cascade was staged" not in text
    assert "legitimate" not in text
    assert "Captured as the current answer" not in text
    assert ctx.solved_plan is None
    assert model.num_calls == 1


def test_legitimate_plan_passes_both_gates():
    """A legitimate goal-reaching plan validates and captures normally."""
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    text, ctx = _run_tool(model,
                          evaluator=_StubEvaluator(goal, legit=True),
                          rollouts=3)
    assert "Captured as the current answer" in text
    assert "Validated 3/3 rollouts" in text
    assert "Task evaluator" in text and "reward=" in text
    assert "solved=True" in text
    assert "legitimate" not in text
    assert ctx.solved_plan is not None
    assert model.num_calls == 3


def test_best_effort_honest_shortfall_is_captured():
    """An honest best-effort shortfall is captured, not refused.

    The plan does not reach the goal (terminated=False), so the
    evaluator marks it legitimate=False - there is no genuine cascade to
    certify. But it is not a reward hack, so under a best-effort
    submission it is captured and executes for its honest reward instead
    of being forfeited.
    """
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    text, ctx = _run_tool(model,
                          evaluator=_StubEvaluator(goal, legit=False),
                          rollouts=3,
                          plan_text=_SHORTFALL_PLAN_TEXT,
                          best_effort=True)
    assert "Captured as the current answer" in text
    assert "best-effort: goal NOT reached" in text
    assert "will not count as a solve" in text
    assert "NOT CAPTURED" not in text
    assert ctx.solved_plan is not None
    assert ctx.solved_plan_reached_goal is False
    assert "Goal achieved: False" in text


def test_best_effort_certificate_rejected_is_captured():
    """A best-effort submission captures even a certificate-rejected rollout.

    The plan reaches the goal atoms (terminated=True) but the evaluator
    scores the route as a non-solve. Outside best-effort mode that is
    refused as a reward hack, but at final submission the budget is
    spent: the plan is captured to execute for its honest reward, marked
    as NOT a validated solve (run_20260714_145053 task 4: this refusal
    forfeited the task entirely).
    """
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    text, ctx = _run_tool(model,
                          evaluator=_StubEvaluator(goal, legit=False),
                          rollouts=3,
                          best_effort=True)
    assert "Captured as the current answer" in text
    assert "best-effort" in text
    assert "will not count as a solve" in text
    assert "stub: the cascade was staged" not in text
    assert "legitimate" not in text
    assert "NOT CAPTURED" not in text
    assert ctx.solved_plan is not None
    assert ctx.solved_plan_reached_goal is False
    # Certificate rejection skips the validation repeats.
    assert model.num_calls == 1


def test_flaky_rejection_escalates_later_captures():
    """A FLAKY rejection escalates the gate for later captures on the task.

    A flaky submission is evidence the agent is tuning in a marginal
    parameter region, where a lucky streak can pass the base 3-rollout
    gate and die on the single real episode (run_20260717_182321: a
    20/20-swept relay placement validated 3/3, then missed the target
    for real). Resubmissions must therefore clear the escalated
    ``agent_plan_validation_rollouts_after_flaky`` gate.
    """
    utils.reset_config({
        "agent_plan_validation_rollouts": 3,
        "agent_plan_validation_rollouts_after_flaky": 6,
    })
    flaky_model = _Model(succeed_first_n=1)
    ctx = _make_ctx(flaky_model)
    text = _call_tool(ctx)
    assert "FLAKY (plan NOT captured)" in text
    assert "captures require 6/6 successful rollouts" in text
    # The resubmission (robust this time) faces the 6-rollout gate.
    robust_model = _Model()
    ctx.option_model = robust_model
    text2 = _call_tool(ctx)
    assert "Captured as the current answer" in text2
    assert "Validated 6/6 rollouts" in text2
    assert ctx.solved_plan is not None
    assert robust_model.num_calls == 6


def test_flaky_escalation_is_per_task():
    """Escalation is keyed to the task: a different test task keeps the base
    gate."""
    utils.reset_config({
        "agent_plan_validation_rollouts": 3,
        "agent_plan_validation_rollouts_after_flaky": 6,
    })
    flaky_model = _Model(succeed_first_n=1)
    ctx = _make_ctx(flaky_model)
    ctx.test_task_idx = 0
    text = _call_tool(ctx)
    assert "FLAKY (plan NOT captured)" in text
    robust_model = _Model()
    ctx.option_model = robust_model
    ctx.test_task_idx = 1
    text2 = _call_tool(ctx)
    assert "Validated 3/3 rollouts" in text2
    assert robust_model.num_calls == 3


def test_validation_rollouts_enter_fresh_env_scope():
    """Each validation repeat runs inside ``ctx.validation_env_scope``; the
    reported first rollout stays on the shared env."""
    utils.reset_config({
        "agent_plan_validation_rollouts": 3,
        "agent_plan_validation_fresh_env": True,
    })
    entered = []

    @contextlib.contextmanager
    def _scope():
        entered.append(True)
        yield

    model = _Model()
    ctx = _make_ctx(model)
    ctx.validation_env_scope = _scope
    text = _call_tool(ctx)
    assert "Captured as the current answer" in text
    assert "freshly constructed simulator" in text
    # Rollouts 2 and 3 of 3; rollout 1 is the reported one.
    assert len(entered) == 2


def test_fresh_env_scope_disabled_by_config():
    """``agent_plan_validation_fresh_env=False`` keeps repeats on the shared
    env even when a scope is installed."""
    utils.reset_config({
        "agent_plan_validation_rollouts": 3,
        "agent_plan_validation_fresh_env": False,
    })
    entered = []

    @contextlib.contextmanager
    def _scope():
        entered.append(True)
        yield

    model = _Model()
    ctx = _make_ctx(model)
    ctx.validation_env_scope = _scope
    text = _call_tool(ctx)
    assert "Captured as the current answer" in text
    assert "freshly constructed simulator" not in text
    assert not entered


class _PhysicsAwareModel(_Model):
    """Fake model whose success depends on a physics 'parameter'.

    Move only applies its parameter while ``friction >= 0.5``, emulating
    a plan whose success band excludes part of the fit posterior. The
    physics-margin scope perturbs ``friction`` the way the real scope
    perturbs the fresh env's physical params.
    """

    def __init__(self):
        super().__init__()
        self.friction = 0.53

    def get_next_state_and_num_actions(self, state, option):
        if self.friction >= 0.5:
            return super().get_next_state_and_num_actions(state, option)
        self.num_calls += 1
        nxt = state.copy()
        self.last_trajectory = LowLevelTrajectory(
            [state, nxt], [Action(np.zeros(1, dtype=np.float32))])
        return nxt, 1


def _physics_scope_ctx(model, points):
    """A ctx whose fresh-env scope applies physics overrides to ``model``."""
    ctx = _make_ctx(model)
    scope_overrides = []

    @contextlib.contextmanager
    def _scope(physical_overrides=None):
        scope_overrides.append(physical_overrides)
        prev = model.friction
        if physical_overrides:
            model.friction = physical_overrides["lateral_friction"]
        try:
            yield
        finally:
            model.friction = prev

    ctx.validation_env_scope = _scope
    ctx.physics_margin_provider = lambda: list(points)
    return ctx, scope_overrides


def test_param_sensitive_plan_is_not_captured():
    """A plan that fails at a -1-sigma physics point is refused.

    Regression for run_20260723_091108: a capture validated 8/8 at the
    fitted lateral_friction 0.5319 failed deterministically at true 0.5
    - execution repeats at the fitted values cannot see zero margin to
    the fit's parameter error.
    """
    utils.reset_config({
        "agent_plan_validation_rollouts": 3,
        "agent_plan_validation_fresh_env": True,
        "agent_plan_validation_physics_margin": True,
    })
    model = _PhysicsAwareModel()
    ctx, scope_overrides = _physics_scope_ctx(model, [{
        "lateral_friction": 0.48
    }, {
        "lateral_friction": 0.59
    }])
    text = _call_tool(ctx)
    assert "PARAM-SENSITIVE (plan NOT captured)" in text
    assert "lateral_friction=0.48" in text
    assert ctx.solved_plan is None
    # 2 execution repeats (no overrides) + 2 physics points.
    assert scope_overrides == [
        None, None, {
            "lateral_friction": 0.48
        }, {
            "lateral_friction": 0.59
        }
    ]


def test_physics_margin_pass_is_captured_with_note():
    """Margin points inside the success band capture with the margin note."""
    utils.reset_config({
        "agent_plan_validation_rollouts": 3,
        "agent_plan_validation_fresh_env": True,
        "agent_plan_validation_physics_margin": True,
    })
    model = _PhysicsAwareModel()
    ctx, _ = _physics_scope_ctx(model, [{
        "lateral_friction": 0.51
    }, {
        "lateral_friction": 0.59
    }])
    text = _call_tool(ctx)
    assert "Captured as the current answer" in text
    assert "Physics-margin check passed" in text


def test_physics_margin_disabled_by_config():
    """The default-off flag skips the margin rollouts entirely."""
    utils.reset_config({
        "agent_plan_validation_rollouts": 3,
        "agent_plan_validation_fresh_env": True,
        "agent_plan_validation_physics_margin": False,
    })
    model = _PhysicsAwareModel()
    ctx, scope_overrides = _physics_scope_ctx(model, [{
        "lateral_friction": 0.48
    }])
    text = _call_tool(ctx)
    assert "Captured as the current answer" in text
    assert "PARAM-SENSITIVE" not in text
    assert scope_overrides == [None, None]


def test_physics_margin_vacuous_without_points():
    """An empty provider (no fit / degenerate posterior) adds no note."""
    utils.reset_config({
        "agent_plan_validation_rollouts": 3,
        "agent_plan_validation_fresh_env": True,
        "agent_plan_validation_physics_margin": True,
    })
    model = _PhysicsAwareModel()
    ctx, scope_overrides = _physics_scope_ctx(model, [])
    text = _call_tool(ctx)
    assert "Captured as the current answer" in text
    assert "Physics-margin check passed" not in text
    assert scope_overrides == [None, None]


def test_best_effort_flaky_plan_is_captured():
    """A best-effort submission captures a flaky plan instead of refusing.

    Rollout 1 solves, rollout 2 misses. Outside best-effort mode that is
    refused as FLAKY (the agent can add margin and resubmit), but at
    final submission there is no budget left, so the plan is captured
    with the flaky detail in the note and marked as NOT a validated
    solve.
    """
    model = _Model(succeed_first_n=1)
    text, ctx = _run_tool(model, rollouts=3, best_effort=True)
    assert "Captured as the current answer" in text
    assert "best-effort" in text
    assert "rollout 2/3 FAILED" in text
    assert "FLAKY (plan NOT captured)" not in text
    assert ctx.solved_plan is not None
    assert ctx.solved_plan_reached_goal is False


def test_capture_stashes_evaluator_reward():
    """A capture records the evaluator verdict's reward for the restart loop's
    cross-attempt ranking."""
    model = _Model()
    goal = {GroundAtom(_ReachedHi, [_block])}
    _, ctx = _run_tool(model,
                       evaluator=_StubEvaluator(goal, legit=True),
                       rollouts=1)
    assert ctx.solved_plan is not None
    assert isinstance(ctx.solved_plan_eval_reward, float)


def test_capture_without_evaluator_has_no_reward():
    """No evaluator: the reward stash stays None (ranked below any rewarded
    capture, above no capture)."""
    model = _Model()
    _, ctx = _run_tool(model, rollouts=1)
    assert ctx.solved_plan is not None
    assert ctx.solved_plan_eval_reward is None


def test_budget_footer_during_attempt():
    """The footer reports THIS call's rollout delta, not the attempt's
    cumulative total masquerading as one."""
    import time as _time  # pylint: disable=import-outside-toplevel
    utils.reset_config({"agent_plan_validation_rollouts": 1})
    model = _Model()
    ctx = _make_ctx(model)
    ctx.attempt_start = _time.monotonic()
    # Simulate a prior explore sweep this attempt.
    ctx.attempt_rollout_count = 50
    text = _call_tool(ctx, _PLAN_TEXT)
    assert "[budget] attempt time" in text
    assert "sim rollouts this attempt: 51 (+1 this call)" in text


def test_no_budget_footer_outside_attempt():
    """No attempt in flight: no footer noise."""
    model = _Model()
    text, _ctx = _run_tool(model, rollouts=1)
    assert "[budget]" not in text


def test_validation_repeats_use_decorrelated_planner_seeds():
    """Each validation repeat rolls out under its own ``CFG.seed``.

    A fresh env per repeat is not enough for independent samples: the
    skills' motion planning reads the constant ``CFG.seed`` at call
    time, so identical-seed repeats are bit-identical replays and the
    flaky gate detects nothing (run_20260722_204632: a 13/13-validated
    capture was a coin flip on the real episode). The capture rollout
    itself must keep the base seed; the repeats offset it; the base seed
    must be restored afterward.
    """

    class _SeedRecordingModel(_Model):
        """Records ``CFG.seed`` at each rollout step."""

        def __init__(self):
            super().__init__()
            self.seeds = []

        def get_next_state_and_num_actions(self, state, option):
            from predicators.settings import \
                CFG  # pylint: disable=import-outside-toplevel
            self.seeds.append(CFG.seed)
            return super().get_next_state_and_num_actions(state, option)

    model = _SeedRecordingModel()
    _, ctx = _run_tool(model, rollouts=3)
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    base = CFG.seed
    assert ctx.solved_plan is not None
    # One capture rollout at the base seed, two decorrelated repeats.
    assert model.seeds == [base, base + 1, base + 2]
