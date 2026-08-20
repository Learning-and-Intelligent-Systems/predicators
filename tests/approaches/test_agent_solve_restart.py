"""Tests for the time-boxed restart loop in AgentModelBasedApproach._solve.

The loop runs up to ``agent_solve_max_attempts`` solve attempts, each on
a fresh conversation when ``agent_solve_fresh_context`` is set: a
validated (evaluator-solved) capture returns immediately, best-effort
captures are banked and ranked by evaluator reward, journal auto-entries
record every attempt, and total failure re-raises the last error.
"""
# pylint: disable=protected-access
import time

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk import journal as journal_mod
from predicators.agent_sdk.session_base import AgentSessionFatalError
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.agent_model_based_approach import \
    AgentModelBasedApproach, _CaptureInfo
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_block0 = Object("block0", _block_type)
_Reached = Predicate("Reached", [_block_type],
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


def _make_approach(overrides=None, sandbox_dir=None):
    state = State({_block0: np.array([0.0], dtype=np.float32)})
    task = Task(state, {GroundAtom(_Reached, [_block0])})
    config = {
        "env": "cover",
        "approach": "agent_bilevel",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "seed": 42,
    }
    config.update(overrides or {})
    utils.reset_config(config)
    approach = AgentModelBasedApproach(
        initial_predicates={_Reached},
        initial_options={_Move},
        types={_block_type},
        action_space=Box(low=-1, high=1, shape=(1, )),
        train_tasks=[task],
        option_model=None,
    )
    if sandbox_dir is not None:
        approach._tool_context.sandbox_dir = sandbox_dir
    return approach, task


class _AttemptScript:
    """Scripted per-attempt outcomes for a stubbed _solve_attempt.

    Each script item is ``("validated", reward)``, ``("best_effort",
    reward)``, ``("fail", message)`` (raises ApproachFailure), or
    ``("error", exception)`` (raises that exception). The stub sets
    ``_last_capture_info`` exactly as _consume_validated_plan would.
    """

    def __init__(self, approach, script):
        self.approach = approach
        self.script = list(script)
        self.calls = 0
        self.policies = []

    def __call__(self, _task):
        kind, value = self.script[self.calls]
        self.calls += 1
        if kind == "fail":
            raise ApproachFailure(value)
        if kind == "error":
            raise value
        self.approach._last_capture_info = _CaptureInfo(
            validated=(kind == "validated"),
            reward=value,
            plan_lines=[f"Move(block0:block)[{0.9 + self.calls / 100.0}]"])
        policy = lambda _s: Action(np.zeros(1, dtype=np.float32))
        self.policies.append(policy)
        return policy


def test_validated_capture_returns_immediately():
    """A validated first attempt short-circuits the loop."""
    approach, task = _make_approach({
        "agent_solve_max_attempts": 3,
        "agent_solve_fresh_context": True,
    })
    closes = []
    approach._close_agent_session = lambda: closes.append(1)
    script = _AttemptScript(approach, [("validated", 0.95)])
    approach._solve_attempt = script
    policy = approach._solve(task, timeout=10)
    assert policy is script.policies[0]
    assert script.calls == 1
    assert len(closes) == 1  # fresh context for the (only) attempt


def test_best_effort_banked_and_best_reward_wins():
    """Best-effort captures are ranked by evaluator reward across attempts."""
    approach, task = _make_approach({"agent_solve_max_attempts": 3})
    script = _AttemptScript(approach, [
        ("best_effort", -0.10),
        ("best_effort", 0.40),
        ("fail", "attempt 3 found nothing"),
    ])
    approach._solve_attempt = script
    policy = approach._solve(task, timeout=10)
    assert script.calls == 3
    assert policy is script.policies[1]  # the reward-0.40 capture


def test_validated_on_later_attempt_beats_banked_best_effort():
    """A later validated capture wins over an earlier banked one."""
    approach, task = _make_approach({"agent_solve_max_attempts": 3})
    script = _AttemptScript(approach, [
        ("best_effort", 0.90),
        ("validated", 0.95),
    ])
    approach._solve_attempt = script
    policy = approach._solve(task, timeout=10)
    assert script.calls == 2
    assert policy is script.policies[1]


def test_all_attempts_fail_reraises():
    """With nothing captured anywhere, the last failure propagates."""
    approach, task = _make_approach({"agent_solve_max_attempts": 2})
    script = _AttemptScript(approach, [
        ("fail", "first"),
        ("fail", "second"),
    ])
    approach._solve_attempt = script
    with pytest.raises(ApproachFailure, match="second"):
        approach._solve(task, timeout=10)
    assert script.calls == 2


def test_no_fresh_context_keeps_session():
    """agent_solve_fresh_context=False never closes the session."""
    approach, task = _make_approach({
        "agent_solve_max_attempts": 2,
        "agent_solve_fresh_context": False,
    })
    closes = []
    approach._close_agent_session = lambda: closes.append(1)
    script = _AttemptScript(approach, [("fail", "a"), ("validated", 0.9)])
    approach._solve_attempt = script
    approach._solve(task, timeout=10)
    assert not closes


def test_journal_auto_entries_record_each_attempt(tmp_path):
    """Every attempt leaves an auto entry with outcome and plan."""
    approach, task = _make_approach(
        {
            "agent_solve_max_attempts": 2,
            "agent_solve_use_journal": True,
        },
        sandbox_dir=str(tmp_path))
    approach._tool_context.test_task_idx = 0
    script = _AttemptScript(approach, [
        ("fail", "nothing"),
        ("best_effort", -0.05),
    ])
    approach._solve_attempt = script
    approach._solve(task, timeout=10)
    content = journal_mod.read_journal(str(tmp_path))
    assert "### task 0 attempt 1/2 (auto)" in content
    assert "- outcome: no capture" in content
    assert "### task 0 attempt 2/2 (auto)" in content
    assert "best-effort capture (evaluator reward -0.05)" in content
    assert "Move(block0:block)" in content
    # Task context (goal + init state dict, the prompt's own
    # representation) is a dedicated entry written once, at the TOP of
    # the task's section - before any attempt entry.
    assert content.count("### task 0 goal + initial state (auto)") == 1
    assert content.count("- goal: Reached(block0:block)") == 1
    assert content.count("- initial state features:") == 1
    assert "'block0:block'" in content
    assert "'x'" in content
    assert content.index("goal + initial state") < content.index(
        "### task 0 attempt 1/2")


def test_attempt_bookkeeping_reset_per_attempt():
    """attempt_index/rollout counter/deadline are set and cleared."""
    approach, task = _make_approach({
        "agent_solve_max_attempts": 2,
        "agent_solve_attempt_wall_clock": 2700,
    })
    ctx = approach._tool_context
    seen = []

    def _attempt(_task):
        seen.append(
            (ctx.attempt_index, ctx.attempt_rollout_count, ctx.attempt_deadline
             is not None))
        ctx.attempt_rollout_count += 7
        raise ApproachFailure("no capture")

    approach._solve_attempt = _attempt
    with pytest.raises(ApproachFailure):
        approach._solve(task, timeout=10)
    assert seen == [(1, 0, True), (2, 0, True)]
    assert ctx.attempt_deadline is None
    assert ctx.attempt_start is None
    # attempt_index resets too: a stale index would mislabel journal
    # entries recorded outside any attempt.
    assert ctx.attempt_index == 0


def test_attempt_wall_spent():
    """_attempt_wall_spent trips BEFORE the deadline, at the spent floor.

    Agents that watch the [budget] footer end their query shortly before
    the deadline (run_20260718_125643 queries 002-003); an attempt whose
    remaining tools would only refuse must be labelled spent, not "no
    submission".
    """
    approach, _task = _make_approach({"agent_solve_attempt_wall_clock": 2700})
    ctx = approach._tool_context
    # No deadline set: never spent.
    assert not approach._attempt_wall_spent()
    ctx.attempt_deadline = time.monotonic() - 1.0
    assert approach._attempt_wall_spent()
    # 2 min left of 45: below the 20% floor (540s), counts as spent.
    ctx.attempt_deadline = time.monotonic() + 120.0
    assert approach._attempt_wall_spent()
    # 20 min left: plenty of budget still on the clock.
    ctx.attempt_deadline = time.monotonic() + 1200.0
    assert not approach._attempt_wall_spent()


def test_nudge_suspends_and_restores_attempt_deadline():
    """The nudge must not permanently disarm the wall clock.

    It suspends the deadline so its own submission is not refused, but
    a helper that leaked the None would let any future mid-attempt
    caller run unbounded - the exact runaway the time-box targets.
    """
    approach, _task = _make_approach({})
    ctx = approach._tool_context
    deadline = time.monotonic() + 1000.0
    ctx.attempt_deadline = deadline
    approach._query_agent_sync = lambda *a, **k: []
    policy = approach._nudge_final_submission()
    assert policy is None
    assert ctx.attempt_deadline == deadline
    assert ctx.capture_best_effort_plan is False


def test_unexpected_error_salvages_banked_capture():
    """A non-ApproachFailure error executes the banked best-effort capture
    instead of forfeiting it, and stops further attempts."""
    approach, task = _make_approach({"agent_solve_max_attempts": 3})
    ctx = approach._tool_context
    script = _AttemptScript(approach, [
        ("best_effort", 0.40),
        ("error", RuntimeError("pybullet exploded")),
    ])
    approach._solve_attempt = script
    policy = approach._solve(task, timeout=10)
    assert policy is script.policies[0]
    assert script.calls == 2  # attempt 3 never runs
    # Bookkeeping fully cleaned despite the unexpected error.
    assert ctx.attempt_start is None
    assert ctx.attempt_deadline is None
    assert ctx.attempt_index == 0


def test_fatal_session_error_reraises_even_with_banked_capture():
    """AgentSessionFatalError is never salvaged by a banked capture: the
    session backend is unusable, so the run must terminate (bookkeeping still
    cleaned by the finally)."""
    approach, task = _make_approach({"agent_solve_max_attempts": 3})
    ctx = approach._tool_context
    script = _AttemptScript(approach, [
        ("best_effort", 0.40),
        ("error", AgentSessionFatalError("3 consecutive agent queries died")),
    ])
    approach._solve_attempt = script
    with pytest.raises(AgentSessionFatalError):
        approach._solve(task, timeout=10)
    assert script.calls == 2  # attempt 3 never runs
    assert ctx.attempt_start is None
    assert ctx.attempt_deadline is None
    assert ctx.attempt_index == 0


def test_unexpected_error_without_bank_reraises_after_cleanup():
    """ApproachTimeout (an ApproachFailure SIBLING) propagates, but only.

    after attempt bookkeeping is cleared - stale fields would pollute
    later sessions sharing the ToolContext.
    """
    approach, task = _make_approach({
        "agent_solve_max_attempts": 3,
        "agent_solve_attempt_wall_clock": 2700,
    })
    ctx = approach._tool_context
    script = _AttemptScript(approach, [("error", ApproachTimeout("slow"))])
    approach._solve_attempt = script
    with pytest.raises(ApproachTimeout):
        approach._solve(task, timeout=10)
    assert ctx.attempt_start is None
    assert ctx.attempt_deadline is None
    assert ctx.attempt_index == 0


def test_journal_task_context_written_once_even_on_resolve(tmp_path):
    """Re-entering _solve for the same task (mid-episode replan) must not
    duplicate the goal + init-state entry."""
    approach, task = _make_approach(
        {
            "agent_solve_max_attempts": 1,
            "agent_solve_use_journal": True,
        },
        sandbox_dir=str(tmp_path))
    approach._tool_context.test_task_idx = 0
    script = _AttemptScript(approach, [("validated", 0.95),
                                       ("validated", 0.96)])
    approach._solve_attempt = script
    approach._solve(task, timeout=10)
    approach._solve(task, timeout=10)
    content = journal_mod.read_journal(str(tmp_path))
    assert content.count("### task 0 goal + initial state (auto)") == 1
    assert content.index("- goal:") < content.index("- outcome:")


def test_test_phase_journal_archived_and_rolled_back(tmp_path):
    """Learning entries persist across evaluations; each evaluation's own
    entries are archived outside the sandbox, then rolled back so the next
    evaluation starts from learning knowledge only (no test-task leaks)."""
    sandbox = tmp_path / "sandbox"
    log_dir = tmp_path / "run_logs"
    approach, task = _make_approach(
        {
            "agent_solve_max_attempts": 1,
            "agent_solve_use_journal": True,
            "log_file": str(log_dir),
        },
        sandbox_dir=str(sandbox))
    ctx = approach._tool_context
    # A learning-phase entry, recorded before any evaluation.
    journal_mod.append_entry(str(sandbox), "Agent notes (pre-test phase)",
                             "- learning fact")
    # First evaluation: one test-task solve writes auto entries.
    approach.begin_test_phase()
    ctx.test_task_idx = 0
    script = _AttemptScript(approach, [("validated", 0.95),
                                       ("validated", 0.96)])
    approach._solve_attempt = script
    approach._solve(task, timeout=10)
    content = journal_mod.read_journal(str(sandbox))
    assert "- learning fact" in content  # learning knowledge visible in eval
    assert "### task 0 goal + initial state (auto)" in content
    approach.end_test_phase()
    # Rolled back: the learning entry survives, eval entries are gone.
    content = journal_mod.read_journal(str(sandbox))
    assert "- learning fact" in content
    assert "task 0" not in content
    # The full eval journal was archived outside the sandbox first, one
    # copy per online-learning cycle.
    archived = (log_dir / "journal_eval_cycle0.md").read_text()
    assert "- learning fact" in archived
    assert "### task 0 goal + initial state (auto)" in archived
    # Second evaluation on the same task, after a learning phase advanced
    # the cycle: the context-entry dedup key was rolled back too, so the
    # goal + init entry is re-written (else the journal's attempt records
    # would be uninterpretable).
    approach._online_learning_cycle = 1
    approach.begin_test_phase()
    ctx.test_task_idx = 0
    approach._solve(task, timeout=10)
    content = journal_mod.read_journal(str(sandbox))
    assert content.count("### task 0 goal + initial state (auto)") == 1
    approach.end_test_phase()
    assert sorted(p.name for p in log_dir.glob("journal_eval*.md")) == [
        "journal_eval_cycle0.md", "journal_eval_cycle1.md"
    ]
    assert journal_mod.read_raw(str(sandbox)) is not None
    assert "task 0" not in journal_mod.read_journal(str(sandbox))


def test_test_phase_journal_rollback_noop_without_journal(tmp_path):
    """With the journal disabled the phase hooks touch nothing."""
    approach, _task = _make_approach({"agent_solve_use_journal": False},
                                     sandbox_dir=str(tmp_path))
    approach.begin_test_phase()
    approach.end_test_phase()
    assert journal_mod.read_raw(str(tmp_path)) is None


def test_journal_records_best_refused_submission(tmp_path):
    """An attempt with no capture journals the best refused submission the
    tools stashed, so the plan survives the fresh-context restart."""
    approach, task = _make_approach(
        {
            "agent_solve_max_attempts": 1,
            "agent_solve_use_journal": True,
        },
        sandbox_dir=str(tmp_path))
    ctx = approach._tool_context
    ctx.test_task_idx = 0

    def _attempt(_task):
        ctx.best_uncaptured_plan_lines = ["Move(block0:block)[0.87]"]
        ctx.best_uncaptured_reward = -0.05
        raise ApproachFailure("no capture")

    approach._solve_attempt = _attempt
    with pytest.raises(ApproachFailure):
        approach._solve(task, timeout=10)
    content = journal_mod.read_journal(str(tmp_path))
    assert ("- best refused submission (evaluator reward -0.05, "
            "not captured):") in content
    assert "Move(block0:block)[0.87]" in content
