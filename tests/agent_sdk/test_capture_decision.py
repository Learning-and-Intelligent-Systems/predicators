"""Direct unit tests for the pure ``_decide_capture`` function.

The e2e harness (``test_evaluate_option_plan_capture.py``) drives the
same policy through the real tool handler; these tests pin the decision
table itself, one test per :class:`CaptureDecision` case, including
guard combinations the e2e tests do not reach:

* capture disabled (``capture_goal_reaching_plans=False``) is silent for
  every otherwise-triggering combination;
* an empty grounded plan never captures (unreachable via the handler,
  whose parser rejects empty plans; the guard is preserved);
* best-effort mode with an existing validated capture ("best-effort
  never displaces validated") falls through to the loud refusals or to
  silence;
* a non-terminated evaluator rejection (illegitimate verdict whose
  ``terminated`` is False while the env goal-check passed) does not
  block a validated capture;
* flaky + evaluator-rejected cannot co-occur in the handler (validation
  repeats are skipped on a rejection) but the ``not
  evaluator_rejected`` guard on the FLAKY refusal is preserved;
* a wrong-task run without a reached goal is silent.
"""

from typing import Any, Dict

from predicators.agent_sdk.tools.capture import BestEffortReason, \
    CaptureDecision, CaptureOutcome, _decide_capture


def _decide(**overrides: Any) -> CaptureOutcome:
    """Call ``_decide_capture`` on the validated-solve baseline with
    overrides."""
    kwargs: Dict[str, Any] = dict(capture_enabled=True,
                                  is_current_task=True,
                                  have_plan=True,
                                  goal_achieved=True,
                                  evaluator_rejected=False,
                                  reward_hack=False,
                                  flaky=False,
                                  best_effort_mode=False,
                                  have_validated_capture=False)
    kwargs.update(overrides)
    return _decide_capture(**kwargs)


def test_validated_capture():
    """A clean goal-reaching plan on the current task is a validated solve.

    e2e: test_robust_plan_is_captured_with_validation_note.
    """
    outcome = _decide()
    assert outcome.decision is CaptureDecision.VALIDATED_CAPTURE
    assert outcome.best_effort_reason is None
    assert outcome.captured


def test_flaky_no_capture():
    """A flaky validation repeat refuses the capture.

    e2e: test_flaky_plan_is_not_captured.
    """
    outcome = _decide(flaky=True)
    assert outcome.decision is CaptureDecision.FLAKY_NO_CAPTURE
    assert outcome.best_effort_reason is None
    assert not outcome.captured


def test_param_sensitive_no_capture():
    """A physics-margin rollout failure refuses the capture.

    e2e: test_param_sensitive_plan_is_not_captured.
    """
    outcome = _decide(param_sensitive=True)
    assert outcome.decision is CaptureDecision.PARAM_SENSITIVE_NO_CAPTURE
    assert outcome.best_effort_reason is None
    assert not outcome.captured


def test_best_effort_param_sensitive():
    """Best-effort mode captures a param-sensitive submission."""
    outcome = _decide(best_effort_mode=True, param_sensitive=True)
    assert outcome.decision is CaptureDecision.BEST_EFFORT_CAPTURE
    assert outcome.best_effort_reason is BestEffortReason.PARAM_SENSITIVE


def test_flaky_outranks_param_sensitive_reason():
    """When both gates fail in best-effort mode, FLAKY is the reason (the.

    handler never produces this combination - the margin loop is skipped
    after a flaky repeat - but the reason ordering is pinned).
    """
    outcome = _decide(best_effort_mode=True, flaky=True, param_sensitive=True)
    assert outcome.decision is CaptureDecision.BEST_EFFORT_CAPTURE
    assert outcome.best_effort_reason is BestEffortReason.FLAKY


def test_reward_hack_no_capture():
    """A goal-atoms-reaching but evaluator-rejected rollout is refused.

    e2e: test_illegitimate_plan_is_not_captured_and_skips_repeats.
    """
    outcome = _decide(evaluator_rejected=True, reward_hack=True)
    assert outcome.decision is CaptureDecision.REWARD_HACK_NO_CAPTURE
    assert not outcome.captured


def test_wrong_task_note():
    """A success on a train task is flagged, never captured."""
    outcome = _decide(is_current_task=False)
    assert outcome.decision is CaptureDecision.WRONG_TASK_NOTE
    assert not outcome.captured


def test_no_capture_on_honest_failure():
    """A plan that simply misses the goal is silent - no capture, no flag."""
    outcome = _decide(goal_achieved=False)
    assert outcome.decision is CaptureDecision.NO_CAPTURE
    assert not outcome.captured


def test_best_effort_honest_shortfall():
    """Best-effort mode captures a goal-missing submission as a shortfall.

    e2e: test_best_effort_honest_shortfall_is_captured.
    """
    outcome = _decide(best_effort_mode=True, goal_achieved=False)
    assert outcome.decision is CaptureDecision.BEST_EFFORT_CAPTURE
    assert outcome.best_effort_reason is BestEffortReason.GOAL_NOT_REACHED
    assert outcome.captured


def test_best_effort_reward_hack():
    """Best-effort mode captures an evaluator-rejected submission.

    e2e: test_best_effort_certificate_rejected_is_captured.
    """
    outcome = _decide(best_effort_mode=True,
                      evaluator_rejected=True,
                      reward_hack=True)
    assert outcome.decision is CaptureDecision.BEST_EFFORT_CAPTURE
    assert outcome.best_effort_reason is BestEffortReason.REWARD_HACK


def test_best_effort_flaky():
    """Best-effort mode captures a flaky submission.

    e2e: test_best_effort_flaky_plan_is_captured.
    """
    outcome = _decide(best_effort_mode=True, flaky=True)
    assert outcome.decision is CaptureDecision.BEST_EFFORT_CAPTURE
    assert outcome.best_effort_reason is BestEffortReason.FLAKY


def test_validated_solve_wins_over_best_effort_mode():
    """Edge: best-effort mode does not demote a fully validated solve."""
    outcome = _decide(best_effort_mode=True)
    assert outcome.decision is CaptureDecision.VALIDATED_CAPTURE
    assert outcome.best_effort_reason is None


def test_best_effort_never_displaces_validated_capture():
    """Edge: with a validated capture already recorded, best-effort mode is
    inert - the loud refusals still fire, everything else is silent."""
    # A flaky resubmission is refused (and would escalate the gate)
    # instead of overwriting the validated capture.
    outcome = _decide(best_effort_mode=True,
                      have_validated_capture=True,
                      flaky=True)
    assert outcome.decision is CaptureDecision.FLAKY_NO_CAPTURE
    # A reward-hack resubmission is likewise refused.
    outcome = _decide(best_effort_mode=True,
                      have_validated_capture=True,
                      evaluator_rejected=True,
                      reward_hack=True)
    assert outcome.decision is CaptureDecision.REWARD_HACK_NO_CAPTURE
    # An honest shortfall is silent.
    outcome = _decide(best_effort_mode=True,
                      have_validated_capture=True,
                      goal_achieved=False)
    assert outcome.decision is CaptureDecision.NO_CAPTURE


def test_capture_disabled_is_always_silent():
    """Edge: without capture_goal_reaching_plans every combination is
    NO_CAPTURE - the open-loop planner must never record captures."""
    for overrides in (
        {},  # would be a validated solve
        {
            "flaky": True
        },
        {
            "evaluator_rejected": True,
            "reward_hack": True
        },
        {
            "is_current_task": False
        },
        {
            "best_effort_mode": True,
            "goal_achieved": False
        },
    ):
        outcome = _decide(capture_enabled=False, **overrides)
        assert outcome.decision is CaptureDecision.NO_CAPTURE


def test_empty_plan_never_captures():
    """Edge: no grounded steps means no capture, even for a would-be
    validated solve or best-effort submission."""
    assert _decide(have_plan=False).decision is CaptureDecision.NO_CAPTURE
    assert _decide(have_plan=False, best_effort_mode=True,
                   goal_achieved=False).decision is CaptureDecision.NO_CAPTURE


def test_non_terminated_evaluator_rejection_does_not_block():
    """Edge: an illegitimate verdict with terminated=False (env goal-check
    passed, evaluator's own termination did not) is not a reward hack and
    does not block the capture."""
    outcome = _decide(evaluator_rejected=True, reward_hack=False)
    assert outcome.decision is CaptureDecision.VALIDATED_CAPTURE


def test_flaky_refusal_requires_no_evaluator_rejection():
    """Edge: the FLAKY refusal's ``not evaluator_rejected`` guard - with a
    non-terminated rejection alongside flakiness the result is silence, not
    a FLAKY refusal (the handler never produces this combination because a
    rejection skips validation repeats)."""
    outcome = _decide(flaky=True, evaluator_rejected=True)
    assert outcome.decision is CaptureDecision.NO_CAPTURE


def test_wrong_task_note_requires_goal():
    """Edge: a train-task run that misses the goal is silent, not flagged."""
    outcome = _decide(is_current_task=False, goal_achieved=False)
    assert outcome.decision is CaptureDecision.NO_CAPTURE
