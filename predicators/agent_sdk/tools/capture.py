"""The ``evaluate_option_plan`` capture decision, as a pure function.

:func:`_decide_capture` encodes the run-verified capture gates in one
side-effect-free place: the handler in ``testing.py`` computes the
inputs, then performs the ctx mutations and message formatting its
decision calls for. Keeping the policy pure makes every guard
combination directly unit-testable
(``tests/agent_sdk/test_capture_decision.py``).
"""
import enum
from dataclasses import dataclass
from typing import Optional


class CaptureDecision(enum.Enum):
    """What ``evaluate_option_plan`` does with the submitted plan."""
    # Goal reached, evaluator-certified, every validation rollout passed:
    # captured and marked as a validated solve.
    VALIDATED_CAPTURE = "validated_capture"
    # Final-submission nudge: captured to execute for its honest reward,
    # but not marked as a solve (see BestEffortReason).
    BEST_EFFORT_CAPTURE = "best_effort_capture"
    # Goal reached on rollout 1 but a validation repeat failed: refused,
    # and later submissions on the task face the escalated gate.
    FLAKY_NO_CAPTURE = "flaky_no_capture"
    # Execution validation passed, but a rollout at +-1-posterior-sigma
    # perturbed physical params failed: the plan has no margin to the
    # physics fit's parameter error, so it is refused (the real env may
    # sit anywhere in that range).
    PARAM_SENSITIVE_NO_CAPTURE = "param_sensitive_no_capture"
    # Goal atoms reached via a route the task evaluator scores as a
    # non-solve: refused.
    REWARD_HACK_NO_CAPTURE = "reward_hack_no_capture"
    # Goal reached on a train task instead of the current task: not
    # captured, flagged loudly.
    WRONG_TASK_NOTE = "wrong_task_note"
    # Nothing to capture and nothing to flag.
    NO_CAPTURE = "no_capture"


class BestEffortReason(enum.Enum):
    """Why a best-effort capture cannot count as a validated solve."""
    GOAL_NOT_REACHED = "goal_not_reached"  # honest shortfall
    REWARD_HACK = "reward_hack"  # evaluator scores the rollout a non-solve
    FLAKY = "flaky"  # a validation repeat failed
    PARAM_SENSITIVE = "param_sensitive"  # failed at perturbed physics


@dataclass(frozen=True)
class CaptureOutcome:
    """A capture decision; ``best_effort_reason`` is set iff the decision is
    ``BEST_EFFORT_CAPTURE``."""
    decision: CaptureDecision
    best_effort_reason: Optional[BestEffortReason] = None

    @property
    def captured(self) -> bool:
        """Whether the plan is captured as the current answer."""
        return self.decision in (CaptureDecision.VALIDATED_CAPTURE,
                                 CaptureDecision.BEST_EFFORT_CAPTURE)


def _decide_capture(*,
                    capture_enabled: bool,
                    is_current_task: bool,
                    have_plan: bool,
                    goal_achieved: bool,
                    evaluator_rejected: bool,
                    reward_hack: bool,
                    flaky: bool,
                    best_effort_mode: bool,
                    have_validated_capture: bool,
                    param_sensitive: bool = False) -> CaptureOutcome:
    """Decide what ``evaluate_option_plan`` does with an evaluated plan.

    Pure: no ctx access, no I/O - the caller supplies exactly what the
    gates read and applies the side effects the decision calls for.

    - ``capture_enabled``: ``ctx.capture_goal_reaching_plans`` - only
      approaches that consume captured plans set it.
    - ``is_current_task``: the plan ran on the current solve/explore
      task (no ``task_idx`` given), the only task whose captures count.
    - ``have_plan``: the parsed plan grounded to at least one step.
    - ``goal_achieved``: goal reached, clean to goal, and within the
      episode horizon on the first rollout.
    - ``evaluator_rejected``: a non-coarse evaluator verdict scored the
      first rollout illegitimate.
    - ``reward_hack``: ``evaluator_rejected`` and the rollout actually
      reached the goal atoms (terminated) - an illegitimate route, as
      opposed to an honest shortfall.
    - ``flaky``: a validation repeat failed.
    - ``best_effort_mode``: ``ctx.capture_best_effort_plan`` - set only
      for the final-submission nudge after an attempt exhausted its
      turn budget.
    - ``have_validated_capture``: a validated-solve capture already
      exists (``ctx.solved_plan_reached_goal``).
    - ``param_sensitive``: execution validation passed but a rollout at
      +-1-posterior-sigma perturbed physical params failed - the plan
      has no margin to the physics fit's parameter error.

    With ``best_effort_mode`` (final-submission nudge after turn-cap
    exhaustion) capture the submission unconditionally: honest
    shortfall, evaluator-rejected rollout, and flaky repeat alike. The
    budget is spent, so executing the agent's best plan for its honest
    reward beats forfeiting the task (run_20260714_145053 task 4: a
    goal-reaching but certificate-rejected final submission was refused
    and the task forfeited, scoring n/a instead of its honest reward).
    Only a validated solve is marked as one; everything else is a
    best-effort capture that executes but cannot count as a solve - the
    certificate still protects the score.
    """
    # A best-effort capture never displaces a validated-solve capture.
    best_effort_capture = best_effort_mode and not have_validated_capture
    validated_solve = (goal_achieved and not reward_hack and not flaky
                       and not param_sensitive)
    if (capture_enabled and is_current_task
            and (validated_solve or best_effort_capture) and have_plan):
        if validated_solve:
            return CaptureOutcome(CaptureDecision.VALIDATED_CAPTURE)
        if not goal_achieved:
            reason = BestEffortReason.GOAL_NOT_REACHED
        elif reward_hack:
            reason = BestEffortReason.REWARD_HACK
        elif flaky:
            reason = BestEffortReason.FLAKY
        else:
            reason = BestEffortReason.PARAM_SENSITIVE
        return CaptureOutcome(CaptureDecision.BEST_EFFORT_CAPTURE, reason)
    if (capture_enabled and is_current_task and goal_achieved
            and not evaluator_rejected and flaky):
        # Loudly refuse a flaky capture: the agent still has this
        # session to add margin and resubmit, which beats discovering
        # the flakiness as a failed real episode.
        return CaptureOutcome(CaptureDecision.FLAKY_NO_CAPTURE)
    if (capture_enabled and is_current_task and goal_achieved
            and not evaluator_rejected and param_sensitive):
        # Loudly refuse a physics-margin failure: the fitted params are
        # uncertain at the reported sigma, and the real env may sit
        # anywhere in that range (run_20260723_091108: a capture
        # validated 8/8 at the fitted friction failed deterministically
        # at the true value just outside the design's success band).
        return CaptureOutcome(CaptureDecision.PARAM_SENSITIVE_NO_CAPTURE)
    if capture_enabled and is_current_task and reward_hack:
        # Loudly refuse a reward hack: the rollout reaches the goal atoms
        # but the evaluator's certificate rejects the route (e.g. the
        # target was knocked over directly), and the real evaluator
        # applies the same certificate, so it can never count as a solve.
        # (Under a best-effort final submission the same plan is instead
        # captured, flagged as a non-solve, to execute for its honest
        # reward.)
        return CaptureOutcome(CaptureDecision.REWARD_HACK_NO_CAPTURE)
    if capture_enabled and not is_current_task and goal_achieved:
        # Loudly flag a success that cannot count: agents have burned
        # whole sessions validating on a train task, believing they
        # were done (run_20260707_112310 test task 0, session 3).
        return CaptureOutcome(CaptureDecision.WRONG_TASK_NOTE)
    return CaptureOutcome(CaptureDecision.NO_CAPTURE)
