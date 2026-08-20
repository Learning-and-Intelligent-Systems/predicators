"""Forward execution and validation of grounded option plans.

Split out of ``bilevel_sketch`` (see that module's docstring for the
full layout); holds the shared forward-execution core
(``execute_plan_forward`` with its ``StepOutcome`` / ``ForwardResult``
records) and the continuous re-execution check
(``validate_plan_forward``) that ``sketch_refinement`` runs after a
successful refinement.
"""
import dataclasses
import logging
from typing import Callable, List, Optional, Sequence, Set, Tuple

from predicators import utils
from predicators.agent_sdk.sketch_types import SketchStep
from predicators.option_model import _OptionModelBase
from predicators.structs import GroundAtom, Object, Predicate, State, Task, \
    _Option


def _fmt_state_features(state: State,
                        objects: Optional[Sequence[Object]] = None) -> str:
    """Compact one-line dump of object features.

    Used by ``validate_plan_forward`` to trace how the continuous
    rollout's state drifts step by step, and by the deepest-failure
    report line. ``objects`` restricts the dump to those objects
    (default: every object in the state).
    """
    parts = []
    objs = sorted(state, key=lambda o: o.name) if objects is None else list(
        dict.fromkeys(objects))
    for obj in objs:
        feats = ", ".join(f"{f}={state.get(obj, f):.4f}"
                          for f in obj.type.feature_names)
        parts.append(f"{obj.name}[{feats}]")
    return " ".join(parts)


@dataclasses.dataclass
class StepOutcome:
    """Result of executing one option in ``execute_plan_forward``.

    ``post_state`` is ``None`` when the option was not initiable or
    raised (no state to continue from). ``failure_reason`` is ``None``
    on a clean step, else the reason (``"not initiable"`` / ``"0
    actions"`` / the option model's last failure / ``"env failure:
    ..."``). ``subgoal_missing`` holds the step's positive subgoal atoms
    that did NOT hold afterwards (only set when a sketch is supplied).
    """
    option: _Option
    pre_state: State
    post_state: Optional[State]
    num_actions: int
    initiable: bool
    failure_reason: Optional[str]
    subgoal_missing: Optional[Set[GroundAtom]]


@dataclasses.dataclass
class ForwardResult:
    """Outcome of executing a grounded plan forward through the option
    model."""
    steps: List[StepOutcome]
    final_state: State
    goal_reached: bool
    # First step that failed to execute (not initiable / 0 actions / env
    # failure), or None if every step executed.
    first_failure_idx: Optional[int]
    # First step whose positive subgoal atoms diverged, or None.
    first_subgoal_divergence_idx: Optional[int]
    # Total low-level actions executed across all options.
    total_actions: int = 0
    # First step index whose post-state satisfies the goal, or None if the
    # goal was never reached along the way.
    goal_step_idx: Optional[int] = None
    # Cumulative low-level actions through ``goal_step_idx`` (the count the
    # real, horizon-capped executor would spend to first reach the goal), or
    # None if the goal was never reached.
    actions_to_goal: Optional[int] = None

    @property
    def executed_all(self) -> bool:
        """True iff every option executed (initiable and >0 actions)."""
        return self.first_failure_idx is None

    @property
    def success(self) -> bool:
        """True iff every option executed AND the goal was reached."""
        return self.executed_all and self.goal_reached

    @property
    def clean_to_goal(self) -> bool:
        """True iff the goal is reached with no hard option failure at or
        before the step that first reaches it.

        Mirrors the real closed-loop executor, which aborts at the first
        failing option: a 0-action / not-initiable failure *after* the
        goal already holds is harmless, but one before it dooms the
        rollout. ``execute_plan_forward`` itself continues past such
        failures (the option model returns an unchanged post-state), so
        this guards against capturing a plan whose goal atoms only hold
        because forward simulation pressed on through a collision the
        real env would abort on.
        """
        if not self.goal_reached or self.goal_step_idx is None:
            return False
        return (self.first_failure_idx is None
                or self.first_failure_idx > self.goal_step_idx)


def execute_plan_forward(
    task: Task,
    plan: List[_Option],
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    sketch: Optional[List[SketchStep]] = None,
    on_step: Optional[Callable[[int, StepOutcome], None]] = None,
    stop_on_failure: bool = False,
) -> ForwardResult:
    """Execute a fully-grounded plan step by step through the option model.

    Shared forward-execution core behind ``validate_plan_forward`` (used
    by ``refine_plan_sketch``) and the ``evaluate_option_plan`` tool.
    State carries forward across options — matching how the real env
    executes. Per step it mirrors ``run_backtracking_refinement``'s
    fixed-plan path: check ``initiable``, call
    ``get_next_state_and_num_actions`` (catching
    ``EnvironmentFailure``), treat 0 actions as a failure, and — when a
    sketch is given — check the step's positive ``subgoal_atoms``
    against the post-state. ``on_step(i, outcome)`` is called after each
    step for callers that emit per-step reporting.

    Execution always stops early when a step is not initiable or raises
    (no post-state to continue from). When ``stop_on_failure`` is True it
    *also* stops at a 0-action step — mirroring the real closed-loop
    executor, which aborts at the first failing option rather than
    pressing on. When False (the default), a 0-action step is recorded as
    a failure but execution continues from the model's returned state
    (kept for ``validate_plan_forward``'s per-step diagnostics).
    """
    state = task.init
    steps: List[StepOutcome] = []
    first_failure_idx: Optional[int] = None
    first_div_idx: Optional[int] = None
    total_actions = 0
    goal_step_idx: Optional[int] = None
    actions_to_goal: Optional[int] = None

    for i, option in enumerate(plan):
        pre = state
        initiable = option.initiable(pre)
        post: Optional[State] = None
        num_actions = 0
        failure_reason: Optional[str] = None

        if not initiable:
            failure_reason = "not initiable"
        else:
            try:
                post, num_actions = \
                    option_model.get_next_state_and_num_actions(pre, option)
            except utils.EnvironmentFailure as e:
                failure_reason = f"env failure: {e}"
                post = None
            except Exception as e:  # pylint: disable=broad-except
                failure_reason = f"execution error: {type(e).__name__}: {e}"
                post = None
            else:
                if num_actions == 0:
                    failure_reason = (getattr(option_model,
                                              "last_execution_failure", None)
                                      or "0 actions")

        subgoal_missing: Optional[Set[GroundAtom]] = None
        if post is not None and sketch is not None and i < len(sketch):
            step = sketch[i]
            if step.subgoal_atoms:
                cur_atoms = utils.abstract(post, predicates)
                missing = step.subgoal_atoms - cur_atoms
                if missing:
                    subgoal_missing = missing
                    if first_div_idx is None:
                        first_div_idx = i

        outcome = StepOutcome(option=option,
                              pre_state=pre,
                              post_state=post,
                              num_actions=num_actions,
                              initiable=initiable,
                              failure_reason=failure_reason,
                              subgoal_missing=subgoal_missing)
        steps.append(outcome)
        if on_step is not None:
            on_step(i, outcome)

        if failure_reason is not None and first_failure_idx is None:
            first_failure_idx = i
        if post is None:
            break  # cannot continue without a post-state
        if failure_reason is not None and stop_on_failure:
            break  # mirror the real executor: abort at the first failure
        state = post
        total_actions += num_actions
        # Record when the goal *first* holds, plus the cumulative actions to
        # get there — the budget the real horizon-capped executor would
        # spend. A plan whose goal only holds after more steps than the
        # episode horizon allows is not real-executable.
        if goal_step_idx is None and task.goal_holds(state):
            goal_step_idx = i
            actions_to_goal = total_actions

    return ForwardResult(
        steps=steps,
        final_state=state,
        goal_reached=task.goal_holds(state),
        first_failure_idx=first_failure_idx,
        first_subgoal_divergence_idx=first_div_idx,
        total_actions=total_actions,
        goal_step_idx=goal_step_idx,
        actions_to_goal=actions_to_goal,
    )


def validate_plan_forward(
    task: Task,
    plan: List[_Option],
    option_model: _OptionModelBase,
    *,
    predicates: Set[Predicate],
    sketch: Optional[List[SketchStep]] = None,
    run_id: str = "bilevel",
) -> Tuple[bool, str]:
    """Re-execute a refined plan continuously, checking goal at the end.

    Runs all options sequentially with state carrying forward — matching
    how the real env will execute, and exposing accumulated state drift
    that refinement's per-step resets hide.

    When ``sketch`` is provided, also checks each step's ``subgoal_atoms``
    against the post-state and logs the first divergence with the missing
    atoms. Without ``sketch``, only the final goal is checked.

    Returns ``(success, diagnosis)``. ``diagnosis`` is a one-line summary
    of why validation failed (or ``""`` on success), suitable for surface
    in synthesis-tool output. The full failure context (state features,
    missing atoms, last option model error) is logged at INFO level.

    Single-shot per option (no resampling) — surfaces stochasticity-
    sensitive plans that refinement's resampling hides. Delegates the
    execution to ``execute_plan_forward``; this wrapper adds the INFO
    logging (per-step subgoal divergence, final state) and the one-line
    diagnosis, with priority execution-failure > goal-not-reached >
    subgoal-divergence.
    """
    n = len(plan)
    if n == 0:
        if task.goal_holds(task.init):
            return True, ""
        return False, "empty plan; init state does not satisfy goal"

    if sketch is not None and len(sketch) != n:
        logging.warning(
            "[%s] validate_plan_forward: sketch length %d != plan length %d; "
            "ignoring sketch (no per-step subgoal diagnostics).", run_id,
            len(sketch), n)
        sketch = None

    result = execute_plan_forward(task,
                                  plan,
                                  option_model,
                                  predicates=predicates,
                                  sketch=sketch)

    # Per-step subgoal divergence is a *signal*, not a hard failure (the
    # plan may establish a subgoal earlier, have it temporarily violated,
    # then re-establish it). Log each; remember the first for the diagnosis.
    first_div_msg = ""
    for i, outcome in enumerate(result.steps):
        if not outcome.subgoal_missing or outcome.post_state is None:
            continue
        step = sketch[i] if sketch is not None else None
        missing_strs = sorted(str(a) for a in outcome.subgoal_missing)
        opt_str = (f"{outcome.option.name}"
                   f"({', '.join(o.name for o in outcome.option.objects)})")
        logging.info(
            "[%s] Forward-validate subgoal divergence at step %d (%s):\n"
            "  expected:  %s\n"
            "  missing:   %s\n"
            "  full features: %s", run_id, i, opt_str,
            sorted(str(a)
                   for a in (step.subgoal_atoms or set())) if step else [],
            missing_strs, _fmt_state_features(outcome.post_state))
        if not first_div_msg:
            first_div_msg = (f"step {i} ({opt_str}): subgoals not satisfied "
                             f"after option (missing {missing_strs})")

    # Final-state log — only when every step executed (matches the old
    # behavior, where it ran at the last step's validation).
    if result.executed_all:
        final = result.final_state
        held = sorted(str(a) for a in task.goal if a.holds(final))
        missing = sorted(str(a) for a in task.goal if not a.holds(final))
        abstract_atoms = sorted(
            str(a) for a in utils.abstract(final, predicates))
        logging.info(
            "[%s] Forward-validate FINAL state%s:\n"
            "  goal atoms held:    %s\n"
            "  goal atoms MISSING: %s\n"
            "  abstract state:     %s\n"
            "  full features:      %s\n"
            "  full state:\n%s", run_id,
            " (goal reached)" if result.goal_reached else " (GOAL NOT "
            "REACHED)", held or "(none)", missing or "(none)", abstract_atoms,
            _fmt_state_features(final), final.pretty_str())

    if result.success:
        return True, ""

    # Diagnosis priority: execution failure > goal-not-reached > divergence.
    if result.first_failure_idx is not None:
        i = result.first_failure_idx
        outcome = result.steps[i]
        opt_str = (f"{outcome.option.name}"
                   f"({', '.join(o.name for o in outcome.option.objects)})")
        reason = outcome.failure_reason or "unknown reason"
        logging.info(
            "[%s] Forward-validate option failure at step %d (%s): %s", run_id,
            i, opt_str, reason)
        return False, (f"option execution failed at step {i} ({opt_str}): "
                       f"{reason}")
    if not result.goal_reached:
        goal_missing = sorted(
            str(a) for a in task.goal if not a.holds(result.final_state))
        return False, (f"goal not reached at final step "
                       f"(missing {goal_missing or '(none)'})")
    return False, first_div_msg or "validation failed"
