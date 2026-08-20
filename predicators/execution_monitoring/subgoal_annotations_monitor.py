"""An execution monitor that checks plan-sketch subgoal annotations at option
boundaries and suggests replanning on divergence."""

import logging
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.structs import State, _Option


@dataclass
class SubgoalExecutionStatus:
    """Live execution status of an annotated plan, exported by an approach via
    ``get_execution_monitoring_info``.

    ``sketch`` items are duck-typed sketch steps exposing
    ``subgoal_atoms``, ``subgoal_neg_atoms`` and ``option`` (see
    ``agent_sdk.bilevel_sketch.SketchStep``); the type is kept loose so
    the monitoring layer does not import agent_sdk. The approach's
    dispensed policy mutates ``steps_initiated``/``current_option`` as
    it executes, so the monitor always sees the live values.
    """
    sketch: Sequence[Any]
    steps_initiated: int = 0
    current_option: Optional[_Option] = None


class SubgoalAnnotationsExecutionMonitor(BaseExecutionMonitor):
    """Suggest replanning when the step that just finished has a subgoal
    annotation that does not hold in the real state.

    The check happens at the exact option boundary: when the currently
    executing option's terminal condition is true in the given state,
    the step it completes is checked before the policy advances to the
    next option. Forward validation only proves a plan works in the
    option model; real execution can still diverge (e.g. a place whose
    drop-settle is chaotic lands off-target), after which the remaining
    open-loop plan is doomed — it burns the episode horizon waiting for
    effects that can no longer occur. Two boundaries are not caught:
    divergence that only manifests inside a non-terminating option, and
    Wait steps ended by the atom-change path in
    ``utils.option_policy_to_policy`` (those terminate exactly when
    their target atoms — derived from the same annotation — hold, so the
    check would pass anyway).
    """

    @classmethod
    def get_name(cls) -> str:
        return "subgoal_annotations"

    def step(self, state: State) -> bool:
        # No active annotated plan (e.g. exploration with an override
        # policy, or replanning disabled): never suggest replanning.
        if not self._approach_info:
            return False
        status = self._approach_info[0]
        if not isinstance(status, SubgoalExecutionStatus):
            return False
        option = status.current_option
        if option is None or status.steps_initiated <= 0:
            return False
        # Note: terminal() is also called by the policy machinery on
        # this same state; skill terminal functions read memory but do
        # not mutate it, so the double call is safe.
        if not option.terminal(state):
            return False
        step_idx = status.steps_initiated - 1
        step = status.sketch[step_idx]
        unsat = [
            str(a) for a in (step.subgoal_atoms or set()) if not a.holds(state)
        ]
        unsat += [
            f"NOT {a}" for a in (step.subgoal_neg_atoms or set())
            if a.holds(state)
        ]
        if not unsat:
            return False
        logging.info(
            "Subgoal divergence after step %d (%s): unsatisfied %s. "
            "Suggesting replan.", step_idx, step.option.name, sorted(unsat))
        return True
