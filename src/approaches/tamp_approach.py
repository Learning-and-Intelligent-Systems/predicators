"""An abstract approach that does TAMP to solve tasks. Uses the SeSamE
planning strategy: SEarch-and-SAMple planning, then Execution.
"""

import abc
from typing import Callable, Set, Any
from predicators.src.approaches import BaseApproach, ApproachFailure
from predicators.src.planning import sesame_plan
from predicators.src.structs import State, Action, Task, Operator


class TAMPApproach(BaseApproach):
    """TAMP approach.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._num_calls = 0

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        seed = self._seed+self._num_calls  # ensure random over successive calls
        plan, metrics = sesame_plan(task, self._simulator,
                                    self._get_current_operators(),
                                    self._initial_predicates, timeout, seed)
        for metric in ["num_skeletons_optimized",
                       "num_failures_discovered",
                       "plan_length"]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        def _policy(_: State) -> Action:
            if not plan:
                raise ApproachFailure("Finished executing plan!")
            return plan.pop(0)
        return _policy

    @abc.abstractmethod
    def _get_current_operators(self) -> Set[Operator]:
        """Get the current set of operators.
        """
        raise NotImplementedError("Override me!")
