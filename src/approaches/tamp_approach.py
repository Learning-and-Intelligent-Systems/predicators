"""An abstract approach that does TAMP to solve tasks. Uses the SeSamE
planning strategy: SEarch-and-SAMple planning, then Execution.
"""

import abc
from typing import Callable, Set, List
from gym.spaces import Box
from predicators.src.approaches import BaseApproach, ApproachFailure
from predicators.src.planning import sesame_plan
from predicators.src.structs import State, Action, Task, Operator, \
    Predicate, ParameterizedOption, Type
from predicators.src.option_model import create_option_model
from predicators.src.settings import CFG


class TAMPApproach(BaseApproach):
    """TAMP approach.
    """
    def __init__(self, simulator: Callable[[State, Action], State],
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task]) -> None:
        super().__init__(simulator, initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._option_model = create_option_model(
            CFG.option_model_name, self._simulator)
        self._num_calls = 0

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        seed = self._seed+self._num_calls  # ensure random over successive calls
        plan, metrics = sesame_plan(task, self._option_model,
                                    self._get_current_operators(),
                                    self._get_current_predicates(),
                                    timeout, seed)
        for metric in ["num_skeletons_optimized",
                       "num_failures_discovered",
                       "plan_length"]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        def _policy(state: State) -> Action:
            if not plan:
                raise ApproachFailure("Finished executing plan!")
            cur_option = plan[0]
            assert cur_option.initiable(state), "Unsound planner output"
            act = cur_option.policy(state)
            if cur_option.terminal(state):
                plan.pop(0)  # this option is exhausted, continue to next
            return act
        return _policy

    @abc.abstractmethod
    def _get_current_operators(self) -> Set[Operator]:
        """Get the current set of operators.
        """
        raise NotImplementedError("Override me!")

    def _get_current_predicates(self) -> Set[Predicate]:
        """Get the current set of predicates.
        Defaults to initial predicates.
        """
        return self._initial_predicates
