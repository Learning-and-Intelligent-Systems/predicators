"""An abstract approach that does TAMP to solve tasks. Uses the SeSamE
planning strategy: SEarch-and-SAMple planning, then Execution.
"""

import abc
from typing import Callable, Set
from gym.spaces import Box
from predicators.src.approaches import BaseApproach, ApproachFailure
from predicators.src.planning import sesame_plan
from predicators.src.structs import State, Action, Task, NSRT, \
    Predicate, ParameterizedOption, Type
from predicators.src.option_model import create_option_model
from predicators.src.settings import CFG
from predicators.src import utils


class TAMPApproach(BaseApproach):
    """TAMP approach.
    """
    def __init__(self, simulator: Callable[[State, Action], State],
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box) -> None:
        super().__init__(simulator, initial_predicates, initial_options, types,
                         action_space)
        self._option_model = create_option_model(
            CFG.option_model_name, self._simulator)
        self._num_calls = 0

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        seed = self._seed+self._num_calls  # ensure random over successive calls
        plan, metrics = sesame_plan(task, self._option_model,
                                    self._get_current_nsrts(),
                                    self._get_current_predicates(),
                                    timeout, seed)
        for metric in ["num_skeletons_optimized",
                       "num_failures_discovered",
                       "num_nodes_expanded",
                       "heuristic1_time",
                       "heuristic2_time",
                       "plan_length"]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        option_policy = utils.option_plan_to_policy(plan)
        def _policy(s: State) -> Action:
            try:
                return option_policy(s)
            except utils.OptionPlanExhausted:
                raise ApproachFailure("Option plan exhausted.")
        return _policy

    @abc.abstractmethod
    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs.
        """
        raise NotImplementedError("Override me!")

    def _get_current_predicates(self) -> Set[Predicate]:
        """Get the current set of predicates.
        Defaults to initial predicates.
        """
        return self._initial_predicates
