"""An abstract approach that does TAMP to solve tasks. Uses the SeSamE
planning strategy: SEarch-and-SAMple planning, then Execution.
"""

import abc
from typing import Callable, Set, List, Sequence
from gym.spaces import Box
from predicators.src.approaches import BaseApproach, ApproachFailure
from predicators.src.planning import sesame_plan
from predicators.src.structs import State, Action, Task, Operator, \
    Predicate, ParameterizedOption, Type, _Option
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
        return option_plan_to_policy(plan)

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


def option_plan_to_policy(plan: Sequence[_Option]
                          ) -> Callable[[State], Action]:
    """Create a policy that executes the options in order.

    We may want to move this out of here later, but I'm leaving
    it for now, because it's annoying to import ApproachFailure
    in utils.py, for example.

    The logic for this is somewhat complicated because we want:
    * If an option's termination and initiation conditions are
      always true, we want the option to execute for one step.
    * After the first step that the option is executed, it
      should terminate as soon as it sees a state that is
      terminal; it should not take one more action after.
    """
    queue = list(plan)  # Don't modify plan, just in case
    initialized = False  # Special case first step
    def _policy(state: State) -> Action:
        nonlocal initialized
        # On the very first state, check initiation condition, and
        # take the action no matter what.
        if not initialized:
            if not queue:
                raise ApproachFailure("Ran out of options in the plan!")
            assert queue[0].initiable(state), "Unsound option plan"
            initialized = True
        elif queue[0].terminal(state):
            queue.pop(0)
            if not queue:
                raise ApproachFailure("Ran out of options in the plan!")
            assert queue[0].initiable(state), "Unsound option plan"
        return queue[0].policy(state)
    return _policy
