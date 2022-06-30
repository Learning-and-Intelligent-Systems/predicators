"""An explorer that takes random options."""

from typing import List, Set

from gym.spaces import Box

from predicators.src.approaches.random_options_approach import \
    RandomOptionsApproach
from predicators.src.interaction import BaseExplorer
from predicators.src.structs import ExplorationStrategy, ParameterizedOption, \
    Predicate, Task, Type


class RandomOptionsExplorer(BaseExplorer):
    """Samples random options."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(predicates, options, types, action_space, train_tasks)
        # Set up a random options approach.
        self._random_options_approach = RandomOptionsApproach(
            predicates, options, types, action_space, train_tasks)

    @classmethod
    def get_name(cls) -> str:
        return "random_options"

    def get_exploration_strategy(self, task: Task,
                                 timeout: int) -> ExplorationStrategy:
        # Take random options.
        policy = self._random_options_approach.solve(task, timeout)
        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return policy, termination_function
