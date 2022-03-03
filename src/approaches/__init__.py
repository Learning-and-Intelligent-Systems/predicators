"""Default imports for approaches folder."""

from typing import Set, Callable, List
from gym.spaces import Box
from predicators.src.approaches.base_approach import BaseApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.approaches.random_actions_approach import \
    RandomActionsApproach
from predicators.src.approaches.random_options_approach import \
    RandomOptionsApproach
from predicators.src.approaches.tamp_approach import TAMPApproach
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.src.approaches.grammar_search_invention_approach import \
    GrammarSearchInventionApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action

__all__ = [
    "BaseApproach",
    "OracleApproach",
    "RandomActionsApproach",
    "RandomOptionsApproach",
    "TAMPApproach",
    "NSRTLearningApproach",
    "InteractiveLearningApproach",
    "GrammarSearchInventionApproach",
    "ApproachTimeout",
    "ApproachFailure",
]


def create_approach(name: str, initial_predicates: Set[Predicate],
                    initial_options: Set[ParameterizedOption],
                    types: Set[Type], action_space: Box,
                    train_tasks: List[Task]) -> BaseApproach:
    """Create an approach given its name."""
    if name == "oracle":
        return OracleApproach(initial_predicates, initial_options, types,
                              action_space, train_tasks)
    if name == "random_actions":
        return RandomActionsApproach(initial_predicates, initial_options,
                                     types, action_space, train_tasks)
    if name == "random_options":
        return RandomOptionsApproach(initial_predicates, initial_options,
                                     types, action_space, train_tasks)
    if name == "nsrt_learning":
        return NSRTLearningApproach(initial_predicates, initial_options, types,
                                    action_space, train_tasks)
    if name == "interactive_learning":
        return InteractiveLearningApproach(initial_predicates, initial_options,
                                           types, action_space, train_tasks)
    if name == "grammar_search_invention":
        return GrammarSearchInventionApproach(initial_predicates,
                                              initial_options, types,
                                              action_space, train_tasks)
    raise NotImplementedError(f"Unknown approach: {name}")
