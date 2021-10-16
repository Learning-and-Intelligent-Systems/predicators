"""Default imports for approaches folder.
"""

from typing import Set, Callable, List
from gym.spaces import Box
from predicators.src.approaches.base_approach import BaseApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.approaches.random_actions_approach import \
    RandomActionsApproach
from predicators.src.approaches.random_options_approach import \
    RandomOptionsApproach
from predicators.src.approaches.tamp_approach import TAMPApproach
from predicators.src.approaches.trivial_learning_approach import \
    TrivialLearningApproach
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src.approaches.operator_learning_approach import \
    OperatorLearningApproach
from predicators.src.approaches.interactive_learning_approach import \
    InteractiveLearningApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action


__all__ = [
    "BaseApproach",
    "OracleApproach",
    "RandomActionsApproach",
    "RandomOptionsApproach",
    "TAMPApproach",
    "TrivialLearningApproach",
    "OperatorLearningApproach",
    "InteractiveLearningApproach",
    "ApproachTimeout",
    "ApproachFailure",
]


def create_approach(name: str,
                    simulator: Callable[[State, Action], State],
                    initial_predicates: Set[Predicate],
                    initial_options: Set[ParameterizedOption],
                    types: Set[Type],
                    action_space: Box,
                    train_tasks: List[Task]) -> BaseApproach:
    """Create an approach given its name.
    """
    if name == "oracle":
        return OracleApproach(simulator, initial_predicates,
                              initial_options, types, action_space,
                              train_tasks)
    if name == "random_actions":
        return RandomActionsApproach(simulator, initial_predicates,
                                     initial_options, types, action_space,
                                     train_tasks)
    if name == "random_options":
        return RandomOptionsApproach(simulator, initial_predicates,
                                     initial_options, types, action_space,
                                     train_tasks)
    if name == "trivial_learning":
        return TrivialLearningApproach(simulator, initial_predicates,
                                       initial_options, types, action_space,
                                       train_tasks)
    if name == "operator_learning":
        return OperatorLearningApproach(simulator, initial_predicates,
                                        initial_options, types, action_space,
                                        train_tasks)
    if name == "interactive_learning":
        return InteractiveLearningApproach(simulator, initial_predicates,
                                           initial_options, types, action_space,
                                           train_tasks)
    raise NotImplementedError(f"Unknown env: {name}")
