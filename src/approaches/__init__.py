"""Default imports for approaches folder.
"""

from typing import Set, Callable, List
from gym.spaces import Box
from predicators.src.approaches.base_approach import BaseApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.approaches.random_actions_approach import \
    RandomActionsApproach
from predicators.src.approaches.tamp_approach import TAMPApproach
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action


__all__ = [
    "BaseApproach",
    "OracleApproach",
    "RandomActionsApproach",
    "TAMPApproach",
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
    if name == "random":
        return RandomActionsApproach(simulator, initial_predicates,
                                     initial_options, types, action_space,
                                     train_tasks)
    raise NotImplementedError(f"Unknown env: {name}")
