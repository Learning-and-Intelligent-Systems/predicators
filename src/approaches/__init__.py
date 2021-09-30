"""Default imports for approaches folder.
"""

from typing import Set, Callable
import numpy as np
from numpy.typing import NDArray
from predicators.src.approaches.base_approach import BaseApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.approaches.random_actions_approach import \
    RandomActionsApproach
from predicators.src.approaches.tamp_approach import TAMPApproach
from predicators.src.approaches.oracle_approach import OracleApproach
from predicators.src.structs import State, Predicate, ParameterizedOption

Array = NDArray[np.float32]

__all__ = [
    "BaseApproach",
    "OracleApproach",
    "RandomActionsApproach",
    "TAMPApproach",
    "ApproachTimeout",
    "ApproachFailure",
]


def create_approach(name: str,
                    simulator: Callable[[State, Array], State],
                    initial_predicates: Set[Predicate],
                    initial_options: Set[ParameterizedOption],
                    action_space) -> BaseApproach:
    """Create an approach given its name.
    """
    if name == "Oracle":
        return OracleApproach(simulator, initial_predicates,
                              initial_options, action_space)
    if name == "Random Actions":
        return RandomActionsApproach(simulator, initial_predicates,
                                     initial_options, action_space)
    raise NotImplementedError(f"Unknown env: {name}")
