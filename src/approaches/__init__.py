"""Handle creation of approaches."""

import pkgutil
from typing import Set, List, TYPE_CHECKING
from gym.spaces import Box
from predicators.src.approaches.base_approach import BaseApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.structs import Predicate, ParameterizedOption, \
    Type, Task
from predicators.src import utils

__all__ = ["BaseApproach", "ApproachTimeout", "ApproachFailure"]


if not TYPE_CHECKING:
    # Load all modules so that utils.get_all_subclasses() works.
    for loader, module_name, _ in pkgutil.walk_packages(__path__):
        if "__init__" not in module_name:
            loader.find_module(module_name).load_module(module_name)


def create_approach(name: str, initial_predicates: Set[Predicate],
                    initial_options: Set[ParameterizedOption],
                    types: Set[Type], action_space: Box,
                    train_tasks: List[Task]) -> BaseApproach:
    """Create an approach given its name."""
    for cls in utils.get_all_subclasses(BaseApproach):
        if cls is not BaseApproach and cls.get_name() == name:
            approach = cls(initial_predicates, initial_options,
                           types, action_space, train_tasks)
            break
    else:
        raise NotImplementedError(f"Unknown approach: {name}")
    return approach
