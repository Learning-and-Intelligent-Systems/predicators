"""Handle creation of explorers."""

import importlib
import pkgutil
from typing import TYPE_CHECKING, List, Set

from gym.spaces import Box

from predicators.src import utils
from predicators.src.interaction.base_explorer import BaseExplorer
from predicators.src.structs import ParameterizedOption, Predicate, Task, Type

__all__ = ["BaseExplorer"]

if not TYPE_CHECKING:
    # Load all modules so that utils.get_all_subclasses() works.
    for _, module_name, _ in pkgutil.walk_packages(__path__):
        if "__init__" not in module_name:
            # Important! We use an absolute import here to avoid issues
            # with isinstance checking when using relative imports.
            importlib.import_module(f"{__name__}.{module_name}")


def create_explorer(name: str, initial_predicates: Set[Predicate],
                    initial_options: Set[ParameterizedOption],
                    types: Set[Type], action_space: Box,
                    train_tasks: List[Task]) -> BaseExplorer:
    """Create an explorer given its name."""
    for cls in utils.get_all_subclasses(BaseExplorer):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            explorer = cls(initial_predicates, initial_options, types,
                           action_space, train_tasks)
            break
    else:
        raise NotImplementedError(f"Unknown explorer: {name}")
    return explorer
