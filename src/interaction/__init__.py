"""Handle creation of explorers."""

import importlib
import pkgutil
from typing import TYPE_CHECKING, Callable, List, Optional, Set

from gym.spaces import Box

from predicators.src import utils
from predicators.src.interaction.base_explorer import BaseExplorer
from predicators.src.interaction.bilevel_planning_explorer import \
    BilevelPlanningExplorer
from predicators.src.interaction.glib_explorer import GLIBExplorer
from predicators.src.option_model import _OptionModelBase
from predicators.src.structs import NSRT, GroundAtom, ParameterizedOption, \
    Predicate, Task, Type

__all__ = ["BaseExplorer"]

if not TYPE_CHECKING:
    # Load all modules so that utils.get_all_subclasses() works.
    for _, module_name, _ in pkgutil.walk_packages(__path__):
        if "__init__" not in module_name:
            # Important! We use an absolute import here to avoid issues
            # with isinstance checking when using relative imports.
            importlib.import_module(f"{__name__}.{module_name}")


def create_explorer(
    name: str,
    initial_predicates: Set[Predicate],
    initial_options: Set[ParameterizedOption],
    types: Set[Type],
    action_space: Box,
    train_tasks: List[Task],
    nsrts: Optional[Set[NSRT]] = None,
    option_model: Optional[_OptionModelBase] = None,
    atom_score_fn: Optional[Callable[[Set[GroundAtom]], float]] = None,
) -> BaseExplorer:
    """Create an explorer given its name."""
    for cls in utils.get_all_subclasses(BaseExplorer):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            # Special case GLIB because it uses an atom score function.
            if issubclass(cls, GLIBExplorer):
                assert nsrts is not None
                assert option_model is not None
                assert atom_score_fn is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks, nsrts, option_model,
                               atom_score_fn)
            # Bilevel planning approaches use NSRTs and an option model.
            elif issubclass(cls, BilevelPlanningExplorer):
                assert nsrts is not None
                assert option_model is not None
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks, nsrts, option_model)
            else:
                explorer = cls(initial_predicates, initial_options, types,
                               action_space, train_tasks)
            break
    else:
        raise NotImplementedError(f"Unknown explorer: {name}")
    return explorer
