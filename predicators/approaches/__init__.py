"""Handle creation of approaches."""

from typing import List, Set
from typing import Type as TypingType

from gym.spaces import Box

from predicators import utils
from predicators.approaches.base_approach import ApproachFailure, \
    ApproachTimeout, BaseApproach, BaseApproachWrapper
from predicators.structs import ParameterizedOption, Predicate, Task, Type,\
    ConceptPredicate

__all__ = ["BaseApproach", "ApproachTimeout", "ApproachFailure"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def _get_approach_cls_from_name(name: str) -> TypingType[BaseApproach]:
    for cls in utils.get_all_subclasses(BaseApproach):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            return cls
    raise NotImplementedError(f"Unknown approach: {name}")


def _get_wrapper_cls_from_name(name: str) -> TypingType[BaseApproachWrapper]:
    for cls in utils.get_all_subclasses(BaseApproachWrapper):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            return cls
    raise NotImplementedError(f"Unknown wrapper approach: {name}")


def create_approach(name: str, initial_predicates: Set[Predicate],
                    initial_options: Set[ParameterizedOption],
                    types: Set[Type], action_space: Box,
                    train_tasks: List[Task]) -> BaseApproach:
    """Create an approach given its name."""
    # Handle approach wrappers.
    if "[" in name:
        idx = name.index("[")
        wrapper_name = name[:idx]
        assert name.endswith("]")
        base_name = name[idx + 1:-1]
        base_approach = create_approach(base_name, initial_predicates,
                                        initial_options, types, action_space,
                                        train_tasks)
        # Find wrapper.
        wrapper_cls = _get_wrapper_cls_from_name(wrapper_name)
        return wrapper_cls(base_approach, initial_predicates, 
                           initial_options,
                           types, action_space, train_tasks,
                           )

    # Handle main approaches.
    cls = _get_approach_cls_from_name(name)
    return cls(initial_predicates, 
               initial_options, types, action_space,
               train_tasks,
               )
