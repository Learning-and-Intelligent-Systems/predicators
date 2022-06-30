"""Base class for an explorer."""

import abc
from typing import Callable, List, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators.src.settings import CFG
from predicators.src.structs import Action, ParameterizedOption, Predicate, \
    State, Task, Type


class BaseExplorer(abc.ABC):
    """Creates a policy and termination function for exploring in a task."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        self._predicates = predicates
        self._options = options
        self._types = types
        self._action_space = action_space
        self._train_tasks = train_tasks
        self._set_seed(CFG.seed)

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this explorer."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def solve(
        self,
        task: Task,
        timeout: int,
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Given a task, create a policy and termination function."""
        raise NotImplementedError("Override me!")

    def _set_seed(self, seed: int) -> None:
        """Reset seed and rng."""
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
