"""Base class for an explorer."""

import abc
from typing import List, Set

import numpy as np
from gym.spaces import Box

from predicators.src.settings import CFG
from predicators.src.structs import ExplorationStrategy, ParameterizedOption, \
    Predicate, Task, Type


class BaseExplorer(abc.ABC):
    """Creates a policy and termination function for exploring in a task.

    The explorer is created at the beginning of every interaction cycle
    with the latest predicates and options.
    """

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
    def get_exploration_strategy(
        self,
        train_task_idx: int,
        timeout: int,
    ) -> ExplorationStrategy:
        """Given a train task idx, create an ExplorationStrategy, which is a
        tuple of a policy and a termination function."""
        raise NotImplementedError("Override me!")

    def _set_seed(self, seed: int) -> None:
        """Reset seed and rng."""
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
