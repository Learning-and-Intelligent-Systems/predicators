"""Base class for an approach."""

import abc
from collections import defaultdict
from typing import Set, Callable, List
import numpy as np
from gym.spaces import Box
from predicators.src.structs import State, Task, Predicate, Type, \
    ParameterizedOption, Action, Dataset, Metrics


class BaseApproach:
    """Base approach."""

    def __init__(self, simulator: Callable[[State, Action], State],
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box) -> None:
        """All approaches are initialized with only the necessary information
        about the environment."""
        self._simulator = simulator
        self._initial_predicates = initial_predicates
        self._initial_options = initial_options
        self._types = types
        self._action_space = action_space
        self._metrics: Metrics = defaultdict(float)
        self.seed(0)

    @property
    @abc.abstractmethod
    def is_learning_based(self) -> bool:
        """Does the approach learn from the training tasks?"""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Return a policy for the given task, within the given number of
        seconds.

        A policy maps states to low-level actions.
        """
        raise NotImplementedError("Override me!")

    def solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Light wrapper around the abstract self._solve().

        Checks that actions are in the action space.
        """
        pi = self._solve(task, timeout)

        def _policy(state: State) -> Action:
            assert isinstance(state, State)
            act = pi(state)
            assert self._action_space.contains(act.arr)
            return act

        return _policy

    def seed(self, seed: int) -> None:
        """Reset seed and rng."""
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._action_space.seed(seed)

    def learn_from_offline_dataset(self, dataset: Dataset,
                                   train_tasks: List[Task]) -> None:
        """For learning-based approaches, learn whatever is needed from the
        given dataset, which was generated from the given train_tasks. Also,
        should save whatever is necessary to load() later.

        Note: this is not an abc.abstractmethod because it does
        not need to be defined by the subclasses. (mypy complains
        if you try to instantiate a subclass with an undefined abc).
        """

    def load(self) -> None:
        """Load anything from CFG.get_save_path_str().

        Only called if self.is_learning_based.
        """

    @property
    def metrics(self) -> Metrics:
        """Return a dictionary of metrics, which can hold arbitrary evaluation
        information about the performance of this approach."""
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset the metrics dictionary."""
        self._metrics = defaultdict(float)


class ApproachTimeout(Exception):
    """Exception raised when approach.solve() times out."""


class ApproachFailure(Exception):
    """Exception raised when approach.solve() fails to compute a policy."""
