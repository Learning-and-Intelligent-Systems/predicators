"""Base class for an approach."""

import abc
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.src.settings import CFG
from predicators.src.structs import Action, Dataset, InteractionRequest, \
    InteractionResult, Metrics, ParameterizedOption, Predicate, State, Task, \
    Type
from predicators.src.utils import ExceptionWithInfo


class BaseApproach(abc.ABC):
    """Base approach."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        """All approaches are initialized with only the necessary information
        about the environment."""
        self._initial_predicates = initial_predicates
        self._initial_options = initial_options
        self._types = types
        self._action_space = action_space
        self._train_tasks = train_tasks
        self._metrics: Metrics = defaultdict(float)
        self._set_seed(CFG.seed)

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this approach, used as the argument to
        `--approach`."""
        raise NotImplementedError("Override me!")

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

    def _set_seed(self, seed: int) -> None:
        """Reset seed and rng."""
        self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._action_space.seed(seed)

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """For learning-based approaches, learn whatever is needed from the
        given dataset.

        Also, save the results of learning so they can be loaded in the
        future via load() with online_learning_cycle = None.
        """

    def load(self, online_learning_cycle: Optional[int]) -> None:
        """Load anything from CFG.get_approach_load_path_str().

        Only called if self.is_learning_based. If online_learning_cycle
        is None, then load the results of learn_from_offline_dataset().

        Otherwise, load the results of the ith call (zero-indexed) to
        learn_from_interaction_results().
        """

    def get_interaction_requests(self) -> List[InteractionRequest]:
        """Based on any learning that has previously occurred, create a list of
        InteractionRequest objects to give back to the environment.

        The results of these requests will define the data that is
        received the next learning cycle, when
        learn_from_interaction_results() is called.
        """
        _ = self  # unused, but maybe useful for subclasses
        return []

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        """Given a list of results of the requests returned by
        get_interaction_requests(), learn whatever.

        Also, save the results of learning so they can be loaded in the
        future via load() with non-None values of online_learning_cycle.
        """

    @property
    def metrics(self) -> Metrics:
        """Return a dictionary of metrics, which can hold arbitrary evaluation
        information about the performance of this approach."""
        return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset the metrics dictionary."""
        self._metrics = defaultdict(float)


class ApproachTimeout(ExceptionWithInfo):
    """Exception raised when approach.solve() times out."""


class ApproachFailure(ExceptionWithInfo):
    """Exception raised when approach.solve() fails to compute a policy."""
