"""Base class for an approach, which can learn operators, predicates,
and/or options.
"""

import abc
from typing import Collection, Callable
import numpy as np
from numpy.typing import NDArray
from gym.spaces import Box  # type: ignore
from predicators.src.structs import State, Task, Predicate, ParameterizedOption

Array = NDArray[np.float32]


class BaseApproach:
    """Base approach.
    """
    def __init__(self, simulator: Callable[[State, Array], State],
                 initial_predicates: Collection[Predicate],
                 initial_options: Collection[ParameterizedOption],
                 action_space: Box):
        """All approaches are initialized with a simulator, initial predicates,
        initial parameterized options, and the action space.
        """
        self._simulator = simulator
        self._initial_predicates = initial_predicates
        self._initial_options = initial_options
        self._action_space = action_space
        self.seed(0)

    @abc.abstractmethod
    def _solve(self, task: Task, timeout: int) -> Callable[[State], Array]:
        """Return a policy for the given task, within the given number of
        seconds. A policy maps states to low-level actions.
        """
        raise NotImplementedError("Override me!")

    def solve(self, task: Task, timeout: int) -> Callable[[State], Array]:
        """Light wrapper around the abstract self._solve(). Checks that
        actions are in the action space.
        """
        pi = self._solve(task, timeout)
        def _policy(state):
            assert isinstance(state, State)
            act = pi(state)
            assert self._action_space.contains(act)
            return act
        return _policy

    def seed(self, seed: int):
        """Reset seed and rng.
        """
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)


class ApproachTimeout(Exception):
    """Exception raised when approach.solve() times out.
    """
