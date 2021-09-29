"""Base class for an approach, which learns operators and predicates.
"""

import abc
import itertools
from typing import Collection, Callable, Set
import numpy as np
from numpy.typing import ArrayLike
from gym.spaces import Box  # type: ignore
from predicators.src.structs import State, Task, Predicate, GroundAtom, \
    ParameterizedOption, GroundAtom


class BaseApproach:
    """Base approach.
    """
    def __init__(self, simulator: Callable[[State, ArrayLike], State],
                 predicates: Collection[Predicate],
                 options: Collection[ParameterizedOption],
                 action_space: Box):
        """All approaches are initialized with a simulator, initial predicates,
        the parameterized options, and the action space.
        """
        self._simulator = simulator
        self._predicates = predicates
        self._options = options
        self._action_space = action_space
        self.seed(0)

    @abc.abstractmethod
    def _solve(self, task: Task, timeout: int) -> Callable[[State], ArrayLike]:
        """Return a policy for the given task, within the given number of
        seconds. A policy maps states to low-level actions.
        """
        raise NotImplementedError("Override me!")

    def solve(self, task: Task, timeout: int) -> Callable[[State], ArrayLike]:
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

    def _abstract(self, state: State) -> Set[GroundAtom]:
        """Get the atomic representation of this state (i.e., a set
        of ground atoms).
        """
        atoms = set()
        for pred in self._predicates:
            domains = []
            for var_type in pred.types:
                domains.append([obj for obj in state if obj.type == var_type])
            for choice in itertools.product(*domains):
                if len(choice) != len(set(choice)):
                    continue  # ignore duplicate arguments
                if pred.holds(state, choice):
                    atoms.add(GroundAtom(pred, choice))
        return atoms


class ApproachTimeout(Exception):
    """Exception raised when approach.solve() times out.
    """
