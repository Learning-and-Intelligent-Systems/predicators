"""Base class for a bridge policy."""

import abc
from typing import Callable, List, Set, Tuple

import numpy as np

from predicators import utils
from predicators.settings import CFG
from predicators.structs import NSRT, BridgeDataset, BridgePolicyFailure, \
    GroundAtom, ParameterizedOption, Predicate, State, Type, _Option


class BridgePolicyDone(Exception):
    """Raised when a bridge policy is done executing."""


class BaseBridgePolicy(abc.ABC):
    """Base bridge policy."""

    def __init__(self, types: Set[Type], predicates: Set[Predicate],
                 options: Set[ParameterizedOption], nsrts: Set[NSRT]) -> None:
        self._types = types
        self._predicates = predicates
        self._options = options
        self._nsrts = nsrts
        self._rng = np.random.default_rng(CFG.seed)
        self._state_history: List[State] = []
        self._atoms_history: List[Set[GroundAtom]] = []
        self._option_history: List[_Option] = []
        self._failure_history: List[Tuple[int, BridgePolicyFailure]] = []

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this bridge policy, for future use as the
        argument to `--bridge_policy`."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def is_learning_based(self) -> bool:
        """Does the bridge policy learn from interaction data?"""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_option_policy(self) -> Callable[[State], _Option]:
        """The main method creating the bridge policy."""
        raise NotImplementedError("Override me!")

    def reset(self) -> None:
        """Called at the beginning of a new task."""
        self._state_history = []
        self._atoms_history = []
        self._option_history = []
        self._failure_history = []

    def record_failure(self, failure: BridgePolicyFailure) -> None:
        """Called when a failure is detected."""
        t = len(self._option_history)
        self._failure_history.append((t, failure))

    def record_state_option(self, state: State, option: _Option) -> None:
        """Called whenever a new option is selected by either the planner or
        the bridge policy itself."""
        self._state_history.append(state)
        self._atoms_history.append(utils.abstract(state, self._predicates))
        self._option_history.append(option)

    def get_internal_state(
        self
    ) -> Tuple[List[State], List[Set[GroundAtom]], List[_Option], List[Tuple[
            int, BridgePolicyFailure]]]:
        """Get that which is sufficient to describe the internal state."""
        return (list(self._state_history), list(self._atoms_history),
                list(self._option_history), list(self._failure_history))

    def set_internal_state(
        self, internal_state: Tuple[List[State], List[Set[GroundAtom]],
                                    List[_Option],
                                    List[Tuple[int, BridgePolicyFailure]]]
    ) -> None:
        """Set the bridge policy's internal state."""
        self._state_history, self._atoms_history, self._option_history, \
            self._failure_history = internal_state

    def learn_from_demos(self, dataset: BridgeDataset) -> None:
        """For learning-based approaches, learn whatever is needed from the
        given dataset.

        By default, nothing is learned. Subclasses may override.
        """
