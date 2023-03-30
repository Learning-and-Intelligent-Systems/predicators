"""Base class for a bridge policy."""

import abc
from typing import Callable, Set

import numpy as np

from predicators.settings import CFG
from predicators.structs import NSRT, Action, BridgeDataset, Predicate, \
    State, _GroundNSRT


class BridgePolicyDone(Exception):
    """Raised when a bridge policy is done executing."""


class BaseBridgePolicy(abc.ABC):
    """Base bridge policy."""

    def __init__(self, predicates: Set[Predicate], nsrts: Set[NSRT]) -> None:
        self._predicates = predicates
        self._nsrts = nsrts
        self._rng = np.random.default_rng(CFG.seed)

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
    def get_policy(self,
                   failed_nsrt: _GroundNSRT) -> Callable[[State], Action]:
        """The main method creating the bridge policy."""
        raise NotImplementedError("Override me!")

    def learn_from_demos(self, dataset: BridgeDataset) -> None:
        """For learning-based approaches, learn whatever is needed from the
        given dataset.

        The dataset consists of tuples of failed_nsrt,
        ground_nsrt_bridge, atoms_bridge, states_bridge.
        """
