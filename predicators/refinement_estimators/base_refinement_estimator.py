"""Base class for a refinement cost estimator."""

import abc
from pathlib import Path
from typing import List, Set

from predicators.structs import GroundAtom, State, _GroundNSRT


class BaseRefinementEstimator(abc.ABC):
    """Base refinement cost estimator."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this refinement cost estimator, for future
        use as the argument to `--refinement_estimator`."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def is_learning_based(self) -> bool:
        """Does the estimator learn from training tasks?"""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_cost(self, initial_state: State, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        """Return an estimated cost for a proposed high-level skeleton."""
        raise NotImplementedError("Override me!")

    def train(self, data: List) -> None:    # pragma: no cover
        """Train the estimator on given training data.

        Only called if is_learning_based is True.
        """
        pass

    def save_state(self, filepath: Path) -> None:    # pragma: no cover
        """Save the training state of the approach to a file.

        Only called if is_learning_based is True.
        """
        pass

    def load_state(self, filepath: Path) -> None:    # pragma: no cover
        """Load the training state of the approach from a file.

        Only called if is_learning_based is True.
        """
        pass
