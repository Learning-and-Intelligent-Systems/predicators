"""Base class for a refinement cost estimator."""

import abc
from typing import List, Set

from predicators.structs import GroundAtom, _GroundNSRT


class BaseRefinementEstimator(abc.ABC):
    """Base refinement cost estimator."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this refinement cost estimator, for future
        use as the argument to `--refinement_estimator`."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_cost(self, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        """Return an estimated cost for a proposed high-level skeleton."""
        raise NotImplementedError("Override me!")
