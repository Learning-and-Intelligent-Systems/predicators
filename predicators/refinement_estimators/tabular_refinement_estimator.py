"""A tabular refinement cost estimator that memorizes a mapping from skeleton
and atoms_sequence to average refinement time."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import dill as pkl
import numpy as np

from predicators.refinement_estimators import BaseRefinementEstimator
from predicators.settings import CFG
from predicators.structs import GroundAtom, RefinementDatapoint, State, \
    _GroundNSRT


class TabularRefinementEstimator(BaseRefinementEstimator):
    """A refinement cost estimator that memorizes refinement data using a
    tabular method."""

    def __init__(self) -> None:
        super().__init__()
        self._cost_dict: Optional[Dict[Tuple[Tuple[_GroundNSRT, ...],
                                             Tuple[FrozenSet[GroundAtom],
                                                   ...]], float]] = None

    @classmethod
    def get_name(cls) -> str:
        return "tabular"

    @property
    def is_learning_based(self) -> bool:
        return True

    def get_cost(self, initial_state: State, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        del initial_state  # unused
        assert self._cost_dict is not None, "Need to train"
        key = (tuple(skeleton),
               tuple(frozenset(atoms) for atoms in atoms_sequence))
        # Try to find key in dictionary, otherwise return infinity
        cost = self._cost_dict.get(key, float('inf'))
        return cost

    def train(self, data: List[RefinementDatapoint]) -> None:
        """Train the tabular refinement estimator on data by computing average
        refinement time per (skeleton, atoms_sequence) pair."""
        grouped_data = defaultdict(list)
        # Go through data and group them by skeleton
        for _, skeleton, atoms_sequence, succeeded, refinement_time in data:
            # Convert skeleton and atoms_sequence into an immutable dict key
            key = (tuple(skeleton),
                   tuple(frozenset(atoms) for atoms in atoms_sequence))
            value = refinement_time
            # Add failed refinement penalty to the value if failure occurred
            if not succeeded:
                value += CFG.refinement_data_failed_refinement_penalty
            grouped_data[key].append(value)
        # Compute average time for each (skeleton, atoms_sequence) key
        processed_data = {
            key: float(np.mean(times))
            for key, times in grouped_data.items()
        }
        self._cost_dict = processed_data

    def save_state(self, filepath: Path) -> None:  # pragma: no cover
        with open(filepath, "wb") as f:
            pkl.dump(self._cost_dict, f)

    def load_state(self, filepath: Path) -> None:  # pragma: no cover
        with open(filepath, "rb") as f:
            self._cost_dict = pkl.load(f)
