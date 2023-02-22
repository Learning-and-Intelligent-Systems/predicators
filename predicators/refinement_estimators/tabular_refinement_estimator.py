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

# Type of the (skeleton, atoms_sequence) key for cost dictionary
# which converts both of them to be immutable
CostDictKey = Tuple[Tuple[_GroundNSRT, ...],  # skeleton converted to tuple
                    Tuple[FrozenSet[GroundAtom], ...]  # atoms_sequence
                    ]


class TabularRefinementEstimator(BaseRefinementEstimator):
    """A refinement cost estimator that memorizes refinement data using a
    tabular method."""

    def __init__(self) -> None:
        super().__init__()
        # _cost_dict maps immutable skeleton atoms_sequence pair to float cost
        self._cost_dict: Optional[Dict[CostDictKey, float]] = None

    @classmethod
    def get_name(cls) -> str:
        return "tabular"

    @property
    def is_learning_based(self) -> bool:
        return True

    def get_cost(self, initial_state: State, skeleton: List[_GroundNSRT],
                 atoms_sequence: List[Set[GroundAtom]]) -> float:
        assert self._cost_dict is not None, "Need to train"
        key = self._immutable_cost_dict_key(skeleton, atoms_sequence)
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
            key = self._immutable_cost_dict_key(skeleton, atoms_sequence)
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

    @staticmethod
    def _immutable_cost_dict_key(
            skeleton: List[_GroundNSRT],
            atoms_sequence: List[Set[GroundAtom]]) -> CostDictKey:
        """Converts a skeleton and atoms_sequence into immutable types to use
        as a key for the cost dictionary."""
        return (tuple(skeleton),
                tuple(frozenset(atoms) for atoms in atoms_sequence))

    def save_model(self, filepath: Path) -> None:
        with open(filepath, "wb") as f:
            pkl.dump(self._cost_dict, f)

    def load_model(self, filepath: Path) -> None:
        with open(filepath, "rb") as f:
            self._cost_dict = pkl.load(f)
