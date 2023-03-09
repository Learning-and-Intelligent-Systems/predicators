"""A tabular refinement cost estimator that memorizes a mapping from skeleton
and atoms_sequence to average refinement time."""

from collections import defaultdict
from typing import List

import numpy as np

from predicators.refinement_estimators.per_skeleton_estimator import \
    PerSkeletonRefinementEstimator
from predicators.settings import CFG
from predicators.structs import RefinementDatapoint, Task


class TabularRefinementEstimator(PerSkeletonRefinementEstimator[float]):
    """A refinement cost estimator that memorizes refinement data using a
    tabular method."""

    @classmethod
    def get_name(cls) -> str:
        return "tabular"

    def _model_predict(self, model: float, initial_task: Task) -> float:
        return model

    def train(self, data: List[RefinementDatapoint]) -> None:
        """Train the tabular refinement estimator on data by computing average
        refinement time per (skeleton, atoms_sequence) pair."""
        grouped_data = defaultdict(list)
        # Go through data and group them by skeleton
        for _, skeleton, atoms_sequence, succeeded, refinement_time in data:
            # Convert skeleton and atoms_sequence into an immutable dict key
            key = self._immutable_model_dict_key(skeleton, atoms_sequence)
            value = sum(refinement_time)
            # Add failed refinement penalty to the value if failure occurred
            if not succeeded:
                value += CFG.refinement_data_failed_refinement_penalty
            grouped_data[key].append(value)
        # Compute average time for each (skeleton, atoms_sequence) key
        processed_data = {
            key: float(np.mean(times))
            for key, times in grouped_data.items()
        }
        self._model_dict = processed_data
