"""A tabular refinement cost estimator that memorizes a mapping from skeleton
and atoms_sequence to average refinement time."""

from collections import defaultdict
from typing import List, Tuple

import numpy as np

from predicators.refinement_estimators.per_skeleton_estimator import \
    PerSkeletonRefinementEstimator
from predicators.settings import CFG
from predicators.structs import RefinementDatapoint, Task


class TabularRefinementEstimator(PerSkeletonRefinementEstimator[Tuple[float,
                                                                      float]]):
    """A refinement cost estimator that memorizes refinement data using a
    tabular method."""

    @classmethod
    def get_name(cls) -> str:
        return "tabular"

    def _model_predict(self, model: Tuple[float, float],
                       initial_task: Task) -> float:
        refinement_time, low_level_count = model
        cost = refinement_time
        if CFG.refinement_data_include_execution_cost:
            cost += (low_level_count *
                     CFG.refinement_data_low_level_execution_cost)
        return cost

    def train(self, data: List[RefinementDatapoint]) -> None:
        """Train the tabular refinement estimator on data by computing average
        refinement time per (skeleton, atoms_sequence) pair."""
        grouped_times = defaultdict(list)
        grouped_counts = defaultdict(list)
        # Go through data and group them by skeleton
        for (_, skeleton, atoms_sequence, succeeded, refinement_time,
             low_level_count) in data:
            # Convert skeleton and atoms_sequence into an immutable dict key
            key = self._immutable_model_dict_key(skeleton, atoms_sequence)
            target_time = sum(refinement_time)
            # Add failed refinement penalty to the value if failure occurred
            if not succeeded:
                target_time += CFG.refinement_data_failed_refinement_penalty
            grouped_times[key].append(target_time)
            grouped_counts[key].append(sum(low_level_count))
        # Compute average time for each (skeleton, atoms_sequence) key
        processed_data = {
            key: (float(np.mean(times)), float(np.mean(grouped_counts[key])))
            for key, times in grouped_times.items()
        }
        self._model_dict = processed_data
