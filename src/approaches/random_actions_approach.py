"""An approach that just takes random low-level actions.
"""
from typing import Callable
from numpy.typing import NDArray
import numpy as np
from predicators.src.approaches.base_approach import BaseApproach
from predicators.src.structs import State, Task

Array = NDArray[np.float32]


class RandomActionsApproach(BaseApproach):
    """Samples random low-level actions.
    """
    @property
    def is_learning_based(self):
        """Not learning-based."""
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Array]:
        """Return a policy for the given task, within the given number of
        seconds. A policy maps states to low-level actions.
        """
        def _policy(_):
            return self._action_space.sample()
        return _policy
