"""A trivial "learning" approach that is actually random. For testing only.
"""

from typing import Callable
from predicators.src.approaches import BaseApproach
from predicators.src.structs import State, Task, Action


class TrivialLearningApproach(BaseApproach):
    """A trivial "learning" approach that is actually random. For testing only.
    """
    @property
    def is_learning_based(self) -> bool:
        return True

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        def _policy(_: State) -> Action:
            return Action(self._action_space.sample())
        return _policy
