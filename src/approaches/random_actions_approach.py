"""An approach that just takes random low-level actions."""

from typing import Callable
from predicators.src.approaches import BaseApproach
from predicators.src.structs import State, Task, Action


class RandomActionsApproach(BaseApproach):
    """Samples random low-level actions."""

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def policy(_: State) -> Action:
            return Action(self._action_space.sample())

        return policy
