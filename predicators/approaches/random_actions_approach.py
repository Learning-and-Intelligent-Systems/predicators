"""An approach that just takes random low-level actions."""

from typing import Callable

from predicators.approaches import BaseApproach
from predicators.structs import Action, State, Task


class RandomActionsApproach(BaseApproach):
    """Samples random low-level actions."""

    @classmethod
    def get_name(cls) -> str:
        return "random_actions"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def _policy(_: State) -> Action:
            return Action(self._action_space.sample())

        return _policy
