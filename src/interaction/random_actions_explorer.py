"""An explorer that just takes random low-level actions."""

from typing import Callable, Tuple

from predicators.src.interaction import BaseExplorer
from predicators.src.structs import Action, State, Task


class RandomActionsExplorer(BaseExplorer):
    """Samples random low-level actions."""

    @classmethod
    def get_name(cls) -> str:
        return "random_actions"

    def solve(
        self, task: Task, timeout: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        # Take random actions.
        policy = lambda _: Action(self._action_space.sample())
        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return policy, termination_function
