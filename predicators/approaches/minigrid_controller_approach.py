"""An approach that just takes random low-level actions."""

from typing import Callable

from predicators.approaches import BaseApproach
from predicators.structs import Action, State, Task


class MinigridControllerApproach(BaseApproach):
    """Samples random low-level actions."""

    @classmethod
    def get_name(cls) -> str:
        return "minigrid_controller"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        zero_vec = self._action_space.low

        def _policy(_: State) -> Action:
            action_vec = zero_vec.copy()
            print(task.goal)
            action_vec[int(input("Action: "))] = 1.0
            print(action_vec)
            return Action(action_vec)

        return _policy
