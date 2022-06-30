"""An explorer that just takes random low-level actions."""

from predicators.src.interaction import BaseExplorer
from predicators.src.structs import Action, ExplorationStrategy, Task


class RandomActionsExplorer(BaseExplorer):
    """Samples random low-level actions."""

    @classmethod
    def get_name(cls) -> str:
        return "random_actions"

    def get_exploration_strategy(self, task: Task,
                                 timeout: int) -> ExplorationStrategy:
        # Take random actions.
        policy = lambda _: Action(self._action_space.sample())
        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return policy, termination_function
