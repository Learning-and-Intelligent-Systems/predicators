"""An explorer that always terminates immediately without taking an action."""

from predicators.src.interaction import BaseExplorer
from predicators.src.structs import Action, ExplorationStrategy, State, Task


class NoExploreExplorer(BaseExplorer):
    """Terminates immediately during exploration."""

    @classmethod
    def get_name(cls) -> str:
        return "no_explore"

    def get_exploration_strategy(self, task: Task,
                                 timeout: int) -> ExplorationStrategy:

        def policy(_: State) -> Action:
            raise RuntimeError("The policy for no-explore shouldn't be used.")

        # Terminate immediately.
        termination_function = lambda _: True

        return policy, termination_function
