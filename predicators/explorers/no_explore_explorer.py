"""An explorer that always terminates immediately without taking an action."""

from predicators.explorers import BaseExplorer
from predicators.structs import Action, ExplorationStrategy, State


class NoExploreExplorer(BaseExplorer):
    """Terminates immediately during exploration."""

    @classmethod
    def get_name(cls) -> str:
        return "no_explore"

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:

        def policy(_: State) -> Action:
            raise RuntimeError("The policy for no-explore shouldn't be used.")

        # Terminate immediately.
        termination_function = lambda _: True

        return policy, termination_function
