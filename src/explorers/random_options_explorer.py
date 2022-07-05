"""An explorer that takes random options."""

from predicators.src import utils
from predicators.src.explorers import BaseExplorer
from predicators.src.structs import ExplorationStrategy


class RandomOptionsExplorer(BaseExplorer):
    """Samples random options."""

    @classmethod
    def get_name(cls) -> str:
        return "random_options"

    def get_exploration_strategy(self, train_task_idx: int,
                                 timeout: int) -> ExplorationStrategy:
        # Take random options.
        policy = utils.create_random_option_policy(self._options, self._rng)
        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return policy, termination_function
