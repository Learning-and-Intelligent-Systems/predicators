"""An explorer that takes random options."""

from predicators import utils
from predicators.explorers import BaseExplorer
from predicators.structs import Action, ExplorationStrategy, State


class RandomOptionsExplorer(BaseExplorer):
    """Samples random options."""

    @classmethod
    def get_name(cls) -> str:
        return "random_options"

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        # Take random options, and raise an exception if no applicable option
        # can be found.

        # Note that this fallback policy is different from the one in
        # RandomOptionsApproach because explorers should raise
        # RequestActPolicyFailure instead of ApproachFailure.
        def fallback_policy(state: State) -> Action:
            del state  # unused
            raise utils.RequestActPolicyFailure(
                "Random option sampling failed!")

        policy = utils.create_random_option_policy(self._options, self._rng,
                                                   fallback_policy)
        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return policy, termination_function
