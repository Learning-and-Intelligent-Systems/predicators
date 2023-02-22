"""An approach that just executes random options."""

from typing import Callable

from predicators import utils
from predicators.approaches import ApproachFailure, BaseApproach
from predicators.structs import Action, State, Task


class RandomOptionsApproach(BaseApproach):
    """Samples random options (and random parameters for those options)."""

    @classmethod
    def get_name(cls) -> str:
        return "random_options"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        def fallback_policy(state: State) -> Action:
            del state  # unused
            raise ApproachFailure("Random option sampling failed!")

        return utils.create_random_option_policy(self._initial_options,
                                                 self._rng, fallback_policy)
