"""An approach that implements a delivery-specific policy.

Example command line:
    python src/main.py --approach delivery_policy --seed 0 \
        --env pddl_easy_delivery_procedural_tasks
"""

from typing import Callable

from predicators.src import utils
from predicators.src.approaches import BaseApproach
from predicators.src.settings import CFG
from predicators.src.structs import Action, DummyOption, State, Task


class DeliverySpecificApproach(BaseApproach):
    """Implements a delivery-specific policy."""

    @classmethod
    def get_name(cls) -> str:
        return "delivery_policy"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        options = sorted(self._initial_options, key=lambda o: o.name)

        def _policy(state: State) -> Action:
            ground_atoms = state.get_ground_atoms()
            # TODO: Add code here!
            # selected_option = ...
            # object_args = ...
            import ipdb; ipdb.set_trace()
            ground_option = selected_option.ground(object_args, [])
            assert ground_option.initiable(state)
            return ground_option.policy(state)


        return _policy
