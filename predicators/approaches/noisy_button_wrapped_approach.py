"""An approach that "wraps" a base approach for the NoisyButtonEnv. The wrapper
always executes a "find" action when receiving a state with unknown button
position. Otherwise it passes control to the regular approach. The way to use
this environment from the command-line is using the flag.

    --approach 'noisy_button_wrapper[base approach name]'

e.g.

    --approach 'noisy_button_wrapper[oracle]'
"""

from typing import Callable, Optional

import numpy as np

from predicators.approaches import BaseApproachWrapper
from predicators.structs import Action, State, Task


class NoisyButtonWrapperApproach(BaseApproachWrapper):
    """Always "find" when the button position is unknown."""

    @classmethod
    def get_name(cls) -> str:
        return "noisy_button_wrapper"

    @property
    def is_learning_based(self) -> bool:
        return self._base_approach.is_learning_based

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        # Maintain policy from the base approach.
        base_approach_policy: Optional[Callable[[State], Action]] = None

        def _policy(state: State) -> Action:
            nonlocal base_approach_policy
            # If the button state is unknown, execute find.
            button, = list(state)
            if state.get(button, "position_known") < 0.5:
                # Reset the base approach policy.
                base_approach_policy = None
                return Action(np.array([0.0, 1.0], dtype=np.float32))
            # Check if we need to re-solve.
            if base_approach_policy is None:
                cur_task = Task(state, task.goal)
                base_approach_policy = self._base_approach.solve(
                    cur_task, timeout)
            # Use the base policy.
            return base_approach_policy(state)

        return _policy
