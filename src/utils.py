"""General utility methods.
"""

from typing import Sequence, Callable, Tuple
from numpy.typing import ArrayLike
from predicators.src.structs import _Option, State


def option_to_trajectory(
        init: State,
        simulator: Callable[[State, ArrayLike], State],
        option: _Option,
        max_num_steps: int) -> Tuple[Sequence[State], Sequence[ArrayLike]]:
    """Convert an option into a trajectory, starting at init, by invoking
    the option policy. This trajectory is a tuple of (state sequence,
    action sequence), where the state sequence includes init.
    """
    actions = []
    assert option.initiable(init)
    state = init
    states = [state]
    for _ in range(max_num_steps):
        if option.terminal(state):
            break
        act = option.policy(state)
        actions.append(act)
        state = simulator(state, act)
        states.append(state)
    assert len(states) == len(actions)+1
    return states, actions
