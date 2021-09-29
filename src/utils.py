"""General utility methods.
"""

from typing import Sequence, Callable, Tuple
from numpy.typing import ArrayLike
from predicators.src.structs import _Option, State


def option_to_actions(
        init: State, simulator: Callable[[State, ArrayLike], State],
        option: _Option) -> Tuple[Sequence[State], Sequence[ArrayLike]]:
    """Convert an option into a sequence of actions by invoking the
    option policy. Also return the state sequence.
    """
    actions = []
    assert option.initiable(init)
    state = init
    states = [state]
    while True:
        if option.terminal(state):
            break
        act = option.policy(state)
        actions.append(act)
        state = simulator(state, act)
        states.append(state)
    return states, actions
