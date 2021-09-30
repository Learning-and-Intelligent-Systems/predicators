"""General utility methods.
"""

import itertools
from typing import Sequence, Callable, Tuple, Collection, Set
from numpy.typing import ArrayLike
from predicators.src.structs import _Option, State, Predicate, GroundAtom


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
        act = option.policy(state)
        actions.append(act)
        state = simulator(state, act)
        states.append(state)
        if option.terminal(state):
            break
    assert len(states) == len(actions)+1
    return states, actions


def abstract(state: State, preds: Collection[Predicate]) -> Set[GroundAtom]:
    """Get the atomic representation of the given state (i.e., a set
    of ground atoms), using the given set of predicates.
    """
    atoms = set()
    for pred in preds:
        domains = []
        for var_type in pred.types:
            domains.append([obj for obj in state if obj.type == var_type])
        for choice in itertools.product(*domains):
            if len(choice) != len(set(choice)):
                continue  # ignore duplicate arguments
            if pred.holds(state, choice):
                atoms.add(GroundAtom(pred, choice))
    return atoms
