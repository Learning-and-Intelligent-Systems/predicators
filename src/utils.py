"""General utility methods.
"""

import itertools
from collections import defaultdict
from typing import List, Callable, Tuple, Collection, Set, Sequence, Iterator
import numpy as np
from numpy.typing import NDArray
from predicators.src.structs import _Option, State, Predicate, GroundAtom, \
    Object, Type, Operator, _GroundOperator

Array = NDArray[np.float32]


def option_to_trajectory(
        init: State,
        simulator: Callable[[State, Array], State],
        option: _Option,
        max_num_steps: int) -> Tuple[List[State], List[Array]]:
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


def get_object_combinations(
        objects: Collection[Object], types: Sequence[Type],
        allow_duplicates: bool) -> Iterator[List[Object]]:
    """Get all combinations of objects satisfying the given types sequence.
    """
    type_to_objs = defaultdict(list)
    for obj in sorted(objects):
        type_to_objs[obj.type].append(obj)
    choices = [type_to_objs[vt] for vt in types]
    for choice in itertools.product(*choices):
        if not allow_duplicates and len(set(choice)) != len(choice):
            continue
        yield list(choice)


def abstract(state: State, preds: Collection[Predicate]) -> Set[GroundAtom]:
    """Get the atomic representation of the given state (i.e., a set
    of ground atoms), using the given set of predicates.

    NOTE: Duplicate arguments in predicates are DISALLOWED.
    """
    atoms = set()
    for pred in preds:
        for choice in get_object_combinations(list(state), pred.types,
                                              allow_duplicates=False):
            if pred.holds(state, choice):
                atoms.add(GroundAtom(pred, choice))
    return atoms


def all_ground_operators(
        op: Operator, objects: Collection[Object]) -> Set[_GroundOperator]:
    """Get all possible groundings of the given operator with the given objects.

    NOTE: Duplicate arguments in ground operators are ALLOWED.
    """
    types = [p.type for p in op.parameters]
    ground_operators = set()
    for choice in get_object_combinations(objects, types,
                                          allow_duplicates=True):
        ground_operators.add(op.ground(choice))
    return ground_operators
