"""General utility methods.
"""

import itertools
from collections import defaultdict
from typing import List, Callable, Tuple, Collection, Set, Sequence, Iterator, \
    Dict
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


def extract_preds_and_types(operators: Collection[Operator]) -> Tuple[
        Dict[str, Predicate], Dict[str, Type]]:
    """Extract the predicates and types used in the given operators.
    """
    preds = {}
    types = {}
    for op in operators:
        for atom in op.preconditions | op.add_effects | op.delete_effects:
            for var_type in atom.predicate.types:
                types[var_type.name] = var_type
            preds[atom.predicate.name] = atom.predicate
    return preds, types


def filter_static_operators(ground_operators: Collection[_GroundOperator],
                            atoms: Collection[GroundAtom]) -> List[
                                _GroundOperator]:
    """Filter out ground operators that don't satisfy static facts.
    """
    static_preds = set()
    for pred in {atom.predicate for atom in atoms}:
        # This predicate is not static if it appears in any operator's effects.
        if any(any(atom.predicate == pred for atom in op.add_effects) or
               any(atom.predicate == pred for atom in op.delete_effects)
               for op in ground_operators):
            continue
        static_preds.add(pred)
    static_facts = {atom for atom in atoms if atom.predicate in static_preds}
    # Perform filtering.
    ground_operators = [op for op in ground_operators
                        if not any(atom.predicate in static_preds
                                   and atom not in static_facts
                                   for atom in op.preconditions)]
    return ground_operators


def is_dr_reachable(ground_operators: Collection[_GroundOperator],
                    atoms: Collection[GroundAtom],
                    goal: Set[GroundAtom]) -> bool:
    """Quickly check whether the given goal is reachable from the given atoms
    under the given operators, using a delete relaxation (dr).
    """
    reachables = set(atoms)
    while True:
        fixed_point_reached = True
        for op in ground_operators:
            if op.preconditions.issubset(reachables):
                for new_reachable_atom in op.add_effects-reachables:
                    fixed_point_reached = False
                    reachables.add(new_reachable_atom)
        if fixed_point_reached:
            break
    return goal.issubset(reachables)


def get_applicable_operators(ground_operators: Collection[_GroundOperator],
                             atoms: Collection[GroundAtom]) -> Iterator[
                                 _GroundOperator]:
    """Iterate over operators whose preconditions are satisfied.
    """
    for operator in ground_operators:
        applicable = operator.preconditions.issubset(atoms)
        if applicable:
            yield operator


def apply_operator(operator: _GroundOperator,
                   atoms: Set[GroundAtom]) -> Collection[GroundAtom]:
    """Get a next set of atoms given a current set and a ground operator.
    """
    new_atoms = atoms.copy()
    for atom in operator.add_effects:
        new_atoms.add(atom)
    for atom in operator.delete_effects:
        new_atoms.discard(atom)
    return new_atoms
