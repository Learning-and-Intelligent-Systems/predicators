"""General utility methods.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import abc
import argparse
import functools
import gc
import itertools
import os
from collections import defaultdict
from typing import List, Callable, Tuple, Collection, Set, Sequence, Iterator, \
    Dict, FrozenSet, Any, Optional, Hashable, TypeVar, Generic, cast, Union
import heapq as hq
import imageio
import matplotlib
import numpy as np
from predicators.src.args import create_arg_parser
from predicators.src.structs import _Option, State, Predicate, GroundAtom, \
    Object, Type, NSRT, _GroundNSRT, Action, Task, LowLevelTrajectory, \
    LiftedAtom, Image, Video, Variable, PyperplanFacts, ObjToVarSub, \
    VarToObjSub, Dataset, GroundAtomTrajectory, STRIPSOperator, \
    _GroundSTRIPSOperator, Array, OptionSpec
from predicators.src.settings import CFG, GlobalSettings
matplotlib.use("Agg")


def always_initiable(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> bool:
    """An initiation function for an option that can always be run.
    """
    del state, memory, objects, params  # unused
    return True


def onestep_terminal(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> bool:
    """A termination function for an option that only lasts 1 timestep.
    """
    del state, memory, objects, params  # unused
    return True


def intersects(p1: Tuple[float, float], p2: Tuple[float, float],
               p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """
    Checks if line segment p1p2 and p3p4 intersect.
    This method, which works by checking relative orientation, allows for
    collinearity, and only checks if each segment straddles the line
    containing the other.
    """
    def subtract(a: Tuple[float, float], b: Tuple[float, float]) \
        -> Tuple[float, float]:
        x1, y1 = a
        x2, y2 = b
        return (x1-x2), (y1-y2)
    def cross_product(a: Tuple[float, float], b: Tuple[float, float]) \
        -> float:
        x1, y1 = b
        x2, y2 = a
        return x1 * y2 - x2 * y1
    def direction(a: Tuple[float, float], b: Tuple[float, float],
                  c: Tuple[float, float]) -> float:
        return cross_product(subtract(a, c), subtract(a, b))
    d1 = direction(p3, p4, p1)
    d2 = direction(p3, p4, p2)
    d3 = direction(p1, p2, p3)
    d4 = direction(p1, p2, p4)
    if ((d2 < 0 < d1) or (d1 < 0 < d2)) and \
    ((d4 < 0 < d3) or (d3 < 0 < d4)):
        return True
    return False


def overlap(l1: Tuple[float, float], r1: Tuple[float, float],
               l2: Tuple[float, float], r2: Tuple[float, float]) -> bool:
    """
    Checks if two rectangles defined by their top left and bottom right
    points overlap, allowing for overlaps of measure zero. The first rectangle
    is defined by (l1, r1) and the second is defined by (l2, r2).
    """

    if (l1[0] >= r2[0] or l2[0] >= r1[0]):  # one rect on left side of other
        return False
    if (r1[1] >= l2[1] or r2[1] >= l1[1]):  # one rect above the other
        return False
    return True


@functools.lru_cache(maxsize=None)
def unify(ground_atoms: FrozenSet[GroundAtom],
          lifted_atoms: FrozenSet[LiftedAtom]) -> Tuple[bool, ObjToVarSub]:
    """Return whether the given ground atom set can be unified
    with the given lifted atom set. Also return the mapping.
    """
    ground_atoms_lst = sorted(ground_atoms)
    lifted_atoms_lst = sorted(lifted_atoms)

    # Terminate quickly if there is a mismatch between predicates
    ground_preds = [atom.predicate for atom in ground_atoms_lst]
    lifted_preds = [atom.predicate for atom in lifted_atoms_lst]
    if ground_preds != lifted_preds:
        return False, {}

    # Terminate quickly if there is a mismatch between numbers
    num_objects = len({o for atom in ground_atoms_lst
                       for o in atom.objects})
    num_variables = len({o for atom in lifted_atoms_lst
                         for o in atom.variables})
    if num_objects != num_variables:
        return False, {}

    # Try to get lucky with a one-to-one mapping
    subs12: ObjToVarSub = {}
    subs21 = {}
    success = True
    for atom_ground, atom_lifted in zip(ground_atoms_lst, lifted_atoms_lst):
        if not success:
            break
        for v1, v2 in zip(atom_ground.objects, atom_lifted.variables):
            if v1 in subs12 and subs12[v1] != v2:
                success = False
                break
            if v2 in subs21:
                success = False
                break
            subs12[v1] = v2
            subs21[v2] = v1
    if success:
        return True, subs12

    # If all else fails, use search
    solved, sub = find_substitution(ground_atoms_lst, lifted_atoms_lst)
    rev_sub = {v: k for k, v in sub.items()}
    return solved, rev_sub


def wrap_atom_predicates_lifted(atoms: Collection[LiftedAtom],
                                prefix: str) -> Set[LiftedAtom]:
    """Return a new set of lifted atoms which adds the given prefix
    string to the name of every predicate in atoms.
    NOTE: the classifier is removed.
    """
    new_atoms = set()
    for atom in atoms:
        new_predicate = Predicate(prefix+atom.predicate.name,
                                  atom.predicate.types,
                                  _classifier=lambda s, o: False)  # dummy
        new_atoms.add(LiftedAtom(new_predicate, atom.variables))
    return new_atoms


def wrap_atom_predicates_ground(atoms: Collection[GroundAtom],
                                prefix: str) -> Set[GroundAtom]:
    """Return a new set of ground atoms which adds the given prefix
    string to the name of every predicate in atoms.
    NOTE: the classifier is removed.
    """
    new_atoms = set()
    for atom in atoms:
        new_predicate = Predicate(prefix+atom.predicate.name,
                                  atom.predicate.types,
                                  _classifier=lambda s, o: False)  # dummy
        new_atoms.add(GroundAtom(new_predicate, atom.objects))
    return new_atoms


def run_policy_on_task(policy: Callable[[State], Action], task: Task,
                       simulator: Callable[[State, Action], State],
                       predicates: Collection[Predicate], max_steps: int,
                       make_video: bool = False,
                       render: Optional[
                           Callable[[State, Task, Action], List[Image]]] = None,
                       annotate_traj_with_goal: bool = False,
                       ) -> Tuple[LowLevelTrajectory, Video, bool]:
    """Execute a policy on a task until goal or max steps.
    Return the low-level trajectory (optionally annotated with the goal),
    and a bool for whether the goal was satisfied at the end.
    """
    state = task.init
    atoms = abstract(state, predicates)
    states = [state]
    actions: List[Action] = []
    video: Video = []
    if task.goal.issubset(atoms):  # goal is already satisfied
        goal_reached = True
    else:
        goal_reached = False
        for _ in range(max_steps):
            act = policy(state)
            if make_video:
                assert render is not None
                video.extend(render(state, task, act))
            state = simulator(state, act)
            atoms = abstract(state, predicates)
            actions.append(act)
            states.append(state)
            if task.goal.issubset(atoms):
                goal_reached = True
                break
    if make_video:
        assert render is not None
        # Explanation of type ignore: mypy currently does not
        # support Callables with optional arguments. mypy
        # extensions does, but for the sake of avoiding an
        # additional dependency, we'll just ignore this here.
        video.extend(render(state, task))  # type: ignore
    if annotate_traj_with_goal:
        traj = LowLevelTrajectory(states, actions, task.goal)
    else:
        traj = LowLevelTrajectory(states, actions)
    return traj, video, goal_reached


def policy_solves_task(policy: Callable[[State], Action], task: Task,
                       simulator: Callable[[State, Action], State],
                       predicates: Collection[Predicate]) -> bool:
    """Return whether the given policy solves the given task.
    """
    _, _, solved = run_policy_on_task(policy, task, simulator, predicates,
                                      CFG.max_num_steps_check_policy)
    return solved


def option_to_trajectory(
        init: State,
        simulator: Callable[[State, Action], State],
        option: _Option,
        max_num_steps: int) -> LowLevelTrajectory:
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
    return LowLevelTrajectory(states, actions)


class OptionPlanExhausted(Exception):
    """An exception for an option plan running out of options.
    """


def option_plan_to_policy(plan: Sequence[_Option]
                          ) -> Callable[[State], Action]:
    """Create a policy that executes the options in order.

    The logic for this is somewhat complicated because we want:
    * If an option's termination and initiation conditions are
      always true, we want the option to execute for one step.
    * After the first step that the option is executed, it
      should terminate as soon as it sees a state that is
      terminal; it should not take one more action after.
    """
    queue = list(plan)  # Don't modify plan, just in case
    initialized = False  # Special case first step
    def _policy(state: State) -> Action:
        nonlocal initialized
        # On the very first state, check initiation condition, and
        # take the action no matter what.
        if not initialized:
            if not queue:
                raise OptionPlanExhausted()
            assert queue[0].initiable(state), "Unsound option plan"
            initialized = True
        elif queue[0].terminal(state):
            queue.pop(0)
            if not queue:
                raise OptionPlanExhausted()
            assert queue[0].initiable(state), "Unsound option plan"
        return queue[0].policy(state)
    return _policy


@functools.lru_cache(maxsize=None)
def get_all_groundings(atoms: FrozenSet[LiftedAtom],
                       objects: FrozenSet[Object]
                       ) -> List[Tuple[FrozenSet[GroundAtom], VarToObjSub]]:
    """Get all the ways to ground the given set of lifted atoms into
    a set of ground atoms, using the given objects. Returns a list
    of (ground atoms, substitution dictionary) tuples.
    """
    variables = set()
    for atom in atoms:
        variables.update(atom.variables)
    sorted_variables = sorted(variables)
    types = [var.type for var in sorted_variables]
    # NOTE: We WON'T use a generator here because that breaks lru_cache.
    result = []
    for choice in get_object_combinations(objects, types):
        sub: VarToObjSub = dict(zip(sorted_variables, choice))
        ground_atoms = {atom.ground(sub) for atom in atoms}
        result.append((frozenset(ground_atoms), sub))
    return result


def get_object_combinations(
        objects: Collection[Object], types: Sequence[Type]
        ) -> Iterator[List[Object]]:
    """Get all combinations of objects satisfying the given types sequence.
    """
    sorted_objects = sorted(objects)
    choices = []
    for vt in types:
        this_choices = []
        for obj in sorted_objects:
            if obj.is_instance(vt):
                this_choices.append(obj)
        choices.append(this_choices)
    for choice in itertools.product(*choices):
        yield list(choice)


def get_random_object_combination(
        objects: Collection[Object], types: Sequence[Type],
        rng: np.random.Generator) -> List[Object]:
    """Get a random list of objects from the given collection that
    satisfy the given sequence of types. Duplicates are always allowed.
    """
    types_to_objs = defaultdict(list)
    for obj in objects:
        types_to_objs[obj.type].append(obj)
    return [types_to_objs[t][rng.choice(len(types_to_objs[t]))]
            for t in types]


def find_substitution(super_atoms: Collection[GroundAtom],
                      sub_atoms: Collection[LiftedAtom],
                      allow_redundant: bool = False,
                      ) -> Tuple[bool, VarToObjSub]:
    """Find a substitution from the objects in super_atoms to the variables
    in sub_atoms s.t. sub_atoms is a subset of super_atoms.

    If allow_redundant is True, then multiple variables in sub_atoms can
    refer to the same single object in super_atoms.

    If no substitution exists, return (False, {}).
    """
    super_objects_by_type: Dict[Type, List[Object]] = defaultdict(list)
    super_pred_to_tuples = defaultdict(set)
    for atom in super_atoms:
        for obj in atom.objects:
            if obj not in super_objects_by_type[obj.type]:
                super_objects_by_type[obj.type].append(obj)
        super_pred_to_tuples[atom.predicate].add(tuple(atom.objects))
    sub_variables = sorted({v for atom in sub_atoms for v in atom.variables})
    return _find_substitution_helper(
        sub_atoms, super_objects_by_type, sub_variables, super_pred_to_tuples,
        {}, allow_redundant)


def _find_substitution_helper(
        sub_atoms: Collection[LiftedAtom],
        super_objects_by_type: Dict[Type, List[Object]],
        remaining_sub_variables: List[Variable],
        super_pred_to_tuples: Dict[Predicate, Set[Tuple[Object, ...]]],
        partial_sub: VarToObjSub,
        allow_redundant: bool) -> Tuple[bool, VarToObjSub]:
    """Helper for find_substitution.
    """
    # Base case: check if all assigned
    if not remaining_sub_variables:
        return True, partial_sub
    # Find next variable to assign
    remaining_sub_variables = remaining_sub_variables.copy()
    next_sub_var = remaining_sub_variables.pop(0)
    # Consider possible assignments
    for super_obj in super_objects_by_type[next_sub_var.type]:
        if not allow_redundant and super_obj in partial_sub.values():
            continue
        new_sub = partial_sub.copy()
        new_sub[next_sub_var] = super_obj
        # Check if consistent
        if not _substitution_consistent(new_sub, super_pred_to_tuples,
                                        sub_atoms):
            continue
        # Backtracking search
        solved, final_sub = _find_substitution_helper(sub_atoms,
            super_objects_by_type, remaining_sub_variables,
            super_pred_to_tuples, new_sub, allow_redundant)
        if solved:
            return solved, final_sub
    # Failure
    return False, {}


def _substitution_consistent(
        partial_sub: VarToObjSub,
        super_pred_to_tuples:  Dict[Predicate, Set[Tuple[Object, ...]]],
        sub_atoms: Collection[LiftedAtom]) -> bool:
    """Helper for _find_substitution_helper.
    """
    for sub_atom in sub_atoms:
        if not set(sub_atom.variables).issubset(partial_sub.keys()):
            continue
        substituted_vars = tuple(partial_sub[e] for e in sub_atom.variables)
        if substituted_vars not in super_pred_to_tuples[sub_atom.predicate]:
            return False
    return True


def powerset(seq: Sequence, exclude_empty: bool) -> Iterator[Sequence]:
    """Get an iterator over the powerset of the given sequence.
    """
    start = 1 if exclude_empty else 0
    return itertools.chain.from_iterable(itertools.combinations(list(seq), r)
                                         for r in range(start, len(seq)+1))


_S = TypeVar('_S', bound=Hashable)  # state in heuristic search
_A = TypeVar('_A')  # action in heuristic search


@dataclass(frozen=True)
class _HeuristicSearchNode(Generic[_S, _A]):
    state: _S
    edge_cost: float
    cumulative_cost: float
    parent: Optional[_HeuristicSearchNode[_S, _A]] = None
    action: Optional[_A] = None


def _run_heuristic_search(
    initial_state: _S,
    check_goal: Callable[[_S], bool],
    get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
    get_priority: Callable[[_HeuristicSearchNode[_S, _A]], Any],
    max_expansions: int = 10000000,
    max_evals: int = 10000000,
    lazy_expansion: bool = False
    ) -> Tuple[List[_S], List[_A]]:
    """A generic heuristic search implementation.

    Depending on get_priority, can implement A*, GBFS, or UCS.

    If no goal is found, returns the state with the best priority.
    """
    queue: List[Tuple[Any, int, _HeuristicSearchNode[_S, _A]]] = []
    state_to_best_path_cost: Dict[_S, float] = \
        defaultdict(lambda : float("inf"))

    root_node: _HeuristicSearchNode[_S, _A] = _HeuristicSearchNode(
        initial_state, 0, 0)
    root_priority = get_priority(root_node)
    best_node = root_node
    best_node_priority = root_priority
    tiebreak = itertools.count()
    hq.heappush(queue, (root_priority, next(tiebreak), root_node))
    num_expansions = 0
    num_evals = 1

    while len(queue) > 0 and num_expansions < max_expansions and \
        num_evals < max_evals:
        _, _, node = hq.heappop(queue)
        # If we already found a better path here, don't bother.
        if state_to_best_path_cost[node.state] < node.cumulative_cost:
            continue
        # If the goal holds, return.
        if check_goal(node.state):
            return _finish_plan(node)
        num_expansions += 1
        # Generate successors.
        for action, child_state, cost in get_successors(node.state):
            child_path_cost = node.cumulative_cost + cost
            # If we already found a better path to child, don't bother.
            if state_to_best_path_cost[child_state] <= child_path_cost:
                continue
            # Add new node.
            child_node = _HeuristicSearchNode(
                state=child_state,
                edge_cost=cost,
                cumulative_cost=child_path_cost,
                parent=node,
                action=action)
            priority = get_priority(child_node)
            num_evals += 1
            hq.heappush(queue, (priority, next(tiebreak), child_node))
            state_to_best_path_cost[child_state] = child_path_cost
            if priority < best_node_priority:
                best_node_priority = priority
                best_node = child_node
                # Optimization: if we've found a better child, immediately
                # explore the child without expanding the rest of the children.
                # Accomplish this by putting the parent node back on the queue.
                if lazy_expansion:
                    hq.heappush(queue, (priority, next(tiebreak), node))
                    break
            if num_evals >= max_evals:
                break

    # Did not find path to goal; return best path seen.
    return _finish_plan(best_node)


def _finish_plan(node: _HeuristicSearchNode[_S, _A]
                 ) -> Tuple[List[_S], List[_A]]:
    """Helper for _run_heuristic_search.
    """
    rev_state_sequence: List[_S] = []
    rev_action_sequence: List[_A] = []

    while node.parent is not None:
        action = cast(_A, node.action)
        rev_action_sequence.append(action)
        rev_state_sequence.append(node.state)
        node = node.parent
    rev_state_sequence.append(node.state)

    return rev_state_sequence[::-1], rev_action_sequence[::-1]


def run_gbfs(
    initial_state: _S,
    check_goal: Callable[[_S], bool],
    get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
    heuristic: Callable[[_S], float],
    max_expansions: int = 10000000,
    max_evals: int = 10000000,
    lazy_expansion: bool = False
    ) -> Tuple[List[_S], List[_A]]:
    """Greedy best-first search.
    """
    get_priority = lambda n: heuristic(n.state)
    return _run_heuristic_search(initial_state, check_goal, get_successors,
        get_priority, max_expansions, max_evals, lazy_expansion)


def strip_predicate(predicate: Predicate) -> Predicate:
    """Remove classifier from predicate to make new Predicate.
    """
    return Predicate(predicate.name, predicate.types, lambda s, o: False)


def abstract(state: State, preds: Collection[Predicate]) -> Set[GroundAtom]:
    """Get the atomic representation of the given state (i.e., a set
    of ground atoms), using the given set of predicates.

    NOTE: Duplicate arguments in predicates are DISALLOWED.
    """
    atoms = set()
    for pred in preds:
        for choice in get_object_combinations(list(state), pred.types):
            if pred.holds(state, choice):
                atoms.add(GroundAtom(pred, choice))
    return atoms


def all_ground_operators(operator: STRIPSOperator,
                         objects: Collection[Object]
                         ) -> Set[_GroundSTRIPSOperator]:
    """Get all possible groundings of the given operator with the given objects.
    """
    types = [p.type for p in operator.parameters]
    ground_operators = set()
    for choice in get_object_combinations(objects, types):
        ground_operators.add(operator.ground(tuple(choice)))
    return ground_operators


def all_ground_operators_given_partial(operator: STRIPSOperator,
                                       objects: Collection[Object],
                                       sub: VarToObjSub
                                       ) -> Set[_GroundSTRIPSOperator]:
    """Get all possible groundings of the given operator with the given objects
    such that the parameters are consistent with the given substitution.
    """
    assert set(sub).issubset(set(operator.parameters))
    ground_ops = set()
    types = [p.type for p in operator.parameters if p not in sub]
    for choice in get_object_combinations(objects, types):
        # Complete the choice with the args that are determined from the sub.
        choice_lst = list(choice)
        choice_lst.reverse()
        completed_choice = []
        for p in operator.parameters:
            if p in sub:
                completed_choice.append(sub[p])
            else:
                completed_choice.append(choice_lst.pop())
        assert not choice_lst
        ground_op = operator.ground(tuple(completed_choice))
        ground_ops.add(ground_op)
    return ground_ops


def all_ground_nsrts(
        nsrt: NSRT, objects: Collection[Object]) -> Set[_GroundNSRT]:
    """Get all possible groundings of the given NSRT with the given objects.
    """
    types = [p.type for p in nsrt.parameters]
    ground_nsrts = set()
    for choice in get_object_combinations(objects, types):
        ground_nsrts.add(nsrt.ground(choice))
    return ground_nsrts


def all_ground_predicates(pred: Predicate,
                          objects: Collection[Object]) -> Set[GroundAtom]:
    """Get all possible groundings of the given predicate with the given
    objects.

    NOTE: Duplicate arguments in predicates are DISALLOWED.
    """
    return {GroundAtom(pred, choice)
            for choice in get_object_combinations(objects, pred.types)}


def all_possible_ground_atoms(state: State, preds: Set[Predicate]) \
        -> List[GroundAtom]:
    """Get a sorted list of all possible ground atoms in a state given the
    predicates. Ignores the predicates' classifiers.
    """
    objects = list(state)
    ground_atoms = set()
    for pred in preds:
        ground_atoms |= all_ground_predicates(pred, objects)
    return sorted(ground_atoms)


def create_ground_atom_dataset(dataset: Dataset, predicates: Set[Predicate]
                               ) -> List[GroundAtomTrajectory]:
    """Apply all predicates to all trajectories in the dataset.
    """
    ground_atom_dataset = []
    for traj in dataset:
        atoms = [abstract(s, predicates) for s in traj.states]
        ground_atom_dataset.append((traj, atoms))
    return ground_atom_dataset


def prune_ground_atom_dataset(ground_atom_dataset: List[GroundAtomTrajectory],
                              kept_predicates: Collection[Predicate]
                              ) -> List[GroundAtomTrajectory]:
    """Create a new ground atom dataset by keeping only some predicates.
    """
    new_ground_atom_dataset = []
    for traj, atoms in ground_atom_dataset:
        assert len(traj.states) == len(atoms)
        kept_atoms = [{a for a in sa if a.predicate in kept_predicates}
                      for sa in atoms]
        new_ground_atom_dataset.append((traj, kept_atoms))
    return new_ground_atom_dataset


def extract_preds_and_types(nsrts: Collection[NSRT]) -> Tuple[
        Dict[str, Predicate], Dict[str, Type]]:
    """Extract the predicates and types used in the given NSRTs.
    """
    preds = {}
    types = {}
    for nsrt in nsrts:
        for atom in nsrt.preconditions | nsrt.add_effects | nsrt.delete_effects:
            for var_type in atom.predicate.types:
                types[var_type.name] = var_type
            preds[atom.predicate.name] = atom.predicate
    return preds, types


def filter_static_nsrts(ground_nsrts: Collection[_GroundNSRT],
                        atoms: Collection[GroundAtom]) -> List[
                            _GroundNSRT]:
    """Filter out ground NSRTs that don't satisfy static facts.
    """
    static_preds = set()
    for pred in {atom.predicate for atom in atoms}:
        # This predicate is not static if it appears in any NSRT's effects.
        if any(any(atom.predicate == pred for atom in nsrt.add_effects) or
               any(atom.predicate == pred for atom in nsrt.delete_effects)
               for nsrt in ground_nsrts):
            continue
        static_preds.add(pred)
    static_facts = {atom for atom in atoms if atom.predicate in static_preds}
    # Perform filtering.
    ground_nsrts = [nsrt for nsrt in ground_nsrts
                    if not any(atom.predicate in static_preds
                               and atom not in static_facts
                               for atom in nsrt.preconditions)]
    return ground_nsrts


def is_dr_reachable(ground_nsrts: Collection[_GroundNSRT],
                    atoms: Collection[GroundAtom],
                    goal: Set[GroundAtom]) -> bool:
    """Quickly check whether the given goal is reachable from the given atoms
    under the given NSRTs, using a delete relaxation (dr).
    """
    reachables = set(atoms)
    while True:
        fixed_point_reached = True
        for nsrt in ground_nsrts:
            if nsrt.preconditions.issubset(reachables):
                for new_reachable_atom in nsrt.add_effects-reachables:
                    fixed_point_reached = False
                    reachables.add(new_reachable_atom)
        if fixed_point_reached:
            break
    return goal.issubset(reachables)


def get_applicable_nsrts(ground_nsrts: Collection[_GroundNSRT],
                         atoms: Collection[GroundAtom]) -> Iterator[
                             _GroundNSRT]:
    """Iterate over NSRTs whose preconditions are satisfied.
    """
    for nsrt in sorted(ground_nsrts):
        applicable = nsrt.preconditions.issubset(atoms)
        if applicable:
            yield nsrt


def get_applicable_operators(ground_ops: Collection[_GroundSTRIPSOperator],
                             atoms: Collection[GroundAtom]) -> Iterator[
                             _GroundSTRIPSOperator]:
    """Iterate over ground operators whose preconditions are satisfied.

    Note: the order may be nondeterministic. Users should be invariant.
    """
    for op in ground_ops:
        applicable = op.preconditions.issubset(atoms)
        if applicable:
            yield op


def apply_nsrt(nsrt: _GroundNSRT, atoms: Set[GroundAtom]
               ) -> Set[GroundAtom]:
    """Get a next set of atoms given a current set and a ground NSRT.
    """
    new_atoms = atoms.copy()
    for atom in nsrt.add_effects:
        new_atoms.add(atom)
    for atom in nsrt.delete_effects:
        new_atoms.discard(atom)
    return new_atoms


def apply_operator(operator: _GroundSTRIPSOperator, atoms: Set[GroundAtom]
                   ) -> Set[GroundAtom]:
    """Get a next set of atoms given a current set and a ground operator.
    """
    new_atoms = atoms.copy()
    for atom in operator.add_effects:
        new_atoms.add(atom)
    for atom in operator.delete_effects:
        new_atoms.discard(atom)
    return new_atoms


def ops_and_specs_to_dummy_nsrts(strips_ops: Sequence[STRIPSOperator],
                                 option_specs: Sequence[OptionSpec]
                                 ) -> Set[NSRT]:
    """Create NSRTs from strips operators and option specs with dummy samplers.
    """
    assert len(strips_ops) == len(option_specs)
    nsrts = set()
    for op, (param_option, option_vars) in zip(strips_ops, option_specs):
        nsrt = op.make_nsrt(param_option, option_vars,
                            lambda s, rng, o: np.zeros(1))  # dummy sampler
        nsrts.add(nsrt)
    return nsrts


def create_heuristic(heuristic_name: str,
                     init_atoms: Collection[GroundAtom],
                     goal: Collection[GroundAtom],
                     ground_ops: Collection[Union[_GroundNSRT,
                                                  _GroundSTRIPSOperator]]
                     ) -> Callable[[PyperplanFacts], float]:
    """Create a task planning heuristic that consumes pyperplan facts and
    estimates the cost-to-go.
    """
    relaxed_operators = frozenset({RelaxedOperator(
        op.name, atoms_to_tuples(op.preconditions),
        atoms_to_tuples(op.add_effects)) for op in ground_ops})
    if heuristic_name == "hadd":
        return _HAddHeuristic(atoms_to_tuples(init_atoms),
                              atoms_to_tuples(goal),
                              relaxed_operators)
    if heuristic_name == "hmax":
        return _HMaxHeuristic(atoms_to_tuples(init_atoms),
                              atoms_to_tuples(goal),
                              relaxed_operators)
    raise ValueError(f"Unrecognized heuristic name: {heuristic_name}.")


@functools.lru_cache(maxsize=None)
def atom_to_tuple(atom: GroundAtom) -> Tuple[str, ...]:
    """Convert atom to tuple for caching.
    """
    return (atom.predicate.name,) + tuple(str(o) for o in atom.objects)


def atoms_to_tuples(atoms: Collection[GroundAtom]) -> PyperplanFacts:
    """Light wrapper around atom_to_tuple() that operates on a
    collection of atoms.
    """
    return frozenset({atom_to_tuple(atom) for atom in atoms})


@dataclass(repr=False, eq=False)
class RelaxedFact:
    """This class represents a relaxed fact.
    Lightly modified from pyperplan's heuristics/relaxation.py.
    """
    name: Tuple[str, ...]
    # A list that contains all operators this fact is a precondition of.
    precondition_of: List[RelaxedOperator] = field(
        init=False, default_factory=list)
    # Whether this fact has been expanded during the Dijkstra forward pass.
    expanded: bool = field(init=False, default=False)
    # The heuristic distance value.
    distance: float = field(init=False, default=float("inf"))


@dataclass(repr=False, eq=False)
class RelaxedOperator:
    """This class represents a relaxed operator (no delete effects).
    Lightly modified from pyperplan's heuristics/relaxation.py.
    """
    name: str
    preconditions: PyperplanFacts
    add_effects: PyperplanFacts
    # Cost of applying this operator.
    cost: int = field(default=1)
    # Alternative method to check whether all preconditions are True.
    counter: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.counter = len(self.preconditions)  # properly initialize counter


class _RelaxationHeuristic:
    """This class is an implementation of delete relaxation heuristics such as
    HMax and HAdd. Lightly modified from pyperplan's heuristics/relaxation.py.
    """
    def __init__(self, initial_state: PyperplanFacts,
                 goals: PyperplanFacts,
                 operators: FrozenSet[RelaxedOperator]) -> None:
        self.facts = {}
        self.operators = []
        self.goals = goals
        self.init = initial_state
        self.tie_breaker = 0
        self.start_state = RelaxedFact(("start",))

        all_facts = initial_state | goals
        for op in operators:
            all_facts |= op.preconditions
            all_facts |= op.add_effects

        # Create relaxed facts for all facts in the task description.
        for fact in all_facts:
            self.facts[fact] = RelaxedFact(fact)

        for ro in operators:
            # Add operators to operator list.
            self.operators.append(ro)

            # Initialize precondition_of-list for each fact
            for var in ro.preconditions:
                self.facts[var].precondition_of.append(ro)

            # Handle operators that have no preconditions.
            if not ro.preconditions:
                # We add this operator to the precondtion_of list of the start
                # state. This way it can be applied to the start state. This
                # helps also when the initial state is empty.
                self.start_state.precondition_of.append(ro)

    @functools.lru_cache(maxsize=None)
    def __call__(self, state: PyperplanFacts) -> float:
        """Compute heuristic value.
        """
        # Reset distance and set to default values.
        self.init_distance(state)

        # Construct the priority queue.
        heap: List[Tuple[float, float, RelaxedFact]] = []
        # Add a dedicated start state, to cope with operators without
        # preconditions and empty initial state.
        hq.heappush(heap, (0, self.tie_breaker, self.start_state))
        self.tie_breaker += 1

        for fact in state:
            # Order is determined by the distance of the facts.
            # As a tie breaker we use a simple counter.
            hq.heappush(heap, (self.facts[fact].distance,
                               self.tie_breaker, self.facts[fact]))
            self.tie_breaker += 1

        # Call the Dijkstra search that performs the forward pass.
        self.dijkstra(heap)

        # Extract the goal heuristic.
        h_value = self.calc_goal_h()

        return h_value

    @staticmethod
    @abc.abstractmethod
    def _accumulate(distances: Collection[float]) -> float:
        """Combine distances to goal facts. Distinguishes different relaxation
        heuristics, e.g., hmax uses a max and hadd uses a sum.
        """
        raise NotImplementedError("Override me!")

    def init_distance(self, state: PyperplanFacts) -> None:
        """This function resets all member variables that store information
        that needs to be recomputed for each call of the heuristic.
        """
        def _reset_fact(fact: RelaxedFact) -> None:
            fact.expanded = False
            if fact.name in state:
                fact.distance = 0
            else:
                fact.distance = float("inf")

        # Reset start state.
        _reset_fact(self.start_state)

        # Reset facts.
        for fact in self.facts.values():
            _reset_fact(fact)

        # Reset operators.
        for operator in self.operators:
            operator.counter = len(operator.preconditions)

    def get_cost(self, operator: RelaxedOperator) -> float:
        """This function calculates the cost of applying an operator.
        """
        # Accumulate the heuristic values of all preconditions.
        cost = self._accumulate([self.facts[pre].distance
                                 for pre in operator.preconditions])
        # Add on operator application cost.
        return cost+operator.cost

    def calc_goal_h(self) -> float:
        """This function calculates the heuristic value of the whole goal.
        """
        return self._accumulate([self.facts[fact].distance
                                 for fact in self.goals])

    def finished(self, achieved_goals: Set[Tuple[str, ...]],
                 queue: List[Tuple[float, float, RelaxedFact]]) -> bool:
        """This function gives a stopping criterion for the Dijkstra search.
        """
        return achieved_goals == self.goals or not queue

    def dijkstra(self, queue: List[Tuple[float, float, RelaxedFact]]) -> None:
        """This function is an implementation of a Dijkstra search.
        For efficiency reasons, it is used instead of an explicit graph
        representation of the problem.
        """
        # Stores the achieved subgoals.
        achieved_goals: Set[Tuple[str, ...]] = set()
        while not self.finished(achieved_goals, queue):
            # Get the fact with the lowest heuristic value.
            (_dist, _tie, fact) = hq.heappop(queue)
            # If this node is part of the goal, we add to the goal set, which
            # is used as an abort criterion.
            if fact.name in self.goals:
                achieved_goals.add(fact.name)
            # Check whether we already expanded this fact.
            if not fact.expanded:
                # Iterate over all operators this fact is a precondition of.
                for operator in fact.precondition_of:
                    # Decrease the precondition counter.
                    operator.counter -= 1
                    # Check whether all preconditions are True and we can apply
                    # this operator.
                    if operator.counter <= 0:
                        for n in operator.add_effects:
                            neighbor = self.facts[n]
                            # Calculate the cost of applying this operator.
                            tmp_dist = self.get_cost(operator)
                            if tmp_dist < neighbor.distance:
                                # If the new costs are cheaper than the old
                                # costs, we change the neighbor's heuristic
                                # values.
                                neighbor.distance = tmp_dist
                                # And push it on the queue.
                                hq.heappush(queue, (
                                    tmp_dist, self.tie_breaker, neighbor))
                                self.tie_breaker += 1
                # Finally the fact is marked as expanded.
                fact.expanded = True


class _HAddHeuristic(_RelaxationHeuristic):
    """Implements the HAdd delete relaxation heuristic.
    """
    @staticmethod
    def _accumulate(distances: Collection[float]) -> float:
        return sum(distances)


class _HMaxHeuristic(_RelaxationHeuristic):
    """Implements the HMax delete relaxation heuristic.
    """
    @staticmethod
    def _accumulate(distances: Collection[float]) -> float:
        return max(distances)


def fig2data(fig: matplotlib.figure.Figure, dpi: int=150) -> Image:
    """Convert matplotlib figure into Image.
    """
    fig.set_dpi(dpi)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(),  # type: ignore
                         dtype=np.uint8).copy()
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]
    return data


def save_video(outfile: str, video: Video) -> None:
    """Save the video to video_dir/outfile.
    """
    outdir = CFG.video_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, outfile)
    imageio.mimwrite(outpath, video, fps=CFG.video_fps)
    print(f"Wrote out to {outpath}")


def update_config(args: Dict[str, Any]) -> None:
    """Args is a dictionary of new arguments to add to the config CFG.
    """
    # Only override attributes, don't create new ones
    allowed_args = set(CFG.__dict__)
    parser = create_arg_parser()
    # Unfortunately, can't figure out any other way to do this
    for parser_action in parser._actions:  # pylint: disable=protected-access
        allowed_args.add(parser_action.dest)
    for k in args:
        if k not in allowed_args:
            raise ValueError(f"Unrecognized arg: {k}")
    for d in [GlobalSettings.get_arg_specific_settings(args), args]:
        for k, v in d.items():
            CFG.__setattr__(k, v)


def get_config_path_str() -> str:
    """Create a filename prefix based on the current CFG.
    """
    return f"{CFG.env}__{CFG.approach}__{CFG.seed}"


def parse_args() -> Dict[str, Any]:
    """Parses command line arguments.
    """
    parser = create_arg_parser()
    args, overrides = parser.parse_known_args()
    print_args(args)
    arg_dict = vars(args)
    if len(overrides) == 0:
        return arg_dict
    # Update initial settings to make sure we're overriding
    # existing flags only
    update_config(arg_dict)
    # Override global settings
    assert len(overrides) >= 2
    assert len(overrides) % 2 == 0
    for flag, value in zip(overrides[:-1:2], overrides[1::2]):
        assert flag.startswith("--")
        setting_name = flag[2:]
        if setting_name not in CFG.__dict__:
            raise ValueError(f"Unrecognized flag: {setting_name}")
        if value.isdigit():
            value = eval(value)
        arg_dict[setting_name] = value
    return arg_dict


def print_args(args: argparse.Namespace) -> None:
    """Print all info for this experiment.
    """
    print(f"Seed: {args.seed}")
    print(f"Env: {args.env}")
    print(f"Approach: {args.approach}")
    print(f"Timeout: {args.timeout}")
    print()


def flush_cache() -> None:
    """Clear all lru caches.
    """
    gc.collect()
    wrappers = [
        a for a in gc.get_objects()
        if isinstance(a, functools._lru_cache_wrapper)]  # pylint: disable=protected-access

    for wrapper in wrappers:
        wrapper.cache_clear()
