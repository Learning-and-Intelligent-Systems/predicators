"""General utility methods."""

from __future__ import annotations
from dataclasses import dataclass
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
from pyperplan.heuristics.heuristic_base import \
    Heuristic as _PyperplanBaseHeuristic
from pyperplan.planner import HEURISTICS as _PYPERPLAN_HEURISTICS
from predicators.src.args import create_arg_parser
from predicators.src.structs import _Option, State, Predicate, GroundAtom, \
    Object, Type, NSRT, _GroundNSRT, Action, Task, LowLevelTrajectory, \
    LiftedAtom, Image, Video, _TypedEntity, VarToObjSub, EntToEntSub, \
    Dataset, GroundAtomTrajectory, STRIPSOperator, \
    _GroundSTRIPSOperator, Array, OptionSpec, LiftedOrGroundAtom, \
    NSRTOrSTRIPSOperator, GroundNSRTOrSTRIPSOperator, ParameterizedOption
from predicators.src.settings import CFG, GlobalSettings

matplotlib.use("Agg")


def get_aabb_volume(lo: Array, hi: Array) -> float:
    """Simple utility function to compute the volume of an aabb.

    lo refers to the minimum values of the bbox in the x, y and z axes,
    while hi refers to the highest values. Both lo and hi must be three-
    dimensional.
    """
    assert np.all(hi >= lo)
    dimension = hi - lo
    return dimension[0] * dimension[1] * dimension[2]


def get_closest_point_on_aabb(xyz: List, lo: Array, hi: Array) -> List[float]:
    """Get the closest point on an aabb from a particular xyz coordinate."""
    assert np.all(hi >= lo)
    closest_point_on_aabb = [0.0, 0.0, 0.0]
    for i in range(3):
        # if the coordinate is between the min and max of the aabb, then
        # use that coordinate directly
        if xyz[i] < hi[i] and xyz[i] > lo[i]:
            closest_point_on_aabb[i] = xyz[i]
        else:
            if abs(xyz[i] - hi[i]) < abs(xyz[i] - lo[i]):
                closest_point_on_aabb[i] = hi[i]
            else:
                closest_point_on_aabb[i] = lo[i]
    return closest_point_on_aabb


def always_initiable(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> bool:
    """An initiation function for an option that can always be run."""
    del state, memory, objects, params  # unused
    return True


def onestep_terminal(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> bool:
    """A termination function for an option that only lasts 1 timestep."""
    del state, memory, objects, params  # unused
    return True


def intersects(p1: Tuple[float, float], p2: Tuple[float, float],
               p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
    """Checks if line segment p1p2 and p3p4 intersect.

    This method, which works by checking relative orientation, allows
    for collinearity, and only checks if each segment straddles the line
    containing the other.
    """
    def subtract(a: Tuple[float, float], b: Tuple[float, float]) \
        -> Tuple[float, float]:
        x1, y1 = a
        x2, y2 = b
        return (x1 - x2), (y1 - y2)
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
    """Checks if two rectangles defined by their top left and bottom right
    points overlap, allowing for overlaps of measure zero.

    The first rectangle is defined by (l1, r1) and the second is defined
    by (l2, r2).
    """

    if (l1[0] >= r2[0] or l2[0] >= r1[0]):  # one rect on left side of other
        return False
    if (r1[1] >= l2[1] or r2[1] >= l1[1]):  # one rect above the other
        return False
    return True


@functools.lru_cache(maxsize=None)
def unify(atoms1: FrozenSet[LiftedOrGroundAtom],
          atoms2: FrozenSet[LiftedOrGroundAtom]) -> Tuple[bool, EntToEntSub]:
    """Return whether the given ground atom set can be unified with the given
    lifted atom set.

    Also return the mapping.
    """
    atoms_lst1 = sorted(atoms1)
    atoms_lst2 = sorted(atoms2)

    # Terminate quickly if there is a mismatch between predicates
    preds1 = [atom.predicate for atom in atoms_lst1]
    preds2 = [atom.predicate for atom in atoms_lst2]
    if preds1 != preds2:
        return False, {}

    # Terminate quickly if there is a mismatch between numbers
    num1 = len({o for atom in atoms_lst1 for o in atom.entities})
    num2 = len({o for atom in atoms_lst2 for o in atom.entities})
    if num1 != num2:
        return False, {}

    # Try to get lucky with a one-to-one mapping
    subs12: EntToEntSub = {}
    subs21 = {}
    success = True
    for atom1, atom2 in zip(atoms_lst1, atoms_lst2):
        if not success:
            break
        for v1, v2 in zip(atom1.entities, atom2.entities):
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
    solved, sub = find_substitution(atoms_lst1, atoms_lst2)
    rev_sub = {v: k for k, v in sub.items()}
    return solved, rev_sub


@functools.lru_cache(maxsize=None)
def unify_preconds_effects_options(
        preconds1: FrozenSet[LiftedOrGroundAtom],
        preconds2: FrozenSet[LiftedOrGroundAtom],
        add_effects1: FrozenSet[LiftedOrGroundAtom],
        add_effects2: FrozenSet[LiftedOrGroundAtom],
        delete_effects1: FrozenSet[LiftedOrGroundAtom],
        delete_effects2: FrozenSet[LiftedOrGroundAtom],
        param_option1: ParameterizedOption, param_option2: ParameterizedOption,
        option_args1: Tuple[_TypedEntity, ...],
        option_args2: Tuple[_TypedEntity, ...]) -> Tuple[bool, EntToEntSub]:
    """Wrapper around unify() that handles option arguments, preconditions, add
    effects, and delete effects.

    Changes predicate names so that all are treated differently by
    unify().
    """
    if param_option1 != param_option2:
        # Can't unify if the parameterized options are different.
        return False, {}
    opt_arg_pred1 = Predicate("OPT-ARGS", [a.type for a in option_args1],
                              _classifier=lambda s, o: False)  # dummy
    f_option_args1 = frozenset({GroundAtom(opt_arg_pred1, option_args1)})
    new_preconds1 = wrap_atom_predicates(preconds1, "PRE-")
    f_new_preconds1 = frozenset(new_preconds1)
    new_add_effects1 = wrap_atom_predicates(add_effects1, "ADD-")
    f_new_add_effects1 = frozenset(new_add_effects1)
    new_delete_effects1 = wrap_atom_predicates(delete_effects1, "DEL-")
    f_new_delete_effects1 = frozenset(new_delete_effects1)

    opt_arg_pred2 = Predicate("OPT-ARGS", [a.type for a in option_args2],
                              _classifier=lambda s, o: False)  # dummy
    f_option_args2 = frozenset({LiftedAtom(opt_arg_pred2, option_args2)})
    new_preconds2 = wrap_atom_predicates(preconds2, "PRE-")
    f_new_preconds2 = frozenset(new_preconds2)
    new_add_effects2 = wrap_atom_predicates(add_effects2, "ADD-")
    f_new_add_effects2 = frozenset(new_add_effects2)
    new_delete_effects2 = wrap_atom_predicates(delete_effects2, "DEL-")
    f_new_delete_effects2 = frozenset(new_delete_effects2)

    all_atoms1 = (f_option_args1 | f_new_preconds1 | f_new_add_effects1
                  | f_new_delete_effects1)
    all_atoms2 = (f_option_args2 | f_new_preconds2 | f_new_add_effects2
                  | f_new_delete_effects2)
    return unify(all_atoms1, all_atoms2)


def wrap_atom_predicates(atoms: Collection[LiftedOrGroundAtom],
                         prefix: str) -> Set[LiftedOrGroundAtom]:
    """Return a new set of atoms which adds the given prefix string to the name
    of every predicate in atoms.

    NOTE: the classifier is removed.
    """
    new_atoms = set()
    for atom in atoms:
        new_predicate = Predicate(prefix + atom.predicate.name,
                                  atom.predicate.types,
                                  _classifier=lambda s, o: False)  # dummy
        new_atoms.add(atom.__class__(new_predicate, atom.entities))
    return new_atoms


def run_policy_until(
        policy: Callable[[State], Action],
        simulator: Callable[[State, Action], State],
        init_state: State,
        termination_function: Callable[[State], Action],
        max_num_steps: int,
) -> Tuple[LowLevelTrajectory, Video, bool]:
    """Execute a policy from an initial state, using a simulator.
    Terminates when any of these conditions hold:
    (1) the termination_function returns True,
    (2) the policy throws OptionPlanExhausted,
    (2) max_num_steps is reached,
    (3) the state does not change within a single step.
    Returns a tuple of:
    (1) a LowLevelTrajectory object,
    (2) a boolean for whether termination occurred due to termination_function,
        i.e., due to condition (1) rather than any of the others.
    """
    state = init_state
    states = [state]
    actions: List[Action] = []
    if termination_function(state):
        term_fn_hit = True
    else:
        term_fn_hit = False
        last_state = state
        for _ in range(max_num_steps):
            try:
                act = policy(state)
            except OptionPlanExhausted:
                break
            state = simulator(state, act)
            actions.append(act)
            states.append(state)
            if termination_function(state):
                term_fn_hit = True
                break
            # Detect if stuck; skip potentially expensive simulation.
            if state.allclose(last_state):
                break
    traj = LowLevelTrajectory(states, actions)
    return traj, term_fn_hit


def run_policy_on_task(
        policy: Callable[[State], Action],
        task: Task,
        simulator: Callable[[State, Action], State],
        predicates: Collection[Predicate],
        max_num_steps: int,
        make_video: bool = False,
        render: Optional[Callable[[State, Task, Action], List[Image]]] = None,
) -> Tuple[LowLevelTrajectory, Video, bool]:
    """A light wrapper around run_policy_until that takes in a task and uses
    achieving the task's goal as the termination_function. Also allows for
    video generation.
    """
    def _terminal(state):
        atoms = abstract(state, predicates)
        return task.goal.issubset(atoms)
    traj, goal_reached = run_policy_until(policy, simulator, task.init,
                                          _terminal, max_num_steps)
    video: Video = []
    if make_video:  # step through the trajectory again, making the video
        assert render is not None
        for i, state in enumerate(traj.states):
            act = traj.actions[i] if i < len(traj.states) - 1 else None
            video.extend(render(state, task, act))
    return traj, video, goal_reached


def policy_solves_task(policy: Callable[[State], Action], task: Task,
                       simulator: Callable[[State, Action], State],
                       predicates: Collection[Predicate]) -> bool:
    """A light wrapper around run_policy_on_task that returns whether the
    given policy solves the given task."""
    _, _, solved = run_policy_on_task(policy, task, simulator, predicates,
                                      CFG.max_num_steps_check_policy)
    return solved


def option_to_trajectory(init: State, simulator: Callable[[State, Action],
                                                          State],
                         option: _Option,
                         max_num_steps: int) -> LowLevelTrajectory:
    """A light wrapper around run_policy_until that takes in an option and
    uses achieving its terminal() condition as the termination_function.
    """
    policy = option_plan_to_policy([option])
    # Don't include a termation condition here because it's automatically
    # handled by the policy raising OptionPlanExhausted().
    traj, _ = run_policy_until(policy, simulator, init, lambda s: False,
                               max_num_steps)
    return traj


class OptionPlanExhausted(Exception):
    """An exception for an option plan running out of options."""


def option_plan_to_policy(
        plan: Sequence[_Option]) -> Callable[[State], Action]:
    """Create a policy that executes the options in order.

    The logic for this is somewhat complicated because:
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
def get_all_groundings(
    atoms: FrozenSet[LiftedAtom], objects: FrozenSet[Object]
) -> List[Tuple[FrozenSet[GroundAtom], VarToObjSub]]:
    """Get all the ways to ground the given set of lifted atoms into a set of
    ground atoms, using the given objects.

    Returns a list of (ground atoms, substitution dictionary) tuples.
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


def get_object_combinations(objects: Collection[Object],
                            types: Sequence[Type]) -> Iterator[List[Object]]:
    """Get all combinations of objects satisfying the given types sequence."""
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


@functools.lru_cache(maxsize=None)
def get_all_ground_atoms_for_predicate(
        predicate: Predicate, objects: FrozenSet[Object]) -> Set[GroundAtom]:
    """Get all groundings of the predicate given objects."""
    ground_atoms = set()
    for args in get_object_combinations(objects, predicate.types):
        ground_atom = GroundAtom(predicate, args)
        ground_atoms.add(ground_atom)
    return ground_atoms


@functools.lru_cache(maxsize=None)
def get_all_ground_atoms(predicates: FrozenSet[Predicate],
                         objects: FrozenSet[Object]) -> Set[GroundAtom]:
    """Get all groundings of the predicates given objects."""
    ground_atoms = set()
    for predicate in predicates:
        ground_atoms.update(
            get_all_ground_atoms_for_predicate(predicate, objects))
    return ground_atoms


def get_random_object_combination(objects: Collection[Object],
                                  types: Sequence[Type],
                                  rng: np.random.Generator) -> List[Object]:
    """Get a random list of objects from the given collection that satisfy the
    given sequence of types.

    Duplicates are always allowed.
    """
    types_to_objs = defaultdict(list)
    for obj in objects:
        types_to_objs[obj.type].append(obj)
    return [types_to_objs[t][rng.choice(len(types_to_objs[t]))] for t in types]


def find_substitution(
    super_atoms: Collection[LiftedOrGroundAtom],
    sub_atoms: Collection[LiftedOrGroundAtom],
    allow_redundant: bool = False,
) -> Tuple[bool, EntToEntSub]:
    """Find a substitution from the objects in super_atoms to the variables in
    sub_atoms s.t. sub_atoms is a subset of super_atoms.

    If allow_redundant is True, then multiple variables in sub_atoms can
    refer to the same single object in super_atoms.

    If no substitution exists, return (False, {}).
    """
    super_entities_by_type: Dict[Type, List[_TypedEntity]] = defaultdict(list)
    super_pred_to_tuples = defaultdict(set)
    for atom in super_atoms:
        for obj in atom.entities:
            if obj not in super_entities_by_type[obj.type]:
                super_entities_by_type[obj.type].append(obj)
        super_pred_to_tuples[atom.predicate].add(tuple(atom.entities))
    sub_variables = sorted({e for atom in sub_atoms for e in atom.entities})
    return _find_substitution_helper(sub_atoms, super_entities_by_type,
                                     sub_variables, super_pred_to_tuples, {},
                                     allow_redundant)


def _find_substitution_helper(
        sub_atoms: Collection[LiftedOrGroundAtom],
        super_entities_by_type: Dict[Type, List[_TypedEntity]],
        remaining_sub_variables: List[_TypedEntity],
        super_pred_to_tuples: Dict[Predicate,
                                   Set[Tuple[_TypedEntity,
                                             ...]]], partial_sub: EntToEntSub,
        allow_redundant: bool) -> Tuple[bool, EntToEntSub]:
    """Helper for find_substitution."""
    # Base case: check if all assigned
    if not remaining_sub_variables:
        return True, partial_sub
    # Find next variable to assign
    remaining_sub_variables = remaining_sub_variables.copy()
    next_sub_var = remaining_sub_variables.pop(0)
    # Consider possible assignments
    for super_obj in super_entities_by_type[next_sub_var.type]:
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
                                                      super_entities_by_type,
                                                      remaining_sub_variables,
                                                      super_pred_to_tuples,
                                                      new_sub, allow_redundant)
        if solved:
            return solved, final_sub
    # Failure
    return False, {}


def _substitution_consistent(
        partial_sub: EntToEntSub,
        super_pred_to_tuples: Dict[Predicate, Set[Tuple[_TypedEntity, ...]]],
        sub_atoms: Collection[LiftedOrGroundAtom]) -> bool:
    """Helper for _find_substitution_helper."""
    for sub_atom in sub_atoms:
        if not set(sub_atom.entities).issubset(partial_sub.keys()):
            continue
        substituted_vars = tuple(partial_sub[e] for e in sub_atom.entities)
        if substituted_vars not in super_pred_to_tuples[sub_atom.predicate]:
            return False
    return True


def powerset(seq: Sequence, exclude_empty: bool) -> Iterator[Sequence]:
    """Get an iterator over the powerset of the given sequence."""
    start = 1 if exclude_empty else 0
    return itertools.chain.from_iterable(
        itertools.combinations(list(seq), r)
        for r in range(start,
                       len(seq) + 1))


_S = TypeVar("_S", bound=Hashable)  # state in heuristic search
_A = TypeVar("_A")  # action in heuristic search


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
        lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
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
            child_node = _HeuristicSearchNode(state=child_state,
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


def _finish_plan(
        node: _HeuristicSearchNode[_S, _A]) -> Tuple[List[_S], List[_A]]:
    """Helper for _run_heuristic_search."""
    rev_state_sequence: List[_S] = []
    rev_action_sequence: List[_A] = []

    while node.parent is not None:
        action = cast(_A, node.action)
        rev_action_sequence.append(action)
        rev_state_sequence.append(node.state)
        node = node.parent
    rev_state_sequence.append(node.state)

    return rev_state_sequence[::-1], rev_action_sequence[::-1]


def run_gbfs(initial_state: _S,
             check_goal: Callable[[_S], bool],
             get_successors: Callable[[_S], Iterator[Tuple[_A, _S, float]]],
             heuristic: Callable[[_S], float],
             max_expansions: int = 10000000,
             max_evals: int = 10000000,
             lazy_expansion: bool = False) -> Tuple[List[_S], List[_A]]:
    """Greedy best-first search."""
    get_priority = lambda n: heuristic(n.state)
    return _run_heuristic_search(initial_state, check_goal, get_successors,
                                 get_priority, max_expansions, max_evals,
                                 lazy_expansion)


def run_hill_climbing(initial_state: _S,
                      check_goal: Callable[[_S], bool],
                      get_successors: Callable[[_S], Iterator[Tuple[_A, _S,
                                                                    float]]],
                      heuristic: Callable[[_S], float],
                      enforced_depth: int = 0) -> Tuple[List[_S], List[_A]]:
    """Enforced hill climbing local search.

    For each node, the best child node is always selected, if that child is
    an improvement over the node. If no children improve on the node, look
    at the children's children, etc., up to enforced_depth, where enforced_depth
    0 corresponds to simple hill climbing. Terminate when no improvement can
    be found.

    Lower heuristic is better.
    """
    assert enforced_depth >= 0
    cur_node: _HeuristicSearchNode[_S, _A] = _HeuristicSearchNode(
        initial_state, 0, 0)
    last_heuristic = heuristic(cur_node.state)
    visited = {initial_state}
    print(f"\n\nStarting hill climbing at state {cur_node.state} "
          f"with heuristic {last_heuristic}")
    while True:
        if check_goal(cur_node.state):
            print("\nTerminating hill climbing, achieved goal")
            break
        best_heuristic = float("inf")
        best_child_node = None
        current_depth_nodes = [cur_node]
        for depth in range(0, enforced_depth + 1):
            print(f"Searching for an improvement at depth {depth}")
            # This is a list to ensure determinism. Note that duplicates are
            # filtered out in the `child_state in visited` check.
            successors_at_depth = []
            for parent in current_depth_nodes:
                for action, child_state, cost in get_successors(parent.state):
                    if child_state in visited:
                        continue
                    visited.add(child_state)
                    child_path_cost = parent.cumulative_cost + cost
                    child_node = _HeuristicSearchNode(
                        state=child_state,
                        edge_cost=cost,
                        cumulative_cost=child_path_cost,
                        parent=parent,
                        action=action)
                    successors_at_depth.append(child_node)
                    child_heuristic = heuristic(child_node.state)
                    if child_heuristic < best_heuristic:
                        best_heuristic = child_heuristic
                        best_child_node = child_node
            # Some improvement found.
            if last_heuristic > best_heuristic:
                print(f"Found an improvement at depth {depth}")
                break
            # Continue on to the next depth.
            current_depth_nodes = successors_at_depth
            print(f"No improvement found at depth {depth}")
        if best_child_node is None:
            print("\nTerminating hill climbing, no more successors")
            break
        if last_heuristic <= best_heuristic:
            print("\nTerminating hill climbing, could not improve score")
            break
        cur_node = best_child_node
        last_heuristic = best_heuristic
        print(f"\nHill climbing reached new state {cur_node.state} "
              f"with heuristic {last_heuristic}")
    return _finish_plan(cur_node)


def strip_predicate(predicate: Predicate) -> Predicate:
    """Remove classifier from predicate to make new Predicate."""
    return Predicate(predicate.name, predicate.types, lambda s, o: False)


def abstract(state: State, preds: Collection[Predicate]) -> Set[GroundAtom]:
    """Get the atomic representation of the given state (i.e., a set of ground
    atoms), using the given set of predicates.

    NOTE: Duplicate arguments in predicates are DISALLOWED.
    """
    atoms = set()
    for pred in preds:
        for choice in get_object_combinations(list(state), pred.types):
            if pred.holds(state, choice):
                atoms.add(GroundAtom(pred, choice))
    return atoms


def all_ground_operators(
        operator: STRIPSOperator,
        objects: Collection[Object]) -> Iterator[_GroundSTRIPSOperator]:
    """Get all possible groundings of the given operator with the given
    objects."""
    types = [p.type for p in operator.parameters]
    for choice in get_object_combinations(objects, types):
        yield operator.ground(tuple(choice))


def all_ground_operators_given_partial(
        operator: STRIPSOperator, objects: Collection[Object],
        sub: VarToObjSub) -> Iterator[_GroundSTRIPSOperator]:
    """Get all possible groundings of the given operator with the given objects
    such that the parameters are consistent with the given substitution."""
    assert set(sub).issubset(set(operator.parameters))
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
        yield ground_op


def all_ground_nsrts(nsrt: NSRT,
                     objects: Collection[Object]) -> Iterator[_GroundNSRT]:
    """Get all possible groundings of the given NSRT with the given objects."""
    types = [p.type for p in nsrt.parameters]
    for choice in get_object_combinations(objects, types):
        yield nsrt.ground(choice)


def all_ground_predicates(pred: Predicate,
                          objects: Collection[Object]) -> Set[GroundAtom]:
    """Get all possible groundings of the given predicate with the given
    objects.

    NOTE: Duplicate arguments in predicates are DISALLOWED.
    """
    return {
        GroundAtom(pred, choice)
        for choice in get_object_combinations(objects, pred.types)
    }


def all_possible_ground_atoms(state: State,
                              preds: Set[Predicate]) -> List[GroundAtom]:
    """Get a sorted list of all possible ground atoms in a state given the
    predicates.

    Ignores the predicates' classifiers.
    """
    objects = list(state)
    ground_atoms = set()
    for pred in preds:
        ground_atoms |= all_ground_predicates(pred, objects)
    return sorted(ground_atoms)


def create_ground_atom_dataset(
        dataset: Dataset,
        predicates: Set[Predicate]) -> List[GroundAtomTrajectory]:
    """Apply all predicates to all trajectories in the dataset."""
    ground_atom_dataset = []
    for traj in dataset:
        atoms = [abstract(s, predicates) for s in traj.states]
        ground_atom_dataset.append((traj, atoms))
    return ground_atom_dataset


def prune_ground_atom_dataset(
        ground_atom_dataset: List[GroundAtomTrajectory],
        kept_predicates: Collection[Predicate]) -> List[GroundAtomTrajectory]:
    """Create a new ground atom dataset by keeping only some predicates."""
    new_ground_atom_dataset = []
    for traj, atoms in ground_atom_dataset:
        assert len(traj.states) == len(atoms)
        kept_atoms = [{a
                       for a in sa if a.predicate in kept_predicates}
                      for sa in atoms]
        new_ground_atom_dataset.append((traj, kept_atoms))
    return new_ground_atom_dataset


def extract_preds_and_types(
    ops: Collection[NSRTOrSTRIPSOperator]
) -> Tuple[Dict[str, Predicate], Dict[str, Type]]:
    """Extract the predicates and types used in the given operators."""
    preds = {}
    types = {}
    for op in ops:
        for atom in op.preconditions | op.add_effects | op.delete_effects:
            for var_type in atom.predicate.types:
                types[var_type.name] = var_type
            preds[atom.predicate.name] = atom.predicate
    return preds, types


def get_static_preds_atoms(
        ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
        atoms: Collection[GroundAtom]
) -> Tuple[Set[Predicate], Set[GroundAtom]]:
    """Get predicates and atoms that are static with respect to the
    operators."""
    static_preds = set()
    for pred in {atom.predicate for atom in atoms}:
        # This predicate is not static if it appears in any op's effects.
        if any(
                any(atom.predicate == pred for atom in op.add_effects) or any(
                    atom.predicate == pred for atom in op.delete_effects)
                for op in ground_ops):
            continue
        static_preds.add(pred)
    static_atoms = {atom for atom in atoms if atom.predicate in static_preds}
    return static_preds, static_atoms


def get_reachable_atoms(ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
                        atoms: Collection[GroundAtom]) -> Set[GroundAtom]:
    """Get all atoms that are reachable from the init atoms."""
    reachables = set(atoms)
    while True:
        fixed_point_reached = True
        for op in ground_ops:
            if op.preconditions.issubset(reachables):
                for new_reachable_atom in op.add_effects - reachables:
                    fixed_point_reached = False
                    reachables.add(new_reachable_atom)
        if fixed_point_reached:
            break
    return reachables


def get_applicable_operators(
        ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
        atoms: Collection[GroundAtom]) -> Iterator[GroundNSRTOrSTRIPSOperator]:
    """Iterate over ground operators whose preconditions are satisfied.

    Note: the order may be nondeterministic. Users should be invariant.
    """
    for op in ground_ops:
        applicable = op.preconditions.issubset(atoms)
        if applicable:
            yield op


def apply_operator(op: GroundNSRTOrSTRIPSOperator,
                   atoms: Set[GroundAtom]) -> Set[GroundAtom]:
    """Get a next set of atoms given a current set and a ground operator."""
    # Note that we are removing the side predicates before the
    # application of the operator, because if the side predicate
    # appears in the effects, we still know that the effects
    # will be true, so we don't want to remove them.
    new_atoms = {a for a in atoms if a.predicate not in op.side_predicates}
    for atom in op.add_effects:
        new_atoms.add(atom)
    for atom in op.delete_effects:
        new_atoms.discard(atom)
    return new_atoms


def get_successors_from_ground_ops(
        atoms: Set[GroundAtom],
        ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
        unique: bool = True) -> Iterator[Set[GroundAtom]]:
    """Get all next atoms from ground operators.

    If unique is true, only yield each unique successor once.
    """
    seen_successors = set()
    for ground_op in get_applicable_operators(ground_ops, atoms):
        next_atoms = apply_operator(ground_op, atoms)
        if unique:
            frozen_next_atoms = frozenset(next_atoms)
            if frozen_next_atoms in seen_successors:
                continue
            seen_successors.add(frozen_next_atoms)
        yield next_atoms


def ops_and_specs_to_dummy_nsrts(
        strips_ops: Sequence[STRIPSOperator],
        option_specs: Sequence[OptionSpec]) -> Set[NSRT]:
    """Create NSRTs from strips operators and option specs with dummy
    samplers."""
    assert len(strips_ops) == len(option_specs)
    nsrts = set()
    for op, (param_option, option_vars) in zip(strips_ops, option_specs):
        nsrt = op.make_nsrt(
            param_option,
            option_vars,  # dummy sampler
            lambda s, rng, o: np.zeros(1, dtype=np.float32))
        nsrts.add(nsrt)
    return nsrts


def create_task_planning_heuristic(
    heuristic_name: str,
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
) -> _TaskPlanningHeuristic:
    """Create a task planning heuristic that consumes ground atoms and
    estimates the cost-to-go."""
    if heuristic_name in _PYPERPLAN_HEURISTICS:
        return _create_pyperplan_heuristic(heuristic_name, init_atoms, goal,
                                           ground_ops, predicates, objects)
    raise ValueError(f"Unrecognized heuristic name: {heuristic_name}.")


@dataclass(frozen=True)
class _TaskPlanningHeuristic:
    """A task planning heuristic."""
    name: str
    init_atoms: Collection[GroundAtom]
    goal: Collection[GroundAtom]
    ground_ops: Collection[Union[_GroundNSRT, _GroundSTRIPSOperator]]

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        raise NotImplementedError("Override me!")


############################### Pyperplan Glue ###############################


def _create_pyperplan_heuristic(
    heuristic_name: str,
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
) -> _PyperplanHeuristicWrapper:
    """Create a pyperplan heuristic that inherits from
    _TaskPlanningHeuristic."""
    assert heuristic_name in _PYPERPLAN_HEURISTICS
    _, static_atoms = get_static_preds_atoms(ground_ops, init_atoms)
    pyperplan_heuristic_cls = _PYPERPLAN_HEURISTICS[heuristic_name]
    pyperplan_task = _create_pyperplan_task(init_atoms, goal, ground_ops,
                                            predicates, objects, static_atoms)
    pyperplan_heuristic = pyperplan_heuristic_cls(pyperplan_task)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    return _PyperplanHeuristicWrapper(heuristic_name, init_atoms, goal,
                                      ground_ops, static_atoms,
                                      pyperplan_heuristic, pyperplan_goal)


_PyperplanFacts = FrozenSet[str]


@dataclass(frozen=True)
class _PyperplanNode:
    """Container glue for pyperplan heuristics."""
    state: _PyperplanFacts
    goal: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanOperator:
    """Container glue for pyperplan heuristics."""
    name: str
    preconditions: _PyperplanFacts
    add_effects: _PyperplanFacts
    del_effects: _PyperplanFacts


@dataclass(frozen=True)
class _PyperplanTask:
    """Container glue for pyperplan heuristics."""
    facts: _PyperplanFacts
    initial_state: _PyperplanFacts
    goals: _PyperplanFacts
    operators: Collection[_PyperplanOperator]


@dataclass(frozen=True)
class _PyperplanHeuristicWrapper(_TaskPlanningHeuristic):
    """A light wrapper around pyperplan's heuristics."""
    _static_atoms: Set[GroundAtom]
    _pyperplan_heuristic: _PyperplanBaseHeuristic
    _pyperplan_goal: _PyperplanFacts

    def __call__(self, atoms: Collection[GroundAtom]) -> float:
        # Note: filtering out static atoms.
        pyperplan_facts = _atoms_to_pyperplan_facts(set(atoms) \
                                                    - self._static_atoms)
        return self._evaluate(pyperplan_facts, self._pyperplan_goal,
                              self._pyperplan_heuristic)

    @staticmethod
    @functools.lru_cache(maxsize=None)
    def _evaluate(pyperplan_facts: _PyperplanFacts,
                  pyperplan_goal: _PyperplanFacts,
                  pyperplan_heuristic: _PyperplanBaseHeuristic) -> float:
        pyperplan_node = _PyperplanNode(pyperplan_facts, pyperplan_goal)
        return pyperplan_heuristic(pyperplan_node)


def _create_pyperplan_task(
    init_atoms: Set[GroundAtom],
    goal: Set[GroundAtom],
    ground_ops: Collection[GroundNSRTOrSTRIPSOperator],
    predicates: Collection[Predicate],
    objects: Collection[Object],
    static_atoms: Set[GroundAtom],
) -> _PyperplanTask:
    """Helper glue for pyperplan heuristics."""
    all_atoms = get_all_ground_atoms(frozenset(predicates), frozenset(objects))
    # Note: removing static atoms.
    pyperplan_facts = _atoms_to_pyperplan_facts(all_atoms - static_atoms)
    pyperplan_state = _atoms_to_pyperplan_facts(init_atoms - static_atoms)
    pyperplan_goal = _atoms_to_pyperplan_facts(goal - static_atoms)
    pyperplan_operators = set()
    for op in ground_ops:
        # Note: the pyperplan operator must include the objects, because hFF
        # uses the operator name in constructing the relaxed plan, and the
        # relaxed plan is a set. If we instead just used op.name, there would
        # be a very nasty bug where two ground operators in the relaxed plan
        # that have different objects are counted as just one.
        name = op.name + "-".join(o.name for o in op.objects)
        pyperplan_operator = _PyperplanOperator(
            name,
            # Note: removing static atoms from preconditions.
            _atoms_to_pyperplan_facts(op.preconditions - static_atoms),
            _atoms_to_pyperplan_facts(op.add_effects),
            _atoms_to_pyperplan_facts(op.delete_effects))
        pyperplan_operators.add(pyperplan_operator)
    return _PyperplanTask(pyperplan_facts, pyperplan_state, pyperplan_goal,
                          pyperplan_operators)


@functools.lru_cache(maxsize=None)
def _atom_to_pyperplan_fact(atom: GroundAtom) -> str:
    """Convert atom to tuple for interface with pyperplan."""
    arg_str = " ".join(o.name for o in atom.objects)
    return f"({atom.predicate.name} {arg_str})"


def _atoms_to_pyperplan_facts(
        atoms: Collection[GroundAtom]) -> _PyperplanFacts:
    """Light wrapper around _atom_to_pyperplan_fact() that operates on a
    collection of atoms."""
    return frozenset({_atom_to_pyperplan_fact(atom) for atom in atoms})


############################## End Pyperplan Glue ##############################


def create_pddl_domain(operators: Collection[NSRTOrSTRIPSOperator],
                       predicates: Collection[Predicate],
                       types: Collection[Type], domain_name: str) -> str:
    """Create a PDDL domain str from STRIPSOperators or NSRTs."""
    # Sort everything to ensure determinism.
    preds_lst = sorted(predicates)
    types_lst = sorted(types)
    ops_lst = sorted(operators)
    types_str = " ".join(t.name for t in types_lst)
    preds_str = "\n    ".join(pred.pddl_str() for pred in preds_lst)
    ops_strs = "\n\n  ".join(op.pddl_str() for op in ops_lst)
    return f"""(define (domain {domain_name})
  (:requirements :typing)
  (:types {types_str})

  (:predicates\n    {preds_str}
  )

  {ops_strs}
)"""


def create_pddl_problem(objects: Collection[Object],
                        init_atoms: Collection[GroundAtom],
                        goal: Collection[GroundAtom], domain_name: str,
                        problem_name: str) -> str:
    """Create a PDDL problem str."""
    # Sort everything to ensure determinism.
    objects_lst = sorted(objects)
    init_atoms_lst = sorted(init_atoms)
    goal_lst = sorted(goal)
    objects_str = "\n    ".join(f"{o.name} - {o.type.name}"
                                for o in objects_lst)
    init_str = "\n    ".join(atom.pddl_str() for atom in init_atoms_lst)
    goal_str = "\n    ".join(atom.pddl_str() for atom in goal_lst)
    return f"""(define (problem {problem_name}) (:domain {domain_name})
  (:objects\n    {objects_str}
  )
  (:init\n    {init_str}
  )
  (:goal (and {goal_str}))
)
"""


def fig2data(fig: matplotlib.figure.Figure, dpi: int = 150) -> Image:
    """Convert matplotlib figure into Image."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    data = np.frombuffer(
        fig.canvas.tostring_argb(),  # type: ignore
        dtype=np.uint8).copy()
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4, ))
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]
    return data


def save_video(outfile: str, video: Video) -> None:
    """Save the video to video_dir/outfile."""
    outdir = CFG.video_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, outfile)
    imageio.mimwrite(outpath, video, fps=CFG.video_fps)
    print(f"Wrote out to {outpath}")


def update_config(args: Dict[str, Any], default_seed: int = 123) -> None:
    """Args is a dictionary of new arguments to add to the config CFG."""
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
    # Maintain the invariant that CFG has some seed set. This is very useful
    # in unit tests, where there are often no commandline args being passed, so
    # no seed is being set. We always want a seed set because environments and
    # approaches use the seed during construction.
    if "seed" not in CFG.__dict__:
        CFG.__setattr__("seed", default_seed)


def get_config_path_str() -> str:
    """Get a filename prefix for configuration based on the current CFG."""
    return f"{CFG.env}__{CFG.approach}__{CFG.seed}__{CFG.excluded_predicates}"


def get_save_path_str() -> str:
    """Get a path for saving and loading models."""
    if not os.path.exists(CFG.save_dir):
        os.makedirs(CFG.save_dir)
    return f"{CFG.save_dir}/{get_config_path_str()}.saved"


def parse_args() -> Dict[str, Any]:
    """Parses command line arguments."""
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
    """Print all info for this experiment."""
    print(f"Seed: {args.seed}")
    print(f"Env: {args.env}")
    print(f"Approach: {args.approach}")
    print(f"Timeout: {args.timeout}")
    print()


def flush_cache() -> None:
    """Clear all lru caches."""
    gc.collect()
    wrappers = [
        a for a in gc.get_objects()
        if isinstance(a, functools._lru_cache_wrapper)  # pylint: disable=protected-access
    ]

    for wrapper in wrappers:
        wrapper.cache_clear()
