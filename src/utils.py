"""General utility methods.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import functools
import itertools
import os
from collections import defaultdict
from typing import List, Callable, Tuple, Collection, Set, Sequence, Iterator, \
    Dict, FrozenSet, Any, Optional
import heapq as hq
import imageio
import matplotlib
import numpy as np
from predicators.src.structs import _Option, State, Predicate, GroundAtom, \
    Object, Type, Operator, _GroundOperator, Action, Task, ActionTrajectory, \
    OptionTrajectory, LiftedAtom, Image, Video, Variable, PyperplanFacts, \
    ObjToVarSub, VarToObjSub
from predicators.src.settings import CFG, GlobalSettings
matplotlib.use("Agg")


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


def run_policy_on_task(policy: Callable[[State], Action], task: Task,
                       simulator: Callable[[State, Action], State],
                       predicates: Collection[Predicate],
                       make_video: bool = False,
                       render: Optional[
                           Callable[[State, Task, Action], List[Image]]] = None,
                       ) -> Tuple[ActionTrajectory, Video, bool]:
    """Execute a policy on a task until goal or max steps.
    Return the state sequence and action sequence, and a bool for
    whether the goal was satisfied at the end.
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
        for _ in range(CFG.max_num_steps_check_policy):
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
    return (states, actions), video, goal_reached


def policy_solves_task(policy: Callable[[State], Action], task: Task,
                       simulator: Callable[[State, Action], State],
                       predicates: Collection[Predicate]) -> bool:
    """Return whether the given policy solves the given task.
    """
    _, _, solved = run_policy_on_task(policy, task, simulator, predicates)
    return solved


def option_to_trajectory(
        init: State,
        simulator: Callable[[State, Action], State],
        option: _Option,
        max_num_steps: int) -> ActionTrajectory:
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


def action_to_option_trajectory(act_traj: ActionTrajectory
                                ) -> OptionTrajectory:
    """Create an option trajectory from an action trajectory.
    """
    states, actions = act_traj
    assert len(states) > 0
    new_states = [states[0]]
    if len(actions) == 0:
        return new_states, []
    current_option = actions[0].get_option()
    options = [current_option]
    for s, a in zip(states[:-1], actions):
        o = a.get_option()
        # This assumes that an option is equal to another
        # option only if they're the same python object.
        if o != current_option:
            new_states.append(s)
            options.append(o)
            current_option = o
    new_states.append(states[-1])
    return new_states, options


@functools.lru_cache(maxsize=None)
def get_all_groundings(atoms: FrozenSet[LiftedAtom],
                       objects: FrozenSet[Object]
                       ) -> List[FrozenSet[GroundAtom]]:
    """Get all the ways to ground the given set of lifted atoms into
    a set of ground atoms, using the given objects.
    """
    choices = []
    for atom in atoms:
        combs = []
        for comb in get_object_combinations(
                objects, atom.predicate.types, allow_duplicates=True):
            combs.append(GroundAtom(atom.predicate, comb))
        choices.append(combs)
    # NOTE: we WON'T use a generator here because that breaks lru_cache.
    result = []
    for choice in itertools.product(*choices):
        result.append(frozenset(choice))
    return result


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


class HAddHeuristic:
    """This class is an implementation of the hADD heuristic.
    Lightly modified from pyperplan's heuristics/relaxation.py.
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
        # Sum over the heuristic values of all preconditions.
        cost = sum([self.facts[pre].distance for pre in operator.preconditions])
        # Add on operator application cost.
        return cost+operator.cost

    def calc_goal_h(self) -> float:
        """This function calculates the heuristic value of the whole goal.
        """
        return sum([self.facts[fact].distance for fact in self.goals])

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
    for d in [GlobalSettings.get_arg_specific_settings(args), args]:
        for k, v in d.items():
            CFG.__setattr__(k, v)


def get_config_path_str() -> str:
    """Create a filename prefix based on the current CFG.
    """
    return f"{CFG.env}__{CFG.approach}__{CFG.seed}"
