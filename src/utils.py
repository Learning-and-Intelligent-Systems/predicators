"""General utility methods.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import functools
import imageio
import itertools
import os
from collections import defaultdict
from typing import List, Callable, Tuple, Collection, Set, Sequence, Iterator, \
    Dict, FrozenSet, Any, Optional, cast
import heapq as hq
from predicators.src.structs import _Option, State, Predicate, GroundAtom, \
    Object, Type, Operator, _GroundOperator, Action, Task, ActionTrajectory, \
    OptionTrajectory, Image, Video
from predicators.src.settings import CFG, GlobalSettings

PyperplanFacts = FrozenSet[Tuple[str, ...]]


def run_policy_on_task(policy: Callable[[State], Action], task: Task,
                       simulator: Callable[[State, Action], State],
                       predicates: Collection[Predicate],
                       make_video: bool = False,
                       render: Optional[Callable[[State], Image]] = None,
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
    if make_video:
        assert render is not None
        render = cast(Callable[[State], Image], render)
        video.append(render(state))
    if task.goal.issubset(atoms):  # goal is already satisfied
        return (states, actions), video, True
    for _ in range(CFG.max_num_steps_check_policy):
        act = policy(state)
        state = simulator(state, act)
        atoms = abstract(state, predicates)
        actions.append(act)
        states.append(state)
        if make_video:
            render = cast(Callable[[State], Image], render)
            video.append(render(state))
        if task.goal.issubset(atoms):
            return (states, actions), video, True
    return (states, actions), video, False


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
    for i in range(max_num_steps):
        act = option.policy(state)
        act.set_option((option, i))
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
    current_option, t = actions[0].get_option()
    assert t == 0
    expected_t = 0
    options = [current_option]
    for s, a in zip(states[:-1], actions):
        o, t = a.get_option()
        if o != current_option:
            assert t == 0
            expected_t = 0
            new_states.append(s)
            options.append(o)
            current_option = o
        else:
            assert t == expected_t
            expected_t += 1
    new_states.append(states[-1])
    return new_states, options


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


def save_video(video: Video, outfile: str) -> None:
    """Save the video to video_dir/outfile.
    """
    outdir = CFG.video_dir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, outfile)
    imageio.mimwrite(video, outpath)
    print(f"Wrote out to {outpath}.")


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
