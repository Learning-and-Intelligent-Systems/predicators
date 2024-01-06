from dataclasses import dataclass, field
import time
from typing import Any, Callable, List, Optional, Self, Set, Tuple, TypeVar, TypeVarTuple
from gym.spaces import Box
import numpy as np
from predicators.approaches.bilevel_planning_approach import BilevelPlanningApproach
from predicators.option_model import _OptionModelBase
from predicators.planning import _DiscoveredFailure, _DiscoveredFailureException
from predicators.settings import CFG
from predicators.structs import NSRT, _GroundNSRT, _Option, Action, Dataset, DefaultState, DummyOption, GroundAtom, LowLevelTrajectory, Metrics, ParameterizedOption, Predicate, Segment, State, Task, Type
from predicators.utils import EnvironmentFailure

__all__ = ["SearchPruningApproach"]

class SearchPruningApproach(BilevelPlanningApproach):
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types, action_space, train_tasks)
        self._nsrts: Set[NSRT] = set()

    @classmethod
    def get_name(cls) -> str:
        return "search_prunning"

    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        pass

    def pseudocode():
        """
        tasks = get_tasks()
        oracle_nsrts = get_oracle_nsrts()
        base_dataset = [run_planning(task, oracle_nsrts) for task in tasks]
        learned_nsrts = learn_base_nsrt_samplers(base_dataset)
        all_feasibility_datasets = []
        for suffix_length in range(1, max(len(traj) for traj in base_dataset) + 1):
            feasibility_dataset = [
                run_planning(Task(traj[-suffix_length].state, traj.task.goal), learned_nsrts)
                for traj in base_dataset if len(traj) >= suffix_length
            ]
            all_feasibility_datasets += feasibility_dataset
            update_feasibility_classifier(learned_nsrts, base_dataset, all_feasibility_datasets)
        run_testing(learned_nsrts)
        """


_NOT_CAUSES_FAILURE = "NotCausesFailure"

@dataclass
class BacktrackingTree:
    _state: State
    _successful_try: Optional[Tuple[_Option, Self]] = None
    _failed_tries: List[Tuple[_Option, Optional[Self]]] = field(default_factory=list)
    _longest_failure: Optional[Tuple[_Option, Self]] = None
    _longest_failure_length: int = 0

    def __post_init__(self):
        self._frozen = False

    @property
    def num_failed_children(self) -> int:
        return len(self._failed_tries)

    @property
    def state(self) -> State:
        return self._state

    def append_failed_search(self, option: _Option, tree: Optional[Self]) -> None:
        """ Adds a failed backtracking branch. Takes the ownership of the tree.
        """
        assert not self._frozen
        tree._frozen = True
        self._failed_tries.append((option, tree))
        if tree._longest_failure_length + 1 > self._longest_failure_length:
            self._longest_failure_length = tree._longest_failure_length + 1
            self._longest_failure = (option, tree)

    def set_successful_try(self, option: _Option, tree: Self) -> None:
        assert self._successful_try is None and not self._frozen
        tree._frozen = True
        self._successful_try = (option, tree)

    def sample_failed_segments(self, rng: np.random.Generator) -> Tuple[List[State], List[_Option]]:
        return self._visit(lambda tree: None if not tree._failed_tries else rng.choice(tree._failed_tries))

    @property
    def successful_trajectory(self) -> Tuple[List[State], List[_Option]]:
        return self._visit(lambda tree: tree._successful_try)

    @property
    def longest_failuire(self) -> Tuple[List[State], List[_Option]]:
        traj = self._visit(lambda tree: tree._longest_failure)
        assert len(traj.actions) == self._longest_failure_length
        return traj

    def _visit(tree: Self, f: Callable[[Self], Optional[Tuple[_Option, Self]]]) -> Tuple[List[State], List[_Option]]:
        states: List[State] = []
        options: List[_Option] = []
        while True:
            states.append(tree.state)
            next_action_tree = f(tree)
            if next_action_tree is None:
                return states
            option, tree = next_action_tree
            options.append(option)

class Backtracking:
    def __init__(self, state: State) -> Self:
        self._search_nodes: List[BacktrackingTree] = [BacktrackingTree(state)]
        self._search_options: List[_Option] = []
        self._

    @property
    def num_tries(self) -> int:
        return self._search_nodes[-1].num_failed_children

    @property
    def current_state(self) -> State:
        return self._search_nodes[-1].state

    @property
    def current_depth(self) -> State:
        return len(self._search_options)

    def append(self, option: _Option, state: State) -> None:
        self._search_options.append(option)
        self._search_nodes.append(BacktrackingTree(state))

    def collapse_single_failure(self) -> None:
        assert self._search_options
        option = self._search_options.pop()
        tree = self._search_nodes.pop()
        self._search_nodes[-1].append_failed_search(option, tree)

    def collapse_failure(self) -> BacktrackingTree:
        for parent, option, child in zip(self._search_nodes[-1::-1], self._search_options[::-1], self._search_nodes[::-1]):
            parent.append_failed_search(option, child)
        root = self._search_nodes[0]
        self._search_nodes, self._search_options = [root], []
        return root

    def collapse_success(self) -> BacktrackingTree: # This class should not be used after this point
        for parent, option, child in zip(self._search_nodes[-1::-1], self._search_options[::-1], self._search_nodes[::-1]):
            parent.set_successful_try(option, child)
        root = self._search_nodes[0]
        self._search_nodes, self._search_options = [root], []
        return root

def run_motion_tree_search(
    task: Task,
    option_model: _OptionModelBase,
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int,
    refinement_time: List[float] = []
) -> Tuple[BacktrackingTree, bool]:
    """Backtracking search over continuous values.

    Returns a sequence of options and a boolean. If the boolean is True,
    the option sequence is a complete low-level plan refining the given
    skeleton. Otherwise, the option sequence is the longest partial
    failed refinement, where the last step did not satisfy the skeleton,
    but all previous steps did. Note that there are multiple low-level
    plans in general; we return the first one found (arbitrarily).
    """
    start_time = time.perf_counter()
    rng_sampler = np.random.default_rng(seed)
    assert CFG.sesame_propagate_failures in \
        {"after_exhaust", "immediately", "never"}
    backtracking = Backtracking(task.init)
    # Optimization: if the params_space for the NSRT option is empty, only
    # sample it once, because all samples are just empty (so equivalent).
    max_tries = [
        CFG.sesame_max_samples_per_step
        if nsrt.option.params_space.shape[0] > 0 else 1 for nsrt in skeleton
    ]
    # Record the refinement time distributed across each step of the skeleton
    assert len(refinement_time) == 0
    refinement_time.extend([0] * len(skeleton))
    # The number of actions taken by each option in the plan. This is to
    # make sure that we do not exceed the task horizon.
    num_actions_per_option = [0 for _ in skeleton]
    # We'll use a maximum of one discovered failure per step, since
    # resampling can render old discovered failures obsolete.
    discovered_failures: List[Optional[_DiscoveredFailure]] = [
        None for _ in skeleton
    ]
    plan_found = False
    while len(backtracking) < len(skeleton):
        if time.perf_counter() - start_time > timeout:
            return backtracking.collapse_failure(), False
        assert backtracking.front.num_tries < max_tries[backtracking.current_depth]

        try_start_time = time.perf_counter()
        state = backtracking.current_state
        nsrt = skeleton[backtracking.current_depth]
        # Ground the NSRT's ParameterizedOption into an _Option.
        # This invokes the NSRT's sampler.
        option = nsrt.sample_option(state, task.goal, rng_sampler)
        metrics["num_samples"] += 1
        # Increment cur_idx. It will be decremented later on if we get stuck.
        cur_idx += 1
        if option.initiable(state):
            try:
                next_state, num_actions = \
                    option_model.get_next_state_and_num_actions(state, option)
            except EnvironmentFailure as e:
                can_continue_on = False
                # Remember only the most recent failure.
                discovered_failures[backtracking.current_depth] = _DiscoveredFailure(e, nsrt)
            else:  # an EnvironmentFailure was not raised
                discovered_failures[backtracking.current_depth] = None
                num_actions_per_option[backtracking.current_depth] = num_actions
                traj[cur_idx] = next_state
                # Check if objects that were outside the scope had a change
                # in state.
                static_obj_changed = False
                if CFG.sesame_check_static_object_changes:
                    static_objs = set(state) - set(nsrt.objects)
                    for obj in sorted(static_objs):
                        if not np.allclose(
                                traj[cur_idx][obj],
                                traj[cur_idx - 1][obj],
                                atol=CFG.sesame_static_object_change_tol):
                            static_obj_changed = True
                            break
                if static_obj_changed:
                    can_continue_on = False
                # Check if we have exceeded the horizon.
                elif np.sum(num_actions_per_option[:cur_idx]) > max_horizon:
                    can_continue_on = False
                # Check if the option was effectively a noop.
                elif num_actions == 0:
                    can_continue_on = False
                elif CFG.sesame_check_expected_atoms:
                    # Check atoms against expected atoms_sequence constraint.
                    assert len(traj) == len(atoms_sequence)
                    # The expected atoms are ones that we definitely expect to
                    # be true at this point in the plan. They are not *all* the
                    # atoms that could be true.
                    expected_atoms = {
                        atom
                        for atom in atoms_sequence[cur_idx]
                        if atom.predicate.name != _NOT_CAUSES_FAILURE
                    }
                    # This "if all" statement is equivalent to, but faster
                    # than, checking whether expected_atoms is a subset of
                    # utils.abstract(traj[cur_idx], predicates).
                    if all(a.holds(traj[cur_idx]) for a in expected_atoms):
                        can_continue_on = True
                        if cur_idx == len(skeleton):
                            plan_found = True
                    else:
                        can_continue_on = False
                else:
                    # If we're not checking expected_atoms, we need to
                    # explicitly check the goal on the final timestep.
                    can_continue_on = True
                    if cur_idx == len(skeleton):
                        if task.goal_holds(traj[cur_idx]):
                            plan_found = True
                        else:
                            can_continue_on = False
        else:
            # The option is not initiable.
            can_continue_on = False
        try_end_time = time.perf_counter()
        refinement_time[backtracking.current_depth] += try_end_time - try_start_time

        if plan_found:
            return backtracking.collapse_success(), True  # success!

        if not can_continue_on:  # we got stuck, time to resample / backtrack!
            # If we're immediately propagating failures, and we got a failure,
            # raise it now. We don't do this right after catching the
            # EnvironmentFailure because we want to make sure to update
            # the longest_failed_refinement first.
            possible_failure = discovered_failures[backtracking.current_depth]
            if possible_failure is not None and \
                CFG.sesame_propagate_failures == "immediately":
                raise _DiscoveredFailureException(
                    "Discovered a failure", possible_failure,
                    {"longest_failed_refinement": backtracking.collapse_failure().longest_failuire})
            # Decrement cur_idx to re-do the step we just did. If num_tries
            # is exhausted, backtrack.
            while backtracking.num_tries == max_tries[backtracking.current_depth]:
                num_actions_per_option[backtracking.current_depth] = 0
                if backtracking.current_depth == 0:
                    # Backtracking exhausted. If we're only propagating failures
                    # after exhaustion, and if there are any failures,
                    # propagate up the EARLIEST one so that high-level search
                    # restarts. Otherwise, return a partial refinement so that
                    # high-level search continues.
                    failure_tree = backtracking.collapse_failure()
                    for possible_failure in discovered_failures:
                        if possible_failure is not None and \
                            CFG.sesame_propagate_failures == "after_exhaust":
                            raise _DiscoveredFailureException(
                                "Discovered a failure", possible_failure, {
                                    "longest_failed_refinement":
                                    failure_tree.longest_failuire
                                })
                    return failure_tree, False
                else:
                    backtracking.collapse_single_failure()
    # Should only get here if the skeleton was empty.
    assert not skeleton
    return backtracking.collapse_success(), True