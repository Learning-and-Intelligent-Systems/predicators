from dataclasses import dataclass, field
import time
from typing import Any, Callable, Iterator, List, Optional, Self, Set, Tuple, TypeVar, TypeVarTuple, Union
from gym.spaces import Box
import numpy as np
from predicators.approaches.bilevel_planning_approach import BilevelPlanningApproach
from predicators.option_model import _OptionModelBase
from predicators.planning import _DiscoveredFailure, _DiscoveredFailureException
from predicators.settings import CFG
from predicators.structs import NSRT, _GroundNSRT, _Option, Action, Dataset, DefaultState, DummyOption, GroundAtom, LowLevelTrajectory, Metrics, Object, ParameterizedOption, Predicate, Segment, State, Task, Type
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
class BacktrackingAttempt:
    option: _Option
    num_actions: Optional[int]
    next_state: Optional[State]
    refinement_time: float

@dataclass
class BacktrackingStep:
    _state: State
    _attempts: List[BacktrackingAttempt] = field(default_factory=list, init=False)
    _num_actions: int = field(default=0, init=False)

    @property
    def num_tries(self) -> int:
        return len(self.attempts)

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def state(self) -> State:
        return self._state

    def __iter__(self) -> Iterator[BacktrackingAttempt]:
        return iter(self._attempts)

    def append_attempt(self, option: _Option, num_actions: Optional[int], next_state: Optional[State], start_time: float):
        end_time = time.perf_counter()
        self._num_actions = num_actions
        self.attempts.append(BacktrackingAttempt(option, num_actions, next_state, end_time - start_time))


def run_low_level_search(
    task: Task,
    option_model: _OptionModelBase,
    skeleton: List[_GroundNSRT],
    atoms_sequence: List[Set[GroundAtom]],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int,
) -> Tuple[List[_Option], bool]:
    """Backtracking search over continuous values.

    Returns a sequence of options and a boolean. If the boolean is True,
    the option sequence is a complete low-level plan refining the given
    skeleton. Otherwise, the option sequence is the longest partial
    failed refinement, where the last step did not satisfy the skeleton,
    but all previous steps did. Note that there are multiple low-level
    plans in general; we return the first one found (arbitrarily).
    """
    assert len(skeleton) + 1 == len(atoms_sequence) or not CFG.sesame_check_expected_atoms

    start_time = time.perf_counter()
    rng_sampler = np.random.default_rng(seed)
    assert CFG.sesame_propagate_failures in \
        {"after_exhaust", "immediately", "never"}
    backtracking: List[BacktrackingStep] = [BacktrackingStep(task.init)]
    longest_failed_refinement: List[_Option] = []
    # Optimization: if the params_space for the NSRT option is empty, only
    # sample it once, because all samples are just empty (so equivalent).
    max_tries = [
        CFG.sesame_max_samples_per_step
        if nsrt.option.params_space.shape[0] > 0 else 1 for nsrt in skeleton
    ]
    # We'll use a maximum of one discovered failure per step, since
    # resampling can render old discovered failures obsolete.
    discovered_failures: List[Optional[_DiscoveredFailure]] = [
        None for _ in skeleton
    ]
    while cur_idx < len(skeleton):
        if time.perf_counter() - start_time > timeout:
            return longest_failed_refinement, False
        assert backtracking[-1].num_tries < max_tries[len(backtracking) - 1]
        try_start_time = time.perf_counter()
        # REPLACE num_tries[cur_idx] += 1
        nsrt = skeleton[len(backtracking) - 1]
        # Ground the NSRT's ParameterizedOption into an _Option.
        # This invokes the NSRT's sampler.
        option = nsrt.sample_option(state, task.goal, rng_sampler)
        # REPLACE plan[cur_idx] = option
        # Increment num_samples metric by 1
        metrics["num_samples"] += 1
        # Increment cur_idx. It will be decremented later on if we get stuck.
        # REPLACE cur_idx += 1
        state = backtracking[-1]._state
        next_state = None
        num_actions = None
        if option.initiable(state):
            try:
                next_state, num_actions = \
                    option_model.get_next_state_and_num_actions(state, option)
            except EnvironmentFailure as e:
                can_continue_on = False
                # Remember only the most recent failure.
                discovered_failures[len(backtracking) - 1] = _DiscoveredFailure(e, nsrt)
            else:  # an EnvironmentFailure was not raised
                discovered_failures[len(backtracking) - 1] = None
                # REPLACE: num_actions_per_option[cur_idx - 1] = num_actions
                # Check if objects that were outside the scope had a change
                # in state.
                if _check_static_object_changed(set(state) - set(nsrt.objects), state, next_state):
                    can_continue_on = False
                # Check if we have exceeded the horizon.
                elif sum(map(lambda step: step.num_actions, backtracking)) > max_horizon:
                    can_continue_on = False
                # Check if the option was effectively a noop.
                elif num_actions == 0:
                    can_continue_on = False
                elif CFG.sesame_check_expected_atoms:
                    # This "if all" statement is equivalent to, but faster
                    # than, checking whether expected_atoms is a subset of
                    # utils.abstract(next_state, predicates).
                    if all(a.holds(next_state) for a in atoms_sequence[len(backtracking)]):
                        can_continue_on = True
                    else:
                        can_continue_on = False
                # If we're not checking expected_atoms, we need to
                # explicitly check the goal on the final timestep.
                elif cur_idx == len(skeleton) and not task.goal_holds(next_state):
                    can_continue_on = False
                else:
                    # If we're not checking expected_atoms, we need to
                    # explicitly check the goal on the final timestep.
                    can_continue_on = True
        else:
            # The option is not initiable.
            can_continue_on = False

        # Record the backtracking step
        backtracking[-1].append_attempt(option, num_actions, next_state, try_start_time)

        if can_continue_on:
            assert next_state is not None and num_actions is not None
            backtracking.append(BacktrackingStep(next_state))
        else:  # we got stuck, time to resample / backtrack!
            # Update the longest_failed_refinement found so far.
            if cur_idx > len(longest_failed_refinement):
                longest_failed_refinement = list(plan[:cur_idx])
            # If we're immediately propagating failures, and we got a failure,
            # raise it now. We don't do this right after catching the
            # EnvironmentFailure because we want to make sure to update
            # the longest_failed_refinement first.
            _make_discovered_failure_exception(
                "immediately", [discovered_failures[len(backtracking) - 1]], longest_failed_refinement
            )
            while backtracking[-1] == max_tries[cur_idx]:
                num_tries[cur_idx] = 0
                plan[cur_idx] = DummyOption
                num_actions_per_option[cur_idx] = 0
                traj[cur_idx + 1] = DefaultState
                cur_idx -= 1
                if cur_idx < 0:
                    # Backtracking exhausted. If we're only propagating failures
                    # after exhaustion, and if there are any failures,
                    # propagate up the EARLIEST one so that high-level search
                    # restarts. Otherwise, return a partial refinement so that
                    # high-level search continues.
                    _make_discovered_failure_exception(
                        "after_exhaust", discovered_failures, longest_failed_refinement
                    )
                    return longest_failed_refinement, False
    return backtracking, True

def _make_discovered_failure_exception(
        mode: str,
        possible_failures: List[Optional[_DiscoveredFailure]],
        longest_failed_refinement: List[_Option]
    ):
    if CFG.sesame_propagate_failures == mode:
        for possible_failure in possible_failures:
            if possible_failure is not None:
                raise _DiscoveredFailureException(
                                "Discovered a failure", possible_failure, {
                                    "longest_failed_refinement":
                                    longest_failed_refinement
                                })


def _check_static_object_changed(static_objs: Set[Object], state: State, next_state: State) -> bool:
    if not CFG.sesame_check_static_object_changes:
        return False
    for obj in sorted(static_objs):
        if not np.allclose(
                next_state[obj],
                state[obj],
                atol=CFG.sesame_static_object_change_tol):
            return True
    return False