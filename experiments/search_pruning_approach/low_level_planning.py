from dataclasses import dataclass, field
import time
from typing import Callable, Iterator, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
from experiments.search_pruning_approach.learning import FeasibilityClassifier
from experiments.shelves2d import Shelves2DEnv
from predicators.option_model import _OptionModelBase
from predicators.planning import _DiscoveredFailure, _DiscoveredFailureException
from predicators.settings import CFG
from predicators.structs import _GroundNSRT, _Option, GroundAtom, Metrics, Object, State, Task
from predicators.utils import EnvironmentFailure

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")

@dataclass
class BacktrackingTree:
    _state: State
    _successful_try: Optional[Tuple[_Option, 'BacktrackingTree', float]] = None
    _failed_tries: List[Tuple[_Option, Optional['BacktrackingTree']]] = field(default_factory=list)
    _longest_failure: Optional[Tuple[_Option, 'BacktrackingTree']] = None
    _longest_failure_length: int = 0

    def __post_init__(self):
        self._frozen = False

    @property
    def num_failed_children(self) -> int:
        return len(self._failed_tries)

    @property
    def state(self) -> State:
        return self._state

    @property
    def failed_tries(self) -> Iterator[Tuple[_Option, Optional['BacktrackingTree']]]:
        return iter(self._failed_tries)

    @property
    def successful_try(self) -> Optional[Tuple[_Option, 'BacktrackingTree', float]]:
        return self._successful_try

    def append_failed_search(self, option: _Option, tree: Optional['BacktrackingTree']) -> None:
        """ Adds a failed backtracking branch. Takes the ownership of the tree.
        """
        assert not self._frozen
        if tree is not None:
            tree._frozen = True
        self._failed_tries.append((option, tree))
        if tree is not None and tree._longest_failure_length + 1 > self._longest_failure_length:
            self._longest_failure_length = tree._longest_failure_length + 1
            self._longest_failure = (option, tree)

    def set_successful_try(self, option: _Option, tree: 'BacktrackingTree', refinement_time: float) -> None:
        assert self._successful_try is None and not self._frozen
        tree._frozen = True
        self._successful_try = (option, tree, refinement_time)

    @property
    def successful_trajectory(self) -> Tuple[List[State], List[_Option]]:
        def visitor(tree: 'BacktrackingTree'):
            if tree._successful_try is None:
                return None
            return tree.successful_try[0:2]
        return self._visit(visitor)

    @property
    def longest_failuire(self) -> Tuple[List[State], List[_Option]]:
        states, options = self._visit(lambda tree: tree._longest_failure)
        assert len(states) == self._longest_failure_length
        return states, options

    def _visit(tree: 'BacktrackingTree', f: Callable[['BacktrackingTree'], Optional[Tuple[_Option, 'BacktrackingTree']]]) -> Tuple[List[State], List[_Option]]:
        states: List[State] = []
        options: List[_Option] = []
        while True:
            states.append(tree.state)
            next_action_tree = f(tree)
            if next_action_tree is None:
                return states, options
            option, tree = next_action_tree
            options.append(option)

def run_low_level_search(
    task: Task,
    option_model: _OptionModelBase,
    skeleton: List[_GroundNSRT],
    feasibility: Optional[FeasibilityClassifier],
    atoms_sequence: List[Set[GroundAtom]],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int
) -> Tuple[BacktrackingTree, bool]:
    """Backtracking search over continuous values.

    Returns a sequence of options and a boolean. If the boolean is True,
    the option sequence is a complete low-level plan refining the given
    skeleton. Otherwise, the option sequence is the longest partial
    failed refinement, where the last step did not satisfy the skeleton,
    but all previous steps did. Note that there are multiple low-level
    plans in general; we return the first one found (arbitrarily).
    """
    assert CFG.sesame_propagate_failures in \
        {"after_exhaust", "immediately", "never"}
    assert not CFG.sesame_check_expected_atoms or (len(skeleton) + 1 == len(atoms_sequence) and task.goal.issubset(atoms_sequence[-1]))
    end_time = time.perf_counter() + timeout
    rng_sampler = np.random.default_rng(seed)
    # Optimization: if the params_space for the NSRT option is empty, only
    # sample it once, because all samples are just empty (so equivalent).
    max_tries = [
        CFG.sesame_max_samples_per_step if nsrt.option.params_space.shape[0] > 0 else 1 for i, nsrt in enumerate(skeleton)
    ]
    tree, success, mb_failure = _backtrack(
        [task.init],
        max_horizon,
        skeleton,
        feasibility,
        atoms_sequence,
        max_tries,
        task.goal,
        option_model,
        rng_sampler,
        metrics,
        end_time,
    )
    if mb_failure:
        failure, _ = mb_failure
        raise _DiscoveredFailureException(
            "Discovered a failure", failure,
            {"longest_failed_refinement": tree.longest_failuire})
    return tree, success

def run_backtracking_with_previous_states(
    previous_states: List[State],
    goal: Set[GroundAtom],
    option_model: _OptionModelBase,
    skeleton: List[_GroundNSRT],
    feasibility: Optional[FeasibilityClassifier],
    atoms_sequence: List[Set[GroundAtom]],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int
) -> Tuple[BacktrackingTree, bool]:
    """ Asssumes that the previous states could have been generated by the given skeleton
    """
    assert CFG.sesame_propagate_failures in \
        {"after_exhaust", "immediately", "never"}
    assert all(atom.holds(state) for atoms, state in zip(atoms_sequence, previous_states) for atom in atoms)
    assert not CFG.sesame_check_expected_atoms or (len(skeleton) + 1 == len(atoms_sequence) and goal.issubset(atoms_sequence[-1]))

    end_time = time.perf_counter() + timeout
    rng_sampler = np.random.default_rng(seed)
    # Optimization: if the params_space for the NSRT option is empty, only
    # sample it once, because all samples are just empty (so equivalent).
    max_tries = [
        CFG.sesame_max_samples_per_step if nsrt.option.params_space.shape[0] > 0 else 1
        for i, nsrt in enumerate(skeleton)
    ]
    tree, success, mb_failure = _backtrack(
        previous_states,
        max_horizon,
        skeleton,
        feasibility,
        atoms_sequence,
        max_tries,
        goal,
        option_model,
        rng_sampler,
        metrics,
        end_time,
    )
    if mb_failure:
        failure, _ = mb_failure
        raise _DiscoveredFailureException(
            "Discovered a failure", failure,
            {"longest_failed_refinement": tree.longest_failuire})
    return tree, success


def _backtrack(
    states: List[State],
    max_horizon: int,
    skeleton: List[_GroundNSRT],
    feasibility: Optional[FeasibilityClassifier],
    atoms_sequence: List[Set[GroundAtom]],
    max_tries: List[int],
    goal: Set[GroundAtom],
    option_model: _OptionModelBase,
    rng_sampler: np.random.Generator,
    metrics: Metrics,
    end_time: float
) -> Tuple[BacktrackingTree, bool, Optional[Tuple[_DiscoveredFailure, int]]]:
    # Shelves2DEnv.render_state_plt(states[-1], None)
    # plt.show()
    if "num_samples" not in metrics:
        metrics["num_samples"] = 0
    assert len(max_tries) == len(skeleton) and len(states) >= 1
    current_depth = len(states) - 1
    current_state = states[-1]
    tree = BacktrackingTree(current_state)
    if current_depth == len(skeleton):
        return tree, all(goal_atom.holds(current_state) for goal_atom in goal), None
    if time.perf_counter() > end_time:
        return tree, False, None

    env_failure: Optional[_DiscoveredFailure] = None
    deepest_next_env_failure: Optional[_DiscoveredFailure] = None
    deepest_next_env_failure_depth: int = len(skeleton) + 1

    nsrt = skeleton[current_depth]
    for iter in range(max_tries[current_depth]):
        try_start_time = time.perf_counter()
        # Ground the NSRT's ParameterizedOption into an _Option.
        # This invokes the NSRT's sampler.
        option = nsrt.sample_option(current_state, goal, rng_sampler)
        metrics["num_samples"] += 1
        try:
            next_state, num_actions = \
                option_model.get_next_state_and_num_actions(current_state, option)
        except EnvironmentFailure as e:
            # logging.info(f"Depth {current_depth} Environment failure")
            tree.append_failed_search(option, None)
            if CFG.sesame_propagate_failures == "immediately":
                return tree, False, (_DiscoveredFailure(e, nsrt), current_depth)
            elif CFG.sesame_propagate_failures == "after_exhaust":
                env_failure = _DiscoveredFailure(e, nsrt)
            continue

        if _check_static_object_changed(set(current_state) - set(nsrt.objects), current_state, next_state) or\
           num_actions > max_horizon or num_actions == 0:
            # logging.info(f"Depth {current_depth} State not changed")
            tree.append_failed_search(option, None)
            continue
        if CFG.sesame_check_expected_atoms and not all(a.holds(next_state) for a in atoms_sequence[current_depth + 1]):
            # logging.info(f"Depth {current_depth} Expected atoms do not hold")
            tree.append_failed_search(option, None) # REMOVED FAILURE MANAGEMENT FROM HERE
            continue

        next_states = states + [next_state]
        if feasibility is not None and len(next_states) < len(skeleton) + 1:
            if not feasibility(next_states, skeleton):
                # logging.info(f"Depth {current_depth} Feasibility classifier does not hold")
                tree.append_failed_search(option, None)
                continue


        try_end_time = time.perf_counter()
        refinement_time = try_end_time - try_start_time

        next_tree, success, mb_next_env_failure = _backtrack(
            next_states,
            max_horizon - num_actions,
            skeleton,
            feasibility,
            atoms_sequence,
            max_tries,
            goal,
            option_model,
            rng_sampler,
            metrics,
            end_time
        )
        if success:
            tree.set_successful_try(option, next_tree, refinement_time)
            return tree, True, None
        # logging.info(f"Depth {current_depth} Subtree Failed")
        tree.append_failed_search(option, next_tree)

        if not mb_next_env_failure:
            continue
        if CFG.sesame_propagate_failures == "immediately":
            return tree, False, mb_next_env_failure
        elif CFG.sesame_propagate_failures == "after_exhaust":
            _, next_env_failure_depth = mb_next_env_failure
            if next_env_failure_depth <= deepest_next_env_failure_depth:
                deepest_next_env_failure, deepest_next_env_failure_depth = mb_next_env_failure

    if env_failure is not None:
        return tree, False, (env_failure, current_depth)
    elif deepest_next_env_failure is not None:
        return tree, False, (deepest_next_env_failure, deepest_next_env_failure_depth)
    else:
        return tree, False, None

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