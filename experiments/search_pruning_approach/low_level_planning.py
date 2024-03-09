from dataclasses import dataclass, field
import time
from types import SimpleNamespace
from typing import Callable, Iterator, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt
from experiments.search_pruning_approach.learning import FeasibilityClassifier
from predicators.option_model import _OptionModelBase
from predicators.planning import _DiscoveredFailure, _DiscoveredFailureException, PlanningTimeout
from predicators.settings import CFG
from predicators.structs import _GroundNSRT, _Option, GroundAtom, Metrics, Object, State, Task
from predicators.utils import EnvironmentFailure
import logging

# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("tkagg")

@dataclass
class BacktrackingTree:
    """Collects all states encountered during backtracking
    """

    _state: State
    _is_successful: bool = False
    _successful_tries: List[Tuple[_Option, 'BacktrackingTree', float]] = field(default_factory=list)
    _failed_tries: List[Tuple[_Option, Optional['BacktrackingTree']]] = field(default_factory=list)
    _longest_failure: Optional[Tuple[_Option, 'BacktrackingTree']] = None
    _longest_failure_length: int = 0

    def __post_init__(self):
        self._frozen = False

    @property
    def num_failed_tries(self) -> int:
        return len(self._failed_tries)

    @property
    def num_successful_tries(self) -> int:
        return len(self._successful_tries)

    @property
    def is_successful(self) -> bool:
        return self._is_successful

    @property
    def num_tries(self) -> int:
        return len(self._successful_tries) + len(self._failed_tries)

    @classmethod
    def create_leaf(self, state: State) -> 'BacktrackingTree':
        return BacktrackingTree(state, _is_successful=True)

    @property
    def state(self) -> State:
        return self._state

    @property
    def failed_tries(self) -> Iterator[Tuple[_Option, Optional['BacktrackingTree']]]:
        return iter(self._failed_tries)

    @property
    def successful_tries(self) -> Iterator[Tuple[_Option, 'BacktrackingTree', float]]:
        return iter(self._successful_tries)

    def append_failed_try(self, option: _Option, tree: Optional['BacktrackingTree']) -> None:
        """ Adds a failed backtracking branch. Takes the ownership of the tree.
        """
        assert not self._frozen
        if tree is not None:
            tree._frozen = True
        self._failed_tries.append((option, tree))
        if tree is not None and tree._longest_failure_length + 1 > self._longest_failure_length:
            self._longest_failure_length = tree._longest_failure_length + 1
            self._longest_failure = (option, tree)

    def append_successful_try(self, option: _Option, tree: 'BacktrackingTree', refinement_time: float) -> None:
        assert not self._frozen
        tree._frozen = True
        self._is_successful = True
        self._successful_tries.append((option, tree, refinement_time))

    @property
    def successful_trajectory(self) -> Optional[Tuple[List[State], List[_Option]]]:
        def visitor(tree: 'BacktrackingTree'):
            if not tree._successful_tries:
                return None
            assert tree._is_successful
            option, tree, _ = tree._successful_tries[0]
            return option, tree
        return self._visit(visitor)

    @property
    def longest_failuire(self) -> Tuple[List[State], List[_Option]]:
        states, options = self._visit(lambda tree: tree._longest_failure)
        assert len(states) == self._longest_failure_length + 1
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
    feasibility_classifier: Optional[FeasibilityClassifier],
    atoms_sequence: List[Set[GroundAtom]],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int,
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
    def search_stop_condition(current_depth: int, tree: BacktrackingTree) -> bool:
        if tree.is_successful:
            return True
        if skeleton[current_depth].option.params_space.shape[0] == 0:
            return bool(tree.num_failed_tries)
        return tree.num_failed_tries >= CFG.sesame_max_samples_per_step

    tree, mb_failure = _backtrack(
        [task.init],
        max_horizon,
        skeleton,
        feasibility_classifier,
        2.0,
        atoms_sequence,
        search_stop_condition,
        task.goal,
        option_model,
        rng_sampler,
        metrics,
        end_time,
    )
    if time.perf_counter() > end_time:
        raise PlanningTimeout("Backtracking timed out!", info={'backtracking_tree': tree})
    if mb_failure:
        failure, _ = mb_failure
        raise _DiscoveredFailureException(
            "Discovered a failure", failure,
            {"longest_failed_refinement": tree.longest_failuire})
    return tree, tree.is_successful

def run_backtracking_for_data_generation(
    previous_states: List[State],
    goal: Set[GroundAtom],
    option_model: _OptionModelBase,
    skeleton: List[_GroundNSRT],
    feasibility_classifier: Optional[FeasibilityClassifier],
    atoms_sequence: List[Set[GroundAtom]],
    search_stop_condition: Callable[[int, BacktrackingTree], bool],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int,
) -> Tuple[BacktrackingTree, bool]: # TODO: update docstring
    """ Asssumes that the previous states could have been generated by the given skeleton
    """
    assert CFG.sesame_propagate_failures in \
        {"after_exhaust", "immediately", "never"}
    assert all(atom.holds(state) for atoms, state in zip(atoms_sequence, previous_states) for atom in atoms)
    assert not CFG.sesame_check_expected_atoms or (len(skeleton) + 1 == len(atoms_sequence) and goal.issubset(atoms_sequence[-1]))

    end_time = time.perf_counter() + timeout
    rng_sampler = np.random.default_rng(seed)
    tree, mb_failure = _backtrack(
        previous_states,
        max_horizon,
        skeleton,
        feasibility_classifier,
        2.0,
        atoms_sequence,
        search_stop_condition,
        goal,
        option_model,
        rng_sampler,
        metrics,
        end_time,
    )
    if time.perf_counter() > end_time:
        raise PlanningTimeout("Backtracking timed out!", info={'backtracking_tree': tree})
    if mb_failure:
        failure, _ = mb_failure
        raise _DiscoveredFailureException(
            "Discovered a failure", failure,
            {"longest_failed_refinement": tree.longest_failuire})
    return tree, tree.is_successful


def _backtrack( # TODO: add comments and docstring
    states: List[State],
    max_horizon: int,
    skeleton: List[_GroundNSRT],
    feasibility_classifier: Optional[FeasibilityClassifier],
    min_confidence: float,
    atoms_sequence: List[Set[GroundAtom]],
    search_stop_condition: Callable[[int, BacktrackingTree], bool],
    goal: Set[GroundAtom],
    option_model: _OptionModelBase,
    rng_sampler: np.random.Generator,
    metrics: Metrics,
    end_time: float,
) -> Tuple[BacktrackingTree, Optional[Tuple[_DiscoveredFailure, int]]]:
    # Shelves2DEnv.render_state_plt(states[-1], None)
    # plt.show()
    if "num_samples" not in metrics:
        metrics["num_samples"] = 0
    assert len(states) >= 1
    current_depth = len(states) - 1
    current_state = states[-1]
    tree = BacktrackingTree(current_state)
    if current_depth == len(skeleton):
        if all(goal_atom.holds(current_state) for goal_atom in goal):
            return BacktrackingTree.create_leaf(current_state), None
        return tree, None
    if time.perf_counter() > end_time:
        return tree, None

    env_failure: Optional[_DiscoveredFailure] = None
    deepest_next_env_failure: Optional[_DiscoveredFailure] = None
    deepest_next_env_failure_depth: int = len(skeleton) + 1

    nsrt = skeleton[current_depth]
    while not search_stop_condition(current_depth, tree):
        try_start_time = time.perf_counter()
        # Ground the NSRT's ParameterizedOption into an _Option.
        # This invokes the NSRT's sampler.
        option = nsrt.sample_option(current_state, goal, rng_sampler, skeleton[current_depth:])
        metrics["num_samples"] += 1
        try:
            next_state, num_actions = \
                option_model.get_next_state_and_num_actions(current_state, option)
        except EnvironmentFailure as e:
            logging.info(f"Depth {current_depth} Environment failure")
            tree.append_failed_try(option, None)
            if CFG.sesame_propagate_failures == "immediately":
                return tree, (_DiscoveredFailure(e, nsrt), current_depth)
            elif CFG.sesame_propagate_failures == "after_exhaust":
                env_failure = _DiscoveredFailure(e, nsrt)
            continue

        if _check_static_object_changed(set(current_state) - set(nsrt.objects), current_state, next_state) or\
          num_actions > max_horizon or num_actions == 0:
            logging.info(f"Depth {current_depth} State not changed")
            tree.append_failed_try(option, None)
            continue
        if CFG.sesame_check_expected_atoms and not all(a.holds(next_state) for a in atoms_sequence[current_depth + 1]):
            logging.info(f"Depth {current_depth} Expected atoms do not hold")
            tree.append_failed_try(option, None) # REMOVED FAILURE MANAGEMENT FROM HERE
            # if current_depth == len(skeleton) - 1 and iter >= max_tries[current_depth] // 2:
            #     Shelves2DEnv.render_state_plt(next_state, None).suptitle(skeleton[-1].name)
            #     plt.show()
            continue

        next_states = states + [next_state]
        confidence = 2.0
        if feasibility_classifier is not None and len(next_states) < len(skeleton) + 1:
            feasible, confidence = feasibility_classifier.classify(next_states, skeleton)
            if not feasible:
                logging.info(f"Depth {current_depth} Feasibility classifier does not hold")
                # if iter >= max_tries[current_depth] // 2:
                    # Shelves2DEnv.render_state_plt(next_state, None).suptitle(skeleton[-1].name)
                    # plt.show()
                tree.append_failed_try(option, None)
                continue
            else:
                logging.info(f"Depth {current_depth} Feasibility classifier holds")
                # Shelves2DEnv.render_state_plt(next_state, None).suptitle(skeleton[-1].name)
                # plt.show()

        try_end_time = time.perf_counter()
        refinement_time = try_end_time - try_start_time

        next_tree, mb_next_env_failure = _backtrack(
            next_states,
            max_horizon - num_actions,
            skeleton,
            feasibility_classifier,
            min(min_confidence, confidence),
            atoms_sequence,
            search_stop_condition,
            goal,
            option_model,
            rng_sampler,
            metrics,
            end_time,
        )

        if time.perf_counter() > end_time: # Timeout handling
            logging.info(f"Depth {current_depth} Timeout")
            return tree, None

        if next_tree.is_successful:
            logging.info(f"Depth {current_depth} Subtree Succeeded")
            tree.append_successful_try(option, next_tree, refinement_time)
            continue

        tree.append_failed_try(option, next_tree)

        if feasibility_classifier is not None and confidence > min_confidence: # Backjumping handling
            logging.info(f"Depth {current_depth} Backjumping")
            return tree, None

        logging.info(f"Depth {current_depth} Subtree Failed")

        if not mb_next_env_failure:
            continue
        if CFG.sesame_propagate_failures == "immediately":
            return tree, mb_next_env_failure
        elif CFG.sesame_propagate_failures == "after_exhaust":
            _, next_env_failure_depth = mb_next_env_failure
            if next_env_failure_depth <= deepest_next_env_failure_depth:
                deepest_next_env_failure, deepest_next_env_failure_depth = mb_next_env_failure

    if env_failure is not None:
        return tree, (env_failure, current_depth)
    elif deepest_next_env_failure is not None:
        return tree, (deepest_next_env_failure, deepest_next_env_failure_depth)
    else:
        return tree, None

def _check_static_object_changed(static_objs: Set[Object], state: State, next_state: State) -> bool:
    if not CFG.sesame_check_static_object_changes:
        return False
    for obj in sorted(static_objs):
        if not np.allclose(
                next_state[obj],
                state[obj],
                atol=CFG.tol):
            return True
    return False