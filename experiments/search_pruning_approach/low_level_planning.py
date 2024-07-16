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
    _tries: List[Tuple[_Option, Optional['BacktrackingTree']]] = field(default_factory=list)

    _num_successful_tries: int = 0

    _longest_failure: Optional[Tuple[_Option, 'BacktrackingTree']] = None
    _longest_failure_length: int = 0

    def __post_init__(self):
        self._frozen = False

    @property
    def num_failed_tries(self) -> int:
        return len(self._tries) - self._num_successful_tries

    @property
    def num_successful_tries(self) -> int:
        return self._num_successful_tries

    @property
    def is_successful(self) -> bool:
        return self._is_successful

    @property
    def num_tries(self) -> int:
        return len(self._tries)

    @classmethod
    def create_leaf(cls, state: State) -> 'BacktrackingTree':
        return BacktrackingTree(state, _is_successful=True)

    @property
    def state(self) -> State:
        return self._state

    @property
    def failed_tries(self) -> Iterator[Tuple[_Option, Optional['BacktrackingTree']]]:
        for option, mb_tree in self._tries:
            if mb_tree is None or not mb_tree.is_successful:
                yield option, mb_tree

    @property
    def successful_tries(self) -> Iterator[Tuple[_Option, 'BacktrackingTree']]:
        for option, mb_tree in self._tries:
            if mb_tree is not None and mb_tree.is_successful:
                yield option, mb_tree

    @property
    def tries(self) -> Iterator[Tuple[_Option, Optional['BacktrackingTree']]]:
        return iter(self._tries)

    def append_try(self, option: _Option, mb_tree: Optional['BacktrackingTree']) -> None:
        assert not self._frozen

        if mb_tree is not None:
            mb_tree._frozen = True
            self._is_successful |= mb_tree.is_successful

        self._tries.append((option, mb_tree))

        if mb_tree is not None and mb_tree.is_successful:
            self._num_successful_tries += 1

        if mb_tree is not None and not mb_tree.is_successful and mb_tree._longest_failure_length + 1 > self._longest_failure_length:
            self._longest_failure_length = mb_tree._longest_failure_length + 1
            self._longest_failure = (option, mb_tree)

    @property
    def successful_trajectory(self) -> Tuple[List[State], List[_Option]]:
        return self._visit(lambda tree: next(tree.successful_tries) if tree.is_successful else None)

    @property
    def longest_failure(self) -> Tuple[List[State], List[_Option]]:
        states, options = self._visit(lambda tree: tree._longest_failure)
        assert len(states) == self._longest_failure_length + 1
        return states, options

    def _visit(tree: 'BacktrackingTree', f: Callable[['BacktrackingTree'], Optional[Tuple[_Option, 'BacktrackingTree']]]) -> Tuple[List[State], List[_Option]]:
        states: List[State] = []
        options: List[_Option] = []
        while True:
            states.append(tree.state)
            mb_next_action_tree = f(tree)
            if mb_next_action_tree is None:
                return states, options
            option, tree = mb_next_action_tree
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
    assert not CFG.sesame_check_expected_atoms or (len(skeleton) + 1 == len(atoms_sequence) and task.goal.issubset(atoms_sequence[-1]))
    end_time = time.perf_counter() + timeout
    rng_sampler = np.random.default_rng(seed)
    # Optimization: if the params_space for the NSRT option is empty, only
    # sample it once, because all samples are just empty (so equivalent).
    def search_stop_condition(states: List[State], tree: BacktrackingTree) -> int:
        if tree.is_successful:
            return -1
        current_depth = len(states) - 1
        if skeleton[current_depth].option.params_space.shape[0] == 0:
            return current_depth - 1 if tree.num_failed_tries else current_depth
        return current_depth - 1 if tree.num_failed_tries >= CFG.sesame_max_samples_per_step else current_depth

    logging.info([(nsrt.name, nsrt.objects) for nsrt in skeleton])

    tree, _ = _backtrack(
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
    return tree, tree.is_successful

def run_backtracking_for_data_generation(
    previous_states: List[State],
    goal: Set[GroundAtom],
    option_model: _OptionModelBase,
    skeleton: List[_GroundNSRT],
    feasibility_classifier: Optional[FeasibilityClassifier],
    atoms_sequence: List[Set[GroundAtom]],
    search_stop_condition: Callable[[List[State], BacktrackingTree], int],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int,
) -> Tuple[BacktrackingTree, bool]: # TODO: update docstring
    """ Asssumes that the previous states could have been generated by the given skeleton
    """
    assert all(atom.holds(state) for atoms, state in zip(atoms_sequence, previous_states) for atom in atoms)
    assert not CFG.sesame_check_expected_atoms or (len(skeleton) + 1 == len(atoms_sequence) and goal.issubset(atoms_sequence[-1]))

    end_time = time.perf_counter() + timeout
    rng_sampler = np.random.default_rng(seed)
    tree, _ = _backtrack(
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
        return tree, False
    return tree, tree.is_successful

def run_backjumping_low_level_search(
    task: Task,
    option_model: _OptionModelBase,
    skeleton: List[_GroundNSRT],
    backjumping_model: Callable[[List[State]], int],
    atoms_sequence: List[Set[GroundAtom]],
    seed: int,
    timeout: float,
    metrics: Metrics,
    max_horizon: int,
) -> Tuple[BacktrackingTree, bool]:
    assert not CFG.sesame_check_expected_atoms or (len(skeleton) + 1 == len(atoms_sequence) and task.goal.issubset(atoms_sequence[-1]))
    end_time = time.perf_counter() + timeout
    rng_sampler = np.random.default_rng(seed)
    def search_stop_condition(states: List[State], tree: BacktrackingTree) -> int:
        if tree.is_successful:
            return -1
        current_depth = len(states) - 1

        # Optimization: if the params_space for the NSRT option is empty, only
        # sample it once, because all samples are just empty (so equivalent).
        if skeleton[current_depth].option.params_space.shape[0] == 0:
            return current_depth - 1 if tree.num_failed_tries else current_depth

        if tree.num_failed_tries < CFG.sesame_max_samples_per_step:
            return current_depth

        # Edge case for if there are no states for the backjumping model
        if current_depth == 0:
            return -1

        return backjumping_model(states[1:])

    logging.info([(nsrt.name, nsrt.objects) for nsrt in skeleton])

    tree, _ = _backtrack(
        [task.init],
        max_horizon,
        skeleton,
        None,
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
    return tree, tree.is_successful

def _backtrack(
    states: List[State],
    max_horizon: int,
    skeleton: List[_GroundNSRT],
    feasibility_classifier: Optional[FeasibilityClassifier],
    min_confidence: float,
    atoms_sequence: List[Set[GroundAtom]],
    search_stop_condition: Callable[[List[State], BacktrackingTree], int],
    goal: Set[GroundAtom],
    option_model: _OptionModelBase,
    rng_sampler: np.random.Generator,
    metrics: Metrics,
    end_time: float,
) -> Tuple[BacktrackingTree, int]:
    assert len(states) >= 1

    if "num_samples" not in metrics:
        metrics["num_samples"] = 0

    current_depth = len(states) - 1
    max_depth = len(skeleton)
    current_state = states[-1]

    tree = BacktrackingTree(current_state)
    if current_depth == len(skeleton):
        if all(goal_atom.holds(current_state) for goal_atom in goal):
            return BacktrackingTree.create_leaf(current_state), current_depth
        return tree, current_depth
    if time.perf_counter() > end_time:
        return tree, current_depth

    nsrt = skeleton[current_depth]
    desired_depth = current_depth
    while True:
        if desired_depth == current_depth:
            desired_depth = search_stop_condition(states, tree)
        assert desired_depth <= current_depth
        if current_depth != desired_depth:
            break

        try_start_time = time.perf_counter()
        # Ground the NSRT's ParameterizedOption into an _Option.
        # This invokes the NSRT's sampler.
        option = nsrt.sample_option(current_state, goal, rng_sampler, skeleton[current_depth:])
        metrics["num_samples"] += 1
        next_state, num_actions = option_model.get_next_state_and_num_actions(current_state, option)

        if _check_static_object_changed(set(current_state) - set(nsrt.objects), current_state, next_state) or\
          num_actions > max_horizon or num_actions == 0:
            logging.info(f"Depth {current_depth}/{max_depth} State not changed")
            tree.append_try(option, None)
            continue
        if CFG.sesame_check_expected_atoms and not all(a.holds(next_state) for a in atoms_sequence[current_depth + 1]):
            logging.info(f"Depth {current_depth}/{max_depth} Expected atoms do not hold")
            tree.append_try(option, None)
            continue

        next_states = states + [next_state]
        confidence = 2.0
        if feasibility_classifier is not None and len(next_states) < len(skeleton) + 1:
            feasible, confidence = feasibility_classifier.classify(next_states, skeleton)
            if not feasible:
                logging.info(f"Depth {current_depth}/{max_depth} Feasibility classifier does not hold")
                tree.append_try(option, BacktrackingTree(next_state))
                continue
            else:
                logging.info(f"Depth {current_depth}/{max_depth} Feasibility classifier holds")

        logging.info(f"Depth {current_depth}/{max_depth} Classifier Confidence {confidence}")

        try_end_time = time.perf_counter()
        refinement_time = try_end_time - try_start_time

        next_tree, new_desired_depth = _backtrack(
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
        desired_depth = min(new_desired_depth, desired_depth)
        tree.append_try(option, next_tree)

        if time.perf_counter() > end_time: # Timeout handling
            logging.info(f"Depth {current_depth}/{max_depth} Timeout")
            return tree, current_depth

        if next_tree.is_successful:
            logging.info(f"Depth {current_depth}/{max_depth} Subtree Succeeded")
            continue

        if feasibility_classifier is not None and confidence > min_confidence and CFG.feasibility_do_backjumping: # Feasibility Classifier related backjumping handling
            assert desired_depth == current_depth
            logging.info(f"Depth {current_depth}/{max_depth} Backjumping, Current Confidence {confidence}, Min Confidence {float(min_confidence)}")
            return tree, desired_depth

        logging.info(f"Depth {current_depth}/{max_depth} Subtree Failed")

    return tree, desired_depth

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