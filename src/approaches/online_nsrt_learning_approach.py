"""An NSRT learning approach that collects and learns from data online.

Example command:
    python src/main.py --env cover --approach online_nsrt_learning --seed 0 \
        --num_train_tasks 1 --num_test_tasks 10
"""
from __future__ import annotations

import abc
import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Callable

import dill as pkl
from gym.spaces import Box
import numpy as np

from predicators.src import utils
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.nsrt_learning.nsrt_learning_main import \
    learn_nsrts_from_data
from predicators.src.planning import task_plan, task_plan_grounding
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Dataset, LowLevelTrajectory, \
    ParameterizedOption, Predicate, Segment, Task, Type, InteractionRequest, \
    _GroundNSRT, State, Action, DummyOption


class OnlineNSRTLearningApproach(NSRTLearningApproach):
    """OnlineNSRTLearningApproach implementation."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._dataset = Dataset([])
        self._online_learning_cycle = 0

    @classmethod
    def get_name(cls) -> str:
        return "online_nsrt_learning"

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # Explore in the train tasks. The number of train tasks that are
        # explored at each time step is a hyperparameter. The train task
        # is randomly selected.
        explorer = self._get_explorer()
        requests = []
        for _ in range(CFG.online_nsrt_learning_tasks_per_request):
            # Select a random task (with replacement).
            task_idx = self._rng.choice(len(self._train_tasks))
            task = self._train_tasks[task_idx]
            # Set up the explorer policy and termination function.
            policy, terminal = explorer.solve(task)
            # Create the interaction request.
            request = InteractionRequest(train_task_idx=task_idx,
                                         act_policy=policy,
                                         query_policy=lambda s: None,
                                         termination_function=terminal)
            requests.append(request)
        return requests

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Update the dataset with the offline data.
        self._dataset = Dataset(dataset.trajectories)
        super().learn_from_offline_dataset(dataset)

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # Add the new data to the cumulative dataset.
        for result in results:
            traj = LowLevelTrajectory(result.states, result.actions)
            self._dataset.append(traj)
        # Re-learn the NSRTs.
        self._learn_nsrts(self._dataset.trajectories, self._online_learning_cycle)
        # Advance the online learning cycle.
        self._online_learning_cycle += 1

    def _get_explorer(self) -> _Explorer:
        predicates = self._get_current_predicates()
        nsrts = self._get_current_nsrts()
        
        if CFG.online_nsrt_learning_explorer == "random_nsrts":
            return _RandomNSRTsExplorer(predicates, nsrts, self._action_space)

        raise NotImplementedError("Unrecognized explorer: "
                                  f"{CFG.online_nsrt_learning_explorer}.")


################################## Explorers ##################################

class _Explorer(abc.ABC):
    """Creates a policy and termination function for exploring in a task."""

    def __init__(self, predicates: Set[Predicate], nsrts: Set[NSRT], action_space: Box) -> None:
        self._predicates = predicates
        self._action_space = action_space
        self._nsrts = nsrts
        self._rng = np.random.default_rng(CFG.seed)

    @abc.abstractmethod
    def solve(self, task: Task) -> Tuple[Callable[[State], Action],
                                         Callable[[State], bool]]:
        """Given a task, create a policy and termination function."""
        raise NotImplementedError("Override me!")


class _RandomNSRTsExplorer(_Explorer):
    """Explores by selecting random applicable NSRTs."""

    def solve(self, task: Task) -> Tuple[Callable[[State], Action],
                                         Callable[[State], bool]]:
        # Ground all NSRTs with the objects in this task.
        ground_nsrts: List[_GroundNSRT] = []
        for nsrt in sorted(self._nsrts):
            ground_nsrts.extend(utils.all_ground_nsrts(nsrt, list(task.init)))

        cur_option = DummyOption

        def _policy(s: State) -> Action:
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(s):
                # Unset the current option.
                cur_option = DummyOption
                # Find an applicable NSRT.
                atoms = utils.abstract(s, self._predicates)
                applicable_nsrts = list(
                    utils.get_applicable_operators(ground_nsrts, atoms))
                if len(applicable_nsrts) == 0:
                    # Default to a completely random action.
                    logging.warning("WARNING: Explorer falling back to random!")
                    return Action(self._action_space.sample())
                idx = self._rng.choice(len(applicable_nsrts))
                ground_nsrt = applicable_nsrts[idx]
                # Sample a random option.
                option = ground_nsrt.sample_option(s, task.goal, self._rng)
                # If the option is not initiable, fall back to random. We could
                # sample multiple times instead, but that would be slower and
                # more complicated, and this should be rare anyway.
                if not option.initiable(s):
                    logging.warning("WARNING: Explorer falling back to random!")
                    return Action(self._action_space.sample())
                # We successfully found a new option.
                cur_option = option
            act = cur_option.policy(s)
            return act

        # Termination is left to the environment, as in
        # CFG.max_num_steps_interaction_request.
        _termination_function = lambda _: False

        return _policy, _termination_function
