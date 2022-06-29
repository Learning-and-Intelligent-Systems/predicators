"""An NSRT learning approach that collects and learns from data online.

Example command:
    python src/main.py --approach online_nsrt_learning --seed 0 \
        --env cover \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 10 \
        --min_data_for_nsrt 10
"""
from __future__ import annotations

import abc
from typing import Callable, List, Sequence, Set, Tuple

from gym.spaces import Box

from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.approaches.random_options_approach import \
    RandomOptionsApproach
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Action, Dataset, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type


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
        # explored at each timestep is a hyperparameter. The train task
        # is randomly selected.
        explorer = self._get_explorer()
        requests = []
        for _ in range(CFG.online_nsrt_learning_tasks_per_cycle):
            # Select a random task (with replacement).
            task_idx = self._rng.choice(len(self._train_tasks))
            task = self._train_tasks[task_idx]
            # Set up the explorer policy and termination function.
            policy, termination_function = explorer.solve(task)
            # Create the interaction request.
            req = InteractionRequest(train_task_idx=task_idx,
                                     act_policy=policy,
                                     query_policy=lambda s: None,
                                     termination_function=termination_function)
            requests.append(req)
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
        self._learn_nsrts(self._dataset.trajectories,
                          self._online_learning_cycle)
        # Advance the online learning cycle.
        self._online_learning_cycle += 1

    def _get_explorer(self) -> _Explorer:
        predicates = self._get_current_predicates()
        options = self._initial_options
        types = self._types
        action_space = self._action_space
        nsrts = self._get_current_nsrts()

        if CFG.online_nsrt_learning_explorer == "random_options":
            return _RandomOptionsExplorer(predicates, options, types,
                                          action_space, self._train_tasks,
                                          nsrts)

        raise NotImplementedError("Unrecognized explorer: "
                                  f"{CFG.online_nsrt_learning_explorer}.")


################################## Explorers ##################################


class _Explorer(abc.ABC):
    """Creates a policy and termination function for exploring in a task."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 nsrts: Set[NSRT]) -> None:
        self._predicates = predicates
        self._options = options
        self._types = types
        self._action_space = action_space
        self._train_tasks = train_tasks
        self._nsrts = nsrts

    @abc.abstractmethod
    def solve(
        self, task: Task
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Given a task, create a policy and termination function."""
        raise NotImplementedError("Override me!")


class _RandomOptionsExplorer(_Explorer):
    """Explores by selecting random options."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 nsrts: Set[NSRT]) -> None:
        super().__init__(predicates, options, types, action_space, train_tasks,
                         nsrts)
        # Reuse the logic that's implemented in the random options approach.
        self._random_options_approach = RandomOptionsApproach(
            predicates, options, types, action_space, train_tasks)

    def solve(
        self, task: Task
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        # Get the random options policy.
        policy = self._random_options_approach.solve(task, timeout=CFG.timeout)
        # Termination is left to the environment, as in
        # CFG.max_num_steps_interaction_request.
        termination_function = lambda _: False
        return policy, termination_function
