"""An NSRT learning approach that collects and learns from data online.

Example command:
    python src/main.py --approach online_nsrt_learning --seed 0 \
        --env cover \
        --explorer random_options \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 10 \
        --min_data_for_nsrt 10
"""
from __future__ import annotations

from typing import List, Sequence, Set

from gym.spaces import Box

from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.explorers import create_explorer
from predicators.src.settings import CFG
from predicators.src.structs import Dataset, InteractionRequest, \
    InteractionResult, LowLevelTrajectory, ParameterizedOption, Predicate, \
    Task, Type


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

        # Create the explorer. Note that GLIB and greedy lookahead are not yet
        # supported.
        explorer = create_explorer(CFG.explorer,
                                   self._get_current_predicates(),
                                   self._initial_options, self._types,
                                   self._action_space, self._train_tasks,
                                   self._get_current_nsrts(),
                                   self._option_model)

        # Create the interaction requests.
        requests = []
        for _ in range(CFG.online_nsrt_learning_requests_per_cycle):
            # Select a random task (with replacement).
            task_idx = self._rng.choice(len(self._train_tasks))
            # Set up the explorer policy and termination function.
            policy, termination_function = explorer.get_exploration_strategy(
                task_idx, CFG.timeout)
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
