"""Online learning of generalized policies via PG3.

Example command line:
    python predicators/main.py --approach online_pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --explorer random_options \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 10 \
        --min_data_for_nsrt 10
"""
from __future__ import annotations

from typing import List, Sequence, Set

from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.approaches.pg3_approach import PG3Approach
from predicators.structs import Box, Dataset, InteractionResult, \
    ParameterizedOption, Predicate, Task, Type


class OnlinePG3Approach(PG3Approach, OnlineNSRTLearningApproach):
    """OnlinePG3Approach implementation."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        # Initializes the generalized policy.
        PG3Approach.__init__(self, initial_predicates, initial_options, types,
                             action_space, train_tasks)
        # Initializes the cumulative dataset.
        OnlineNSRTLearningApproach.__init__(self, initial_predicates,
                                            initial_options, types,
                                            action_space, train_tasks)

    @classmethod
    def get_name(cls) -> str:
        return "online_pg3"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Update the dataset with the offline data.
        self._dataset = Dataset(dataset.trajectories)
        # Learn NSRTs and generalized policy.
        return PG3Approach.learn_from_offline_dataset(self, dataset)

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # This does three things: adds data to self._dataset, re-learns NSRTs,
        # and advances the online learning cycle counter.
        old_nsrts = self._nsrts
        OnlineNSRTLearningApproach.learn_from_interaction_results(
            self, results)
        save_cycle = self._online_learning_cycle - 1
        # Then, relearn the generalized policy, but only if the NSRTs have
        # changed, because LDL learning is only a function of the NSRTs.
        if old_nsrts != self._nsrts:
            self._learn_ldl(online_learning_cycle=save_cycle)
