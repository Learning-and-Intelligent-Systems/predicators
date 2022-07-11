"""Online learning of generalized policies via PG3.

Example command line:
    python3 src/main.py --approach online_pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --explorer random_options \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 10 \
        --min_data_for_nsrt 10
"""
from __future__ import annotations

print()
print("imported annotations")
print()

from typing import List, Sequence, Set

print()
print("imported typing yeet")
print()

from predicators.src.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
    
print()
print("Import OnlineNSRTLearningApproach")
print()   
    

from predicators.src.approaches.pg3_approach import PG3Approach

print()
print("import PG3Approach")
print()

from predicators.src.structs import Box, Dataset, InteractionResult, \
    ParameterizedOption, Predicate, Task, Type
    
print()
print("import predicators structs")
print()


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
        
        print()
        print("initilizing...")
        print()

    @classmethod
    def get_name(cls) -> str:
        print()
        print("get_name")
        print()
        
        return "online_pg3"
    

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Update the dataset with the offline data.
        self._dataset = Dataset(dataset.trajectories)
        # Learn NSRTs and generalized policy.
        print()
        print("learn from offline dataset function")
        print()
        return PG3Approach.learn_from_offline_dataset(self, dataset)
    

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        # This does three things: adds data to self._dataset, re-learns NSRTs,
        # and advances the online learning cycle counter.
        OnlineNSRTLearningApproach.learn_from_interaction_results(
            self, results)
        # Then, relearn the generalized policy.
        save_cycle = self._online_learning_cycle - 1
        self._learn_ldl(online_learning_cycle=save_cycle)
        print()
        print("learn_from_interaction_results")
