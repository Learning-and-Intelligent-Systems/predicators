from typing import List, Set
from gym.spaces import Box
from predicators.structs import Dataset, ParameterizedOption, \
    Predicate, Segment, Task, Type
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach

class NSRTLearningMTLApproach(NSRTLearningApproach):
    """ Naive incremental NSRT learning that re-learns over all data"""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._accumulated_dataset = Dataset([])

    @property
    def supports_incremental_learning(self) -> bool:
       return True

    @classmethod
    def get_name(cls) -> str:
        return "nsrt_mtl_learning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
       assert len(dataset.trajectories) == 1
       assert not dataset.has_annotations
       self._accumulated_dataset.append(dataset.trajectories[0])
       super().learn_from_offline_dataset(self._accumulated_dataset)