from typing import Dict, List, Set, Tuple, Type
from gym.spaces import Box
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.settings import CFG
from predicators.structs import GroundAtomTrajectory, LowLevelTrajectory, \
    NSRT, ParameterizedOption, Predicate, Segment, Task


class NSRTLearningIncrementalApproach(NSRTLearningApproach):
    """Proper incremental NSRT learning that keeps track of learned operators."""

    def __init__(self, initial_predicates: Set[Predicate], 
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # self._accumulated_dataset = Dataset([])

    @property
    def supports_incremental_learning(self) -> bool:
        return True

    @classmethod
    def get_name(cls) -> str:
        return "nsrt_incremental_learning"

    def _learn_nsrts_from_data(self,
        trajectories: List[LowLevelTrajectory], 
        ground_atom_dataset: List[GroundAtomTrajectory],
    ) -> Tuple[Set[NSRT], List[List[Segment]], Dict[Segment, NSRT]]:
        return learn_nsrts_from_data(trajectories,
                                     self._train_tasks,
                                     self._get_current_predicates(),
                                     self._initial_options,
                                     self._action_space,
                                     ground_atom_dataset,
                                     sampler_learner=CFG.sampler_learner,
                                     existing_nsrts=self._nsrts)
