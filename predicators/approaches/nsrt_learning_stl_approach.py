from predicators.structs import Dataset
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach

class NSRTLearningSTLApproach(NSRTLearningApproach):
    """Naive incremental NSRT learning that learns new NSRTs for each demonstrated task"""

    @property
    def supports_incremental_learning(self) -> bool:
       return True
    
    @classmethod
    def get_name(cls) -> str:
        return "nsrt_stl_learning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
       old_nsrts, old_segmented_trajs, old_seg_to_nsrt = \
         self._nsrts, self._segmented_trajs, self._seg_to_nsrt
       super().learn_from_offline_dataset(dataset)
       self._nsrts = old_nsrts.union(self._nsrts)
       self._segmented_trajs = old_segmented_trajs + self._segmented_trajs
       self._seg_to_nsrt = {**old_seg_to_nsrt, **self._seg_to_nsrt}