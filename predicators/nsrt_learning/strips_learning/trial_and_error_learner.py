"""Learns operators via trial and error, using the various algorithms for STRIPS
learning in this directory."""

from pathos.multiprocessing import ProcessPool
from multiprocess import TimeoutError

from typing import List
from predicators.structs import PNAD
from predicators.settings import CFG

from predicators.nsrt_learning.strips_learning.clustering_learner import ClusterAndIntersectSTRIPSLearner
from predicators.nsrt_learning.strips_learning.pnad_search_learner import PNADSearchSTRIPSLearner

class TrialAndErrorLearner(PNADSearchSTRIPSLearner, ClusterAndIntersectSTRIPSLearner):
    """Tries pnad_search, and if it times out, runs cluster_and_intersect."""

    def _learn(self) -> List[PNAD]:
        pool = ProcessPool(1)
        result = pool.apipe(PNADSearchSTRIPSLearner._learn, self)
        
        try:
            result = result.get(timeout=CFG.trial_and_error_timeout)
            print("pnad_search successful.")
            return result

        except TimeoutError:
            print("pnad_search timed out. Running cluster_and_intersect.")
            pool.terminate()
            pool.restart()

        return ClusterAndIntersectSTRIPSLearner._learn(self)

    @classmethod
    def get_name(cls) -> str:
        return "trial_and_error"
