"""This directory contains algorithms for STRIPS operator learning."""

from typing import Any, List, Optional, Set

from predicators import utils
from predicators.nsrt_learning.strips_learning.base_strips_learner import \
    BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, LowLevelTrajectory, Predicate, Segment, \
    Task

__all__ = ["BaseSTRIPSLearner"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def learn_strips_operators(trajectories: List[LowLevelTrajectory],
                           train_tasks: List[Task],
                           predicates: Set[Predicate],
                           segmented_trajs: List[List[Segment]],
                           verify_harmlessness: bool,
                           annotations: Optional[List[Any]],
                           verbose: bool = True) -> List[PNAD]:
    """Learn strips operators on the given data segments.

    Return a list of PNADs with op (STRIPSOperator), datastore, and
    option_spec fields filled in (but not sampler).
    """
    for cls in utils.get_all_subclasses(BaseSTRIPSLearner):
        if not cls.__abstractmethods__ and \
           cls.get_name() == CFG.strips_learner:
            learner = cls(trajectories, train_tasks, predicates,
                          segmented_trajs, verify_harmlessness, annotations,
                          verbose)
            break
    else:
        raise ValueError(f"Unrecognized STRIPS learner: {CFG.strips_learner}")
    return learner.learn()
