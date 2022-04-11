"""This directory contains algorithms for STRIPS operator learning.
"""

from typing import List, Sequence, Set, cast

from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.structs import DummyOption, LiftedAtom, Segment, \
    PartialNSRTAndDatastore, Predicate, LowLevelTrajectory, STRIPSOperator, \
    VarToObjSub, Task
from predicators.src.nsrt_learning.strips_learning.base_strips_learner import \
    BaseSTRIPSLearner
from predicators.src.nsrt_learning.strips_learning.clustering_learner import \
    ClusterAndIntersectSTRIPSLearner, ClusterAndSearchSTRIPSLearner, \
    ClusterAndIntersectSidelinePredictionErrorSTRIPSLearner, \
    ClusterAndIntersectSidelineHarmlessnessSTRIPSLearner
from predicators.src.nsrt_learning.strips_learning.general_to_specific_learner \
    import BackchainingSTRIPSLearner
from predicators.src.nsrt_learning.strips_learning.oracle_learner import \
    OracleSTRIPSLearner


def learn_strips_operators(
        trajectories: List[LowLevelTrajectory],
        train_tasks: List[Task], predicates: Set[Predicate],
        segmented_trajs: List[List[Segment]],
        verbose: bool = True,
) -> List[PartialNSRTAndDatastore]:
    """Learn strips operators on the given data segments.

    Return a list of PNADs with op (STRIPSOperator), datastore, and
    option_spec fields filled in (but not sampler).
    """
    if CFG.strips_learner == "cluster_and_intersect":
        cls: BaseSTRIPSLearner = ClusterAndIntersectSTRIPSLearner
    elif CFG.strips_learner == "cluster_and_intersect_sideline_prederror":
        cls = ClusterAndIntersectSidelinePredictionErrorSTRIPSLearner
    elif CFG.strips_learner == "cluster_and_intersect_sideline_harmlessness":
        cls = ClusterAndIntersectSidelineHarmlessnessSTRIPSLearner
    elif CFG.strips_learner == "cluster_and_search":
        cls = ClusterAndSearchSTRIPSLearner
    elif CFG.strips_learner == "backchaining":
        cls = BackchainingSTRIPSLearner
    elif CFG.strips_learner == "oracle":
        cls = OracleSTRIPSLearner
    else:
        raise Exception(f"Unrecognized STRIPS learner: {CFG.strips_learner}")
    learner = cls(trajectories, train_tasks, predicates, segmented_trajs,
                  verbose)
    return learner.learn()
