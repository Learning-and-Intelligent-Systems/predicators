"""Oracle for STRIPS learning.
"""

import abc
import logging
from typing import List, Sequence, Set, cast

from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.envs import get_or_create_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.structs import DummyOption, LiftedAtom, Segment, \
    PartialNSRTAndDatastore, Predicate, LowLevelTrajectory, STRIPSOperator, \
    VarToObjSub, Task, Datastore
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner

class OracleSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for an oracle STRIPS learner.
    """
    def _learn(self) -> List[PartialNSRTAndDatastore]:
        env = get_or_create_env(CFG.env)
        gt_nsrts = get_gt_nsrts(env.predicates, env.options)
        pnads: List[PartialNSRTAndDatastore] = []
        for nsrt in gt_nsrts:
            op = STRIPSOperator(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                nsrt.add_effects, nsrt.delete_effects,
                                nsrt.side_predicates)
            datastore: Datastore = []
            option_spec = (nsrt.option, list(nsrt.option_vars))
            pnads.append(PartialNSRTAndDatastore(
                op, datastore, option_spec))
        return pnads
