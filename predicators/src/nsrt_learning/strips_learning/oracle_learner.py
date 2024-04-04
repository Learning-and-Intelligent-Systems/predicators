"""Oracle for STRIPS learning."""

from typing import List

from predicators.src.envs import get_or_create_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.src.settings import CFG
from predicators.src.structs import Datastore, DummyOption, \
    PartialNSRTAndDatastore, STRIPSOperator


class OracleSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for an oracle STRIPS learner."""

    def _learn(self) -> List[PartialNSRTAndDatastore]:
        env = get_or_create_env(CFG.env)
        gt_nsrts = get_gt_nsrts(env.predicates, env.options)
        pnads: List[PartialNSRTAndDatastore] = []
        for nsrt in gt_nsrts:
            op = STRIPSOperator(nsrt.name, nsrt.parameters, nsrt.preconditions,
                                nsrt.add_effects, nsrt.delete_effects,
                                nsrt.side_predicates)
            datastore: Datastore = []
            # If options are unknown, use a dummy option spec.
            if CFG.option_learner == "no_learning":
                option_spec = (nsrt.option, list(nsrt.option_vars))
            else:
                option_spec = (DummyOption.parent, [])
            pnads.append(PartialNSRTAndDatastore(op, datastore, option_spec))
        self._recompute_datastores_from_segments(pnads)
        return pnads

    @classmethod
    def get_name(cls) -> str:
        return "oracle"
