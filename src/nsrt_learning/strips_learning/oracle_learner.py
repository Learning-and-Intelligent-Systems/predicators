"""Oracle for STRIPS learning."""

import logging
from typing import List

from predicators.src.envs import get_or_create_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.src.settings import CFG
from predicators.src.structs import Datastore, DummyOption, \
    PartialNSRTAndDatastore


class OracleSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for an oracle STRIPS learner."""

    def _learn(self) -> List[PartialNSRTAndDatastore]:
        env = get_or_create_env(CFG.env)
        gt_nsrts = get_gt_nsrts(env.predicates, env.options)
        pnads: List[PartialNSRTAndDatastore] = []
        for nsrt in gt_nsrts:
            datastore: Datastore = []
            # If options are unknown, use a dummy option spec.
            if CFG.option_learner == "no_learning":
                option_spec = (nsrt.option, list(nsrt.option_vars))
            else:
                option_spec = (DummyOption.parent, [])
            pnads.append(
                PartialNSRTAndDatastore(nsrt.op, datastore, option_spec))
        self._recompute_datastores_from_segments(pnads)
        # Filter out any pnad that has an empty datastore. This can occur when
        # using non-standard settings with environments that cause certain
        # operators to be unnecessary. For example, in painting, when using
        # --painting_goal_receptacles box, the operator for picking from
        # the side becomes unnecessary (and no demo data will cover it).
        nontrivial_pnads = []
        for pnad in pnads:
            if not pnad.datastore:
                logging.warning(f"Discarding PNAD with no data: {pnad}")
                continue
            nontrivial_pnads.append(pnad)
        return nontrivial_pnads

    @classmethod
    def get_name(cls) -> str:
        return "oracle"
