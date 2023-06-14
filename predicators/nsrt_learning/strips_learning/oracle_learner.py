"""Oracle for STRIPS learning."""

import logging
from typing import List

from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import PNAD, Datastore, DummyOption


class OracleSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for an oracle STRIPS learner."""

    def _learn(self) -> List[PNAD]:
        env = get_or_create_env(CFG.env)
        env_options = get_gt_options(env.get_name())
        gt_nsrts = get_gt_nsrts(env.get_name(), env.predicates, env_options)
        pnads: List[PNAD] = []
        for nsrt in gt_nsrts:
            datastore: Datastore = []
            # If options are unknown, use a dummy option spec.
            if CFG.option_learner == "no_learning":
                option_spec = (nsrt.option, list(nsrt.option_vars))
            else:
                option_spec = (DummyOption.parent, [])
            pnads.append(PNAD(nsrt.op, datastore, option_spec))
        self._recompute_datastores_from_segments(pnads)
        # If we are using oracle sampler and option learning in addition to
        # oracle strips learning, then we do not need to filter out PNADs with
        # no data. But if we are sampler learning, we do need to filter,
        # because sampler learning will crash if there are PNADs without data.
        if CFG.sampler_learner == "oracle" and \
           CFG.option_learner == "no_learning":
            return pnads
        # Filter out any pnad that has an empty datastore. This can occur when
        # using non-standard settings with environments that cause certain
        # operators to be unnecessary. For example, in painting, when using
        # `--painting_goal_receptacles box`, the operator for picking from
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
