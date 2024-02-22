"""STRIPS learner that leverages access to oracle operators used to generate
demonstrations via bilevel planning."""

from typing import List, Set, Tuple

from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.nsrt_learning.strips_learning import BaseSTRIPSLearner
from predicators.settings import CFG
from predicators.structs import NSRT, PNAD, Datastore, DummyOption, \
    LiftedAtom, ParameterizedOption, Variable


class KnownPNADsSTRIPSLearner(BaseSTRIPSLearner):
    """Base class for a STRIPS learner that uses oracle operators but re-learns
    all the components via currently-implemented methods in the base class.

    This is different from the oracle learner because here, we assume
    that our demo data is annotated with the ground-truth operators used
    to produce it. We thus know exactly how to associate (i.e, cluster)
    demos into sets corresponding to each operator.
    """

    def _learn(self) -> List[PNAD]:
        return self._known_pnads

    @classmethod
    def get_name(cls) -> str:
        return "known_pnads"
