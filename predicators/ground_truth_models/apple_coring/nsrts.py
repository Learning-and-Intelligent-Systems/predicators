"""Ground-truth NSRTs for the blocks environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class AppleCoringGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the apple_coring environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"apple_coring"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # For now, there are just no NSRTs
        return set()
