"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class PyBulletGrowGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the grow environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}
