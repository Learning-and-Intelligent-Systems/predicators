"""Ground-truth NSRTs for the pybullet shelf environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.pybullet_shelf import PyBulletShelfEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class PyBulletShelfGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the pybullet shelf environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_shelf"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        import ipdb
        ipdb.set_trace()
