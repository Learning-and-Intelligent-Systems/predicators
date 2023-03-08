"""Ground-truth options for the pybullet shelf environment."""

from typing import ClassVar, Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.playroom import PlayroomEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, \
    ParameterizedInitiable, ParameterizedOption, ParameterizedPolicy, \
    Predicate, State, Type


class PyBulletShelfGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the pybullet shelf environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_shelf"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # TODO
        return set()
