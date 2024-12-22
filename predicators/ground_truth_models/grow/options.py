"""Ground-truth options for the coffee environment."""

import logging
from functools import lru_cache
from typing import ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    get_change_fingers_action, get_move_end_effector_to_pose_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type, _TypedEntity


class PyBulletGrowGroundTruthOptionFactory(GroundTruthOptionFactory):

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_grow"}
    
    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        return set()