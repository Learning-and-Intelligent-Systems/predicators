"""Ground-truth options for the cluttered table environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type


class ClutteredTableGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cluttered table environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cluttered_table"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Grasp
        can_type = types["can"]

        def _Grasp_policy(state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        Grasp = utils.SingletonParameterizedOption("Grasp",
                                                   _Grasp_policy,
                                                   types=[can_type],
                                                   params_space=Box(
                                                       0, 1, (4, )))

        # Dump
        def _Dump_policy(state: State, memory: Dict, objects: Sequence[Object],
                         params: Array) -> Action:
            del state, memory, objects, params  # unused
            return Action(np.zeros(
                4, dtype=np.float32))  # no parameter for dumping

        Dump = utils.SingletonParameterizedOption("Dump", _Dump_policy)

        return {Grasp, Dump}


class ClutteredTablePlaceGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cluttered table place environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cluttered_table_place"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Policy is the same in both cases
        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        # Grasp
        can_type = types["can"]

        Grasp = utils.SingletonParameterizedOption("Grasp",
                                                   _policy,
                                                   types=[can_type],
                                                   params_space=Box(
                                                       np.array([0, 0, 0, 0]),
                                                       np.array([1, 1, 1, 1])))

        # Place
        Place = utils.SingletonParameterizedOption("Place",
                                                   _policy,
                                                   types=[can_type],
                                                   params_space=Box(
                                                       np.array([0, 0, 0, 0]),
                                                       np.array([1, 1, 1, 1])))

        return {Grasp, Place}
