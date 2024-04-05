"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class AppleCoringGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the (non-pybullet) blocks environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"apple_coring"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        del env_name, predicates  # unused.

        object_type = types["object"]
        goal_object_type = types["goal_object"]
        apple_type = types["apple"]
        slicing_tool_type = types["slicing_tool"]
        plate_type = types["plate"]
        hand_type = types["hand"]

        PickApple = utils.SingletonParameterizedOption(
            # variables: [object to pick]
            # params: []
            "pick",
            cls._create_dummy_policy(action_space),
            types=[object_type])

        PlaceOn = utils.SingletonParameterizedOption(
            # variables: [object to pick, thing to place on]
            # params: []
            "place_on",
            cls._create_dummy_policy(action_space),
            types=[apple_type, plate_type])

        Slice = utils.SingletonParameterizedOption(
            # variables: [tool to slice with, obj to slice, robot]
            # params: []
            "slice",
            cls._create_dummy_policy(action_space),
            types=[slicing_tool_type, apple_type, hand_type])

        return {PickApple, PlaceOn, Slice}

    @classmethod
    def _create_dummy_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
