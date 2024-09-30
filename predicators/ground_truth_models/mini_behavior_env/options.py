"""Ground-truth options for the sokoban environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from enum import IntEnum
from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type

class Actions(IntEnum):
    left = 0
    right = 1
    forward = 2
    toggle = 3
    open = 4
    close = 5
    slice = 6
    cook = 7
    drop_in = 8
    pickup_0 = 9
    pickup_1 = 10
    pickup_2 = 11
    drop_0 = 12
    drop_1 = 13
    drop_2 = 14

class MiniBehaviorGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the MiniBehavior environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"mini_behavior_env"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Reformat names for consistency with other option naming.
        def _format_name(name: str) -> str:
            return "".join([n.capitalize() for n in name.split(" ")])

        options: Set[ParameterizedOption] = {
            utils.SingletonParameterizedOption(
                _format_name(name), cls._create_policy(discrete_action=i))
            for i, name in {value: key for key, value in Actions.__members__.items()}.items()
        }

        # FindObj option.
        object_type = types["obj"]
        FindObjOption = ParameterizedOption(
                                "FindObj",
                                [object_type],
                                Box(low=np.array([]), high=np.array([]), shape=(0, )),
                                policy=cls._create_find_obj_policy(),
                                initiable=lambda s, m, o, p: True,
                                terminal=lambda s, m, o, p: s.get(o[0], "type") == 8 and s.get(o[0], "state") != -1) # 8 is the goal enum type
        options.add(FindObjOption)

        # ReplanToObj option.
        ReplanToObj = utils.SingletonParameterizedOption("ReplanToObj", cls._create_policy(discrete_action=6))
        options.add(ReplanToObj)
        
        return options

    @classmethod
    def _create_policy(cls, discrete_action: int) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused.
            arr = np.zeros(7, dtype=np.float32)
            arr[discrete_action] = 1
            return Action(arr)

        return policy
    
    @classmethod
    def _create_find_obj_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused.
            arr = np.zeros(7, dtype=np.float32)
            arr[np.random.choice([0, 1, 2], 1, p=[0.2, 0.2, 0.6])[0]] = 1
            return Action(arr)

        return policy
