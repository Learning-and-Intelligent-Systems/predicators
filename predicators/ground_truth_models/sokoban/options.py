"""Ground-truth options for the sokoban environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.screws import ScrewsEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class SokobanGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the sokoban environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"sokoban"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Reference for discrete actions:
        # https://github.com/mpSchrader/gym-sokoban
        discrete_action_names = [
            "PushUp", "PushDown", "PushLeft", "PushRight", "MoveUp",
            "MoveDown", "MoveLeft", "MoveRight"
        ]

        options = {
            utils.SingletonParameterizedOption(
                name, cls._create_policy(discrete_action=(i + 1)))
            for i, name in enumerate(discrete_action_names)
        }

        return options

    @classmethod
    def _create_policy(cls, discrete_action: int) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            arr = np.zeros(9, dtype=np.float32)
            arr[discrete_action] = 1
            return Action(arr)

        return policy
