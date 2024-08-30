"""Ground-truth options for the sokoban environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box
from minigrid.core.actions import Actions

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class MiniGridGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the minigrid environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"minigrid_env"}

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
