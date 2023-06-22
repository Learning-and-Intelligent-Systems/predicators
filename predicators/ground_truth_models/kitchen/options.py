"""Ground-truth options for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type

from mujoco_kitchen.utils import primitive_and_params_to_primitive_action, primitive_idx_to_name, primitive_name_to_action_idx


class KitchenGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the Kitchen environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # Reformat names for consistency with other option naming.
        def _format_name(name: str) -> str:
            return "".join([n.capitalize() for n in name.split(" ")])

        options: Set[ParameterizedOption] = set()
        for val in primitive_idx_to_name.values():
            if type(primitive_name_to_action_idx[val]) == int:
                param_space = Box(-np.ones(1), np.ones(1))
            else:
                n = len(primitive_name_to_action_idx[val])
                param_space = Box(-np.ones(n), np.ones(n))
            options.add(utils.SingletonParameterizedOption(
                _format_name(val), cls._create_policy(name=val), params_space=param_space))

        return options

    @classmethod
    def _create_policy(cls, name: str) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused.
            arr = primitive_and_params_to_primitive_action(name, params)
            return Action(arr)

        return policy
