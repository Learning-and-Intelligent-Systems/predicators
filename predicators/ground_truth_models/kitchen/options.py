"""Ground-truth options for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box
from mujoco_kitchen.utils import primitive_and_params_to_primitive_action, \
    primitive_idx_to_name, primitive_name_to_action_idx

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, ParameterizedTerminal, Predicate, State, Type


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
            if isinstance(primitive_name_to_action_idx[val], int):
                param_space = Box(-np.ones(1) * 5, np.ones(1) * 5)
            else:
                n = len(primitive_name_to_action_idx[val])
                param_space = Box(-np.ones(n) * 5, np.ones(n) * 5)
            options.add(
                utils.ParameterizedOption(
                    name=_format_name(val),
                    types=list(types.values()),
                    params_space=param_space,
                    policy=cls._create_policy(name=val),
                    initiable=lambda _1, _2, _3, _4: True,
                    terminal=cls._create_terminal(name=val)))

        return options

    @classmethod
    def _create_policy(cls, name: str) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused.
            if name.lower() == "move_delta_ee_pose":
                gripper, _ = objects
                gx = state.get(gripper, "x")
                gy = state.get(gripper, "y")
                gz = state.get(gripper, "z")
                dx = params[0] - gx
                dy = params[1] - gy
                dz = params[2] - gz
                primitive_params = np.array([dx, dy, dz],
                                            dtype=np.float32).clip(-1.0, 1.0)
            else:
                primitive_params = params
            arr = primitive_and_params_to_primitive_action(
                name, primitive_params)
            return Action(arr)

        return policy

    @classmethod
    def _create_terminal(cls, name: str) -> ParameterizedTerminal:

        def terminal(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> bool:
            del memory, params
            if name.lower() == "move_delta_ee_pose":
                gripper, target = objects
                gx = state.get(gripper, "x")
                gy = state.get(gripper, "y")
                gz = state.get(gripper, "z")
                tx = state.get(target, "x")
                ty = state.get(target, "y")
                tz = state.get(target, "z")
                return np.allclose([gx, gy, gz], [tx, ty, tz], atol=0.09)
            return True

        return terminal
