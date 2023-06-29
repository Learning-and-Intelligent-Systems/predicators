"""Ground-truth options for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

try:
    from mujoco_kitchen.utils import \
        primitive_and_params_to_primitive_action, primitive_idx_to_name, \
        primitive_name_to_action_idx
except ImportError:
    pass
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
            if _format_name(val) == "Move_delta_ee_pose":
                params_space = Box(np.array([-5.0, -5.0, -5.0, -100.0]),
                                   np.array([5.0, 5.0, 5.0, 100.0]))
                obj_types = list(types.values())
            elif isinstance(primitive_name_to_action_idx[val], int):
                params_space = Box(-np.ones(1) * 5, np.ones(1) * 5)
                obj_types = []
            else:
                n = len(primitive_name_to_action_idx[val])
                params_space = Box(-np.ones(n) * 5, np.ones(n) * 5)
                obj_types = []
            options.add(
                utils.ParameterizedOption(
                    name=_format_name(val),
                    types=obj_types,
                    params_space=params_space,
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
                assert len(params) == 4
                if params[3] == 0.0:
                    gripper, _ = objects
                    gx = state.get(gripper, "x")
                    gy = state.get(gripper, "y")
                    gz = state.get(gripper, "z")
                    dx = params[0] - gx
                    dy = params[1] - gy
                    dz = params[2] - gz
                    primitive_params = np.array([dx, dy, dz],
                                                dtype=np.float32).clip(
                                                    -1.0, 1.0)
                else:
                    primitive_params = np.array(
                        [params[0], params[1], params[2]],
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
            if name.lower() == "move_delta_ee_pose":
                gripper, _ = objects
                if params[3] == 0.0:
                    gx = state.get(gripper, "x")
                    gy = state.get(gripper, "y")
                    gz = state.get(gripper, "z")
                    tx = params[0]
                    ty = params[1]
                    tz = params[2]
                    return np.allclose([gx, gy, gz], [tx, ty, tz], atol=0.09)
                if "steps" in memory:
                    if memory["steps"] == -1:
                        memory["steps"] = int(params[3])
                    elif memory["steps"] == 0:
                        memory["steps"] = -1
                        return True
                else:
                    memory["steps"] = int(params[3])
                memory["steps"] -= 1
                return False
            return True

        return terminal
