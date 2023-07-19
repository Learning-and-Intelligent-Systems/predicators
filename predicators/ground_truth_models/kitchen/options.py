"""Ground-truth options for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

try:
    from mujoco_kitchen.utils import primitive_and_params_to_primitive_action
    _MJKITCHEN_IMPORTED = True
except ImportError:
    _MJKITCHEN_IMPORTED = False
from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Type


class KitchenGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the Kitchen environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        assert _MJKITCHEN_IMPORTED

        # Types
        gripper_type = types["gripper"]
        object_type = types["obj"]

        # Predicates
        TurnedOn = predicates["TurnedOn"]
        OnTop = predicates["OnTop"]

        options: Set[ParameterizedOption] = set()

        # MoveTo
        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
            del memory  # unused
            # Use move_delta_ee_pose directly.
            max_delta_mag = 0.1  # don't move more than this per step
            gripper = objects[0]
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            delta_ee = params - (gx, gy, gz)
            delta_ee = np.clip(delta_ee, -max_delta_mag, max_delta_mag)
            arr = primitive_and_params_to_primitive_action(
                "move_delta_ee_pose", delta_ee)
            return Action(arr)

        def _MoveTo_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del memory  # unused
            gripper = objects[0]
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            return np.allclose((gx, gy, gz), params, atol=KitchenEnv.at_atol)

        MoveTo = ParameterizedOption(
            "MoveTo",
            types=[gripper_type, object_type],
            # Parameter is an absolute position to move to.
            params_space=Box(-5, 5, (3, )),
            policy=_MoveTo_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_MoveTo_terminal)

        options.add(MoveTo)

        # PushObjOnObjForward
        def _PushObjOnObjForward_policy(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> Action:
            # Currently this is identical to the MoveTo policy, but this is
            # expected to change soon.
            return _MoveTo_policy(state, memory, objects, params)

        def _PushObjOnObjForward_terminal(state: State, memory: Dict,
                                          objects: Sequence[Object],
                                          params: Array) -> bool:
            del memory, params  # unused
            _, obj, obj2 = objects
            return GroundAtom(OnTop, [obj, obj2]).holds(state)

        PushObjOnObjForward = ParameterizedOption(
            "PushObjOnObjForward",
            types=[gripper_type, object_type, object_type],
            # Parameter is an absolute position to move to.
            params_space=Box(-5, 5, (3, )),
            policy=_PushObjOnObjForward_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PushObjOnObjForward_terminal)

        options.add(PushObjOnObjForward)

        # PushObjTurnOnRight
        def _PushObjTurnOnRight_policy(state: State, memory: Dict,
                                       objects: Sequence[Object],
                                       params: Array) -> Action:
            # Currently this is identical to the MoveTo policy, but this is
            # expected to change soon.
            return _MoveTo_policy(state, memory, objects, params)

        def _PushObjTurnOnRight_terminal(state: State, memory: Dict,
                                         objects: Sequence[Object],
                                         params: Array) -> bool:
            del memory, params  # unused
            _, obj = objects
            return GroundAtom(TurnedOn, [obj]).holds(state)

        PushObjTurnOnRight = ParameterizedOption(
            "PushObjTurnOnRight",
            types=[gripper_type, object_type],
            # Parameter is an absolute position to move to.
            params_space=Box(-5, 5, (3, )),
            policy=_PushObjTurnOnRight_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PushObjTurnOnRight_terminal)

        options.add(PushObjTurnOnRight)

        return options
