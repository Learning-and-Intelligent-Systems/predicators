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

        max_delta_mag = 0.1  # don't move more than this per step

        # MoveTo
        def _MoveTo_initiable(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            # Store the target pose.
            _, obj = objects
            ox = state.get(obj, "x")
            oy = state.get(obj, "y")
            oz = state.get(obj, "z")
            target_pose = params + (ox, oy, oz)
            memory["target_pose"] = target_pose
            # We always move backward for 8 steps before executing the move.
            # Ideally we would move to a home position, but it's not possible
            # to do that reliably with end effector control. Moving for 8 steps
            # seems to work well enough for now, but it's likely that this will
            # need to be improved in the future as we add more skills,
            # especially considering that 5 and 10 backward steps don't work,
            # indicating that this is very fickle.
            memory["reset_count"] = 8
            return True

        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
            del params  # unused
            gripper = objects[0]
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            if memory["reset_count"] > 0:
                arr = primitive_and_params_to_primitive_action(
                    "move_backward", [1.0])
                memory["reset_count"] -= 1
            else:
                target_pose = memory["target_pose"]
                delta_ee = target_pose - (gx, gy, gz)
                delta_ee = np.clip(delta_ee, -max_delta_mag, max_delta_mag)
                arr = primitive_and_params_to_primitive_action(
                    "move_delta_ee_pose", delta_ee)
            return Action(arr)

        def _MoveTo_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del params  # unused
            if memory["reset_count"] > 0:
                return False
            gripper = objects[0]
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            return np.allclose((gx, gy, gz),
                               memory["target_pose"],
                               atol=KitchenEnv.at_atol)

        MoveTo = ParameterizedOption(
            "MoveTo",
            types=[gripper_type, object_type],
            # Parameter is a position to move to relative to the object.
            params_space=Box(-5, 5, (3, )),
            policy=_MoveTo_policy,
            initiable=_MoveTo_initiable,
            terminal=_MoveTo_terminal)

        options.add(MoveTo)

        # PushObjOnObjForward
        def _PushObjOnObjForward_policy(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> Action:
            del state, memory, objects  # unused
            arr = primitive_and_params_to_primitive_action(
                "move_forward", params)
            return Action(arr)

        def _PushObjOnObjForward_terminal(state: State, memory: Dict,
                                          objects: Sequence[Object],
                                          params: Array) -> bool:
            del memory, params  # unused
            _, obj, obj2 = objects
            if not GroundAtom(OnTop, [obj, obj2]).holds(state):
                return False
            # Stronger check to deal with case where push release leads object
            # to be no longer OnTop.
            return state.get(
                obj, "y") > state.get(obj2, "y") - KitchenEnv.at_atol / 2

        PushObjOnObjForward = ParameterizedOption(
            "PushObjOnObjForward",
            types=[gripper_type, object_type, object_type],
            # Parameter is a magnitude for pushing forward.
            params_space=Box(0.0, max_delta_mag, (1, )),
            policy=_PushObjOnObjForward_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PushObjOnObjForward_terminal)

        options.add(PushObjOnObjForward)

        # PushObjTurnOnRight
        def _PushObjTurnOnRight_policy(state: State, memory: Dict,
                                       objects: Sequence[Object],
                                       params: Array) -> Action:
            del state, memory, objects  # unused
            arr = primitive_and_params_to_primitive_action(
                "move_right", params)
            return Action(arr)

        def _PushObjTurnOnRight_terminal(state: State, memory: Dict,
                                         objects: Sequence[Object],
                                         params: Array) -> bool:
            del memory, params  # unused
            _, obj = objects
            return GroundAtom(TurnedOn, [obj]).holds(state)

        PushObjTurnOnRight = ParameterizedOption(
            "PushObjTurnOnRight",
            types=[gripper_type, object_type],
            # Parameter is a magnitude for pushing right.
            params_space=Box(0.0, max_delta_mag, (1, )),
            policy=_PushObjTurnOnRight_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PushObjTurnOnRight_terminal)

        options.add(PushObjTurnOnRight)

        return options
