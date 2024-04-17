
from typing import Sequence
import numpy as np
from predicators.structs import State, Object, Predicate, Type
from predicators.envs.stick_button import StickButtonEnv
        
_hand_type = StickButtonEnv._hand_type
_button_type = StickButtonEnv._button_type
_stick_type = StickButtonEnv._stick_type
_holder_type = StickButtonEnv._holder_type

def _InReach_holds(state: State, objects: Sequence[Object]) -> bool:
    hand, obj = objects
    hand_position = state.get(hand, 'x'), state.get(hand, 'y')
    obj_position = state.get(obj, 'x'), state.get(obj, 'y')
    max_reach_distance = 5.0  # Assume some reasonable maximum reach distance
    return bool(np.linalg.norm(np.array(hand_position) - np.array(obj_position)) <= max_reach_distance)

InReach = Predicate("InReach", [_hand_type, _button_type], _InReach_holds)