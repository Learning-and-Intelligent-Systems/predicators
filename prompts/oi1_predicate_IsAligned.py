from typing import Sequence
import numpy as np
from predicators.structs import State, Object, Predicate, Type
from predicators.envs.stick_button import StickButtonEnv
        
_hand_type = StickButtonEnv._hand_type
_button_type = StickButtonEnv._button_type
_stick_type = StickButtonEnv._stick_type
_holder_type = StickButtonEnv._holder_type

def _IsAligned_holds(state: State, objects: Sequence[Object]) -> bool:
    hand, obj = objects
    hand_y = state.get(hand, 'y')
    obj_y = state.get(obj, 'y')
    return abs(hand_y - obj_y) < 0.1  # Small threshold to consider as aligned

IsAligned = Predicate("IsAligned", [_hand_type, Type], _IsAligned_holds)