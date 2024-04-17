from typing import Sequence
import numpy as np
from predicators.structs import State, Object, Predicate, Type
from predicators.envs.stick_button import StickButtonEnv
        
_hand_type = StickButtonEnv._hand_type
_button_type = StickButtonEnv._button_type
_stick_type = StickButtonEnv._stick_type
_holder_type = StickButtonEnv._holder_type

def _IsAbove_holds(state: State, objects: Sequence[Object]) -> bool:
    above_obj, below_obj = objects
    above_x, above_y = state.get(above_obj, 'x'), state.get(above_obj, 'y')
    below_x, below_y = state.get(below_obj, 'x'), state.get(below_obj, 'y')
    return abs(above_x - below_x) < 0.1 and above_y > below_y

IsAbove = Predicate("IsAbove", [Type, Type], _IsAbove_holds)