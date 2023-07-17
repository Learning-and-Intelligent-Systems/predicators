"""Ground-truth NSRTs for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.ground_truth_models.kitchen.operators import \
    KitchenGroundTruthOperatorFactory
from predicators.structs import NSRT, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class KitchenGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Kitchen environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Operators
        operators = KitchenGroundTruthOperatorFactory.get_operators(
            env_name, types, predicates)
        op_name_to_op = {op.name: op for op in operators}

        # Types
        gripper_type = types["gripper"]
        object_type = types["obj"]

        # Objects
        gripper = Variable("?gripper", gripper_type)
        obj = Variable("?obj", object_type)
        obj2 = Variable("?obj2", object_type)

        nsrts = set()

        # Samplers
        def moveto_sampler(state: State, goal: Set[GroundAtom],
                           _rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del goal
            _, obj = objs
            ox = state.get(obj, "x")
            oy = state.get(obj, "y")
            oz = state.get(obj, "z")

            if obj.name == 'knob3':
                return np.array([ox - 0.2, oy, oz - 0.2], dtype=np.float32)
            if obj.name == 'kettle':
                return np.array([ox + 0.1, oy - 0.4, oz - 0.2],
                                dtype=np.float32)
            return np.array([ox, oy, oz], dtype=np.float32)

        def push_sampler(state: State, goal: Set[GroundAtom],
                         _rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
            del goal
            if len(objs) == 2:
                gripper, obj = objs
            else:
                assert len(objs) == 3
                gripper, obj, _ = objs
            x = state.get(gripper, "x")
            y = state.get(gripper, "y")
            z = state.get(gripper, "z")
            if obj.name == 'knob3':
                return np.array([x + 1.0, y, z], dtype=np.float32)
            if obj.name == 'kettle':
                rand_dx = _rng.uniform(0.0, 1.0)
                return np.array([x + rand_dx, y + 5.0, z - 0.3],
                                dtype=np.float32)
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # MoveTo
        op_name = "MoveTo"
        op = op_name_to_op[op_name]
        option = options[op_name.lower() + "_option"]
        option_vars = [gripper, obj]
        move_to_nsrt = NSRT(op_name, op.parameters, op.preconditions,
                            op.add_effects, op.delete_effects,
                            op.ignore_effects, option, option_vars,
                            moveto_sampler)
        nsrts.add(move_to_nsrt)

        # PushObjOnObjForward
        op_name = "PushObjOnObjForward"
        op = op_name_to_op[op_name]
        option = options[op_name.lower() + "_option"]
        option_vars = [gripper, obj, obj2]
        push_obj_on_obj_forward_nsrt = NSRT(op_name, op.parameters,
                                            op.preconditions, op.add_effects,
                                            op.delete_effects,
                                            op.ignore_effects, option,
                                            option_vars, push_sampler)
        nsrts.add(push_obj_on_obj_forward_nsrt)

        # PushObjTurnOnRight
        op_name = "PushObjTurnOnRight"
        op = op_name_to_op[op_name]
        option = options[op_name.lower() + "_option"]
        option_vars = [gripper, obj]
        push_obj_turn_on_right_nsrt = NSRT(op_name, op.parameters,
                                           op.preconditions, op.add_effects,
                                           op.delete_effects,
                                           op.ignore_effects, option,
                                           option_vars, push_sampler)
        nsrts.add(push_obj_turn_on_right_nsrt)

        return nsrts
