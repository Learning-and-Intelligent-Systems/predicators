"""Ground-truth NSRTs for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
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
        # Types
        gripper_type = types["gripper"]
        object_type = types["obj"]

        # Objects
        gripper = Variable("?gripper", gripper_type)
        obj = Variable("?obj", object_type)
        obj2 = Variable("?obj2", object_type)

        # Predicates
        At = predicates["At"]
        On = predicates["On"]
        OnTop = predicates["OnTop"]

        # Options
        MoveTo = options["Move_delta_ee_pose"]

        nsrts = set()

        # Samplers
        def moveto_sampler(state: State, goal: Set[GroundAtom],
                           _rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del goal, _rng
            _, obj = objs
            ox = state.get(obj, "x")
            oy = state.get(obj, "y")
            oz = state.get(obj, "z")
            if obj.name == 'knob2':
                return np.array([ox - 0.05, oy + 0.05, oz - 0.05, 0.0],
                                dtype=np.float32)
            if obj.name == 'knob3':
                return np.array([ox - 0.1, oy + 0.02, oz - 0.1, 0.0],
                                dtype=np.float32)
            if obj.name == 'kettle':
                return np.array([ox + 0.15, oy - 0.4, oz - 0.15, 0.0],
                                dtype=np.float32)
            return np.array([ox, oy, oz, 0.0], dtype=np.float32)

        def push_sampler(state: State, goal: Set[GroundAtom],
                         _rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
            del state, goal, _rng
            if len(objs) == 2:
                _, obj = objs
            else:
                assert len(objs) == 3
                _, obj, _ = objs
            if obj.name == 'knob2':
                return np.array([0.15, 0.0, 0.0, 5.0], dtype=np.float32)
            if obj.name == 'knob3':
                return np.array([0.1, 0.0, 0.0, 5.0], dtype=np.float32)
            if obj.name == 'kettle':
                return np.array([0.0, 0.1, 0.0, 50.0], dtype=np.float32)
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        # MoveTo
        parameters = [gripper, obj]
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(At, [gripper, obj])}
        delete_effects: Set[LiftedAtom] = set()
        option = MoveTo
        option_vars = [gripper, obj]
        move_to_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                            delete_effects, {At}, option, option_vars,
                            moveto_sampler)
        nsrts.add(move_to_nsrt)

        # PushObjOnObjForward
        parameters = [gripper, obj, obj2]
        preconditions = {LiftedAtom(At, [gripper, obj])}
        add_effects = {LiftedAtom(OnTop, [obj, obj2])}
        delete_effects = set()
        option = MoveTo
        option_vars = [gripper, obj]
        push_obj_on_obj_forward_nsrt = NSRT("PushObjOnObjForward", parameters,
                                            preconditions, add_effects,
                                            delete_effects, {OnTop}, option,
                                            option_vars, push_sampler)
        nsrts.add(push_obj_on_obj_forward_nsrt)

        # PushObjTurnOnRight
        parameters = [gripper, obj]
        preconditions = {LiftedAtom(At, [gripper, obj])}
        add_effects = {LiftedAtom(On, [obj])}
        delete_effects = set()
        option = MoveTo
        option_vars = [gripper, obj]
        push_obj_turn_on_right_nsrt = NSRT("PushObjTurnOnRight", parameters,
                                           preconditions, add_effects,
                                           delete_effects, {On}, option,
                                           option_vars, push_sampler)
        nsrts.add(push_obj_turn_on_right_nsrt)

        return nsrts
