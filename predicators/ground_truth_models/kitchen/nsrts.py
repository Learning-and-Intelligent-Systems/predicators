"""Ground-truth NSRTs for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
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

        # Options
        MoveTo = options["MoveTo"]
        PushObjOnObjForward = options["PushObjOnObjForward"]
        PushObjTurnOnRight = options["PushObjTurnOnRight"]

        # Predicates
        At = predicates["At"]
        TurnedOn = predicates["TurnedOn"]
        OnTop = predicates["OnTop"]

        nsrts = set()

        # MoveTo
        parameters = [gripper, obj]
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(At, [gripper, obj])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {At}
        option = MoveTo
        option_vars = [gripper, obj]

        def moveto_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del goal  # unused
            gripper, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj),
                              dtype=np.float32)
            # NOTE: this is a legitimately hard function to hand-write. I could
            # not figure out how to do it in a state-independent way.
            if CFG.kitchen_use_perfect_samplers:
                if state.get(gripper, "x") > -0.15:
                    params[2] += 0.1
            else:
                params[0] += rng.uniform(-0.5, 0.5)
            return params

        move_to_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                            delete_effects, ignore_effects, option,
                            option_vars, moveto_sampler)
        nsrts.add(move_to_nsrt)

        # PushObjOnObjForward
        parameters = [gripper, obj, obj2]
        preconditions = {LiftedAtom(At, [gripper, obj])}
        add_effects = {LiftedAtom(OnTop, [obj, obj2])}
        delete_effects = set()
        ignore_effects = set()
        option = PushObjOnObjForward
        option_vars = [gripper, obj, obj2]

        def push_obj_on_obj_forward_sampler(state: State,
                                            goal: Set[GroundAtom],
                                            rng: np.random.Generator,
                                            objs: Sequence[Object]) -> Array:
            del state, goal, objs, rng  # unused
            dy = 0.1
            return np.array([dy], dtype=np.float32)

        push_obj_on_obj_forward_nsrt = NSRT("PushObjOnObjForward", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            push_obj_on_obj_forward_sampler)
        nsrts.add(push_obj_on_obj_forward_nsrt)

        # PushObjTurnOnRight
        parameters = [gripper, obj]
        preconditions = {LiftedAtom(At, [gripper, obj])}
        add_effects = {LiftedAtom(TurnedOn, [obj])}
        delete_effects = set()
        ignore_effects = set()
        option = PushObjTurnOnRight
        option_vars = [gripper, obj]

        def push_obj_turn_on_right_sampler(state: State, goal: Set[GroundAtom],
                                           rng: np.random.Generator,
                                           objs: Sequence[Object]) -> Array:
            del state, goal, objs, rng  # unused
            dx = 0.1
            return np.array([dx], dtype=np.float32)

        push_obj_turn_on_right_nsrt = NSRT("PushObjTurnOnRight", parameters,
                                           preconditions, add_effects,
                                           delete_effects, ignore_effects,
                                           option, option_vars,
                                           push_obj_turn_on_right_sampler)
        nsrts.add(push_obj_turn_on_right_nsrt)

        return nsrts
