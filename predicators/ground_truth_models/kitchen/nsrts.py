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
        PushObjTurnOnLeftRight = options["PushObjTurnOnLeftRight"]

        # Predicates
        At = predicates["At"]
        TurnedOn = predicates["TurnedOn"]
        TurnedOff = predicates["TurnedOff"]
        OnTop = predicates["OnTop"]
        NotOnTop = predicates["NotOnTop"]

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
            del state, goal  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj),
                              dtype=np.float32)
            if not CFG.kitchen_use_perfect_samplers:
                # Others coming soon...
                if obj.name == "kettle":
                    # Truncated on the right to avoid robot self collisions.
                    params[0] += rng.uniform(-0.25, 0.05)
            return params

        move_to_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                            delete_effects, ignore_effects, option,
                            option_vars, moveto_sampler)
        nsrts.add(move_to_nsrt)

        # PushObjOnObjForward
        parameters = [gripper, obj, obj2]
        preconditions = {
            LiftedAtom(At, [gripper, obj]),
            LiftedAtom(NotOnTop, [obj, obj2])
        }
        add_effects = {LiftedAtom(OnTop, [obj, obj2])}
        delete_effects = {LiftedAtom(NotOnTop, [obj, obj2])}
        ignore_effects = set()
        option = PushObjOnObjForward
        option_vars = [gripper, obj, obj2]

        def push_obj_on_obj_forward_sampler(state: State,
                                            goal: Set[GroundAtom],
                                            rng: np.random.Generator,
                                            objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to push w.r.t. the y axis.
            if CFG.kitchen_use_perfect_samplers:
                push_angle = 0.0
            else:
                push_angle = rng.uniform(-np.pi / 4, np.pi / 4)
            return np.array([push_angle], dtype=np.float32)

        push_obj_on_obj_forward_nsrt = NSRT("PushObjOnObjForward", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            push_obj_on_obj_forward_sampler)
        nsrts.add(push_obj_on_obj_forward_nsrt)

        # PushObjTurnOnLeftRight
        parameters = [gripper, obj]
        preconditions = {
            LiftedAtom(At, [gripper, obj]),
            LiftedAtom(TurnedOff, [obj])
        }
        add_effects = {LiftedAtom(TurnedOn, [obj])}
        delete_effects = {LiftedAtom(TurnedOff, [obj])}
        ignore_effects = set()
        option = PushObjTurnOnLeftRight
        option_vars = [gripper, obj]

        def push_obj_turn_on_sampler(state: State, goal: Set[GroundAtom],
                                     rng: np.random.Generator,
                                     objs: Sequence[Object]) -> Array:
            del state, goal  # unused
            _, obj = objs
            # Sample a direction to push w.r.t. the x axis.
            if CFG.kitchen_use_perfect_samplers:
                # Push slightly inward.
                if "knob" in obj.name:
                    push_angle = np.pi / 8
                else:
                    push_angle = np.pi / 4
            else:
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        push_obj_turn_on_nsrt = NSRT("PushObjTurnOnLeftRight", parameters,
                                     preconditions, add_effects,
                                     delete_effects, ignore_effects, option,
                                     option_vars, push_obj_turn_on_sampler)
        nsrts.add(push_obj_turn_on_nsrt)

        return nsrts
