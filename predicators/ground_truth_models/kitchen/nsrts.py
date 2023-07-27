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
        MoveToPrePushOnTop = options["MoveToPrePushOnTop"]
        MoveToPreTurnOff = options["MoveToPreTurnOff"]
        MoveToPreTurnOn = options["MoveToPreTurnOn"]
        PushObjOnObjForward = options["PushObjOnObjForward"]
        PushObjTurnOffLeftRight = options["PushObjTurnOffLeftRight"]
        PushObjTurnOnLeftRight = options["PushObjTurnOnLeftRight"]

        # Predicates
        AtPreTurnOn = predicates["AtPreTurnOn"]
        AtPreTurnOff = predicates["AtPreTurnOff"]
        AtPrePushOnTop = predicates["AtPrePushOnTop"]
        TurnedOn = predicates["TurnedOn"]
        TurnedOff = predicates["TurnedOff"]
        OnTop = predicates["OnTop"]
        NotOnTop = predicates["NotOnTop"]

        nsrts = set()

        # MoveToPreTurnOff
        parameters = [gripper, obj]
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(AtPreTurnOff, [gripper, obj])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff}
        option = MoveToPreTurnOff
        option_vars = [gripper, obj]

        def moveto_preturnoff_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj, "off"),
                              dtype=np.float32)
            return params

        move_to_pre_turn_off_nsrt = NSRT("MoveToPreTurnOff", parameters,
                                         preconditions, add_effects,
                                         delete_effects, ignore_effects,
                                         option, option_vars,
                                         moveto_preturnoff_sampler)
        nsrts.add(move_to_pre_turn_off_nsrt)

        # MoveToPreTurnOn
        parameters = [gripper, obj]
        preconditions = set()
        add_effects = {LiftedAtom(AtPreTurnOn, [gripper, obj])}
        delete_effects = set()
        ignore_effects = {AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff}
        option = MoveToPreTurnOn
        option_vars = [gripper, obj]

        def moveto_preturnon_sampler(state: State, goal: Set[GroundAtom],
                                     rng: np.random.Generator,
                                     objs: Sequence[Object]) -> Array:
            del state, goal, rng  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj, "on"),
                              dtype=np.float32)
            return params

        move_to_pre_turn_on_nsrt = NSRT("MoveToPreTurnOn", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, moveto_preturnon_sampler)
        nsrts.add(move_to_pre_turn_on_nsrt)

        # MoveToPrePushOnTop
        parameters = [gripper, obj]
        preconditions = set()
        add_effects = {LiftedAtom(AtPrePushOnTop, [gripper, obj])}
        delete_effects = set()
        ignore_effects = {AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff}
        option = MoveToPrePushOnTop
        option_vars = [gripper, obj]

        def moveto_prepushontop_sampler(state: State, goal: Set[GroundAtom],
                                        rng: np.random.Generator,
                                        objs: Sequence[Object]) -> Array:
            del state, goal  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj, "on"),
                              dtype=np.float32)
            if not CFG.kitchen_use_perfect_samplers:
                # Truncated on the right to avoid robot self collisions.
                params[0] += rng.uniform(-0.25, 0.05)
            return params

        move_to_pre_push_on_top_nsrt = NSRT("MoveToPrePushOnTop", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            moveto_prepushontop_sampler)
        nsrts.add(move_to_pre_push_on_top_nsrt)

        # PushObjOnObjForward
        parameters = [gripper, obj, obj2]
        preconditions = {
            LiftedAtom(AtPrePushOnTop, [gripper, obj]),
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
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        push_obj_on_obj_forward_nsrt = NSRT("PushObjOnObjForward", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            push_obj_on_obj_forward_sampler)
        nsrts.add(push_obj_on_obj_forward_nsrt)

        # PushObjTurnOffLeftRight
        parameters = [gripper, obj]
        preconditions = {
            LiftedAtom(AtPreTurnOff, [gripper, obj]),
            LiftedAtom(TurnedOn, [obj])
        }
        add_effects = {LiftedAtom(TurnedOff, [obj])}
        delete_effects = {LiftedAtom(TurnedOn, [obj])}
        ignore_effects = set()
        option = PushObjTurnOffLeftRight
        option_vars = [gripper, obj]

        # The same sampler is used for both on and off, since the option
        # internally takes care of the direction change.
        def push_obj_turn_sampler(state: State, goal: Set[GroundAtom],
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

        push_obj_turn_on_nsrt = NSRT("PushObjTurnOffLeftRight", parameters,
                                     preconditions, add_effects,
                                     delete_effects, ignore_effects, option,
                                     option_vars, push_obj_turn_sampler)
        nsrts.add(push_obj_turn_on_nsrt)

        # PushObjTurnOnLeftRight
        parameters = [gripper, obj]
        preconditions = {
            LiftedAtom(AtPreTurnOn, [gripper, obj]),
            LiftedAtom(TurnedOff, [obj])
        }
        add_effects = {LiftedAtom(TurnedOn, [obj])}
        delete_effects = {LiftedAtom(TurnedOff, [obj])}
        ignore_effects = set()
        option = PushObjTurnOnLeftRight
        option_vars = [gripper, obj]

        push_obj_turn_on_nsrt = NSRT("PushObjTurnOnLeftRight", parameters,
                                     preconditions, add_effects,
                                     delete_effects, ignore_effects, option,
                                     option_vars, push_obj_turn_sampler)
        nsrts.add(push_obj_turn_on_nsrt)

        return nsrts
