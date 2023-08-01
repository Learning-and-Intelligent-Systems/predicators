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
        on_off_type = types["on_off"]
        kettle_type = types["kettle"]
        surface_type = types["surface"]
        switch_type = types["switch"]
        knob_type = types["knob"]

        # Objects
        gripper = Variable("?gripper", gripper_type)
        on_off_obj = Variable("?on_off_obj", on_off_type)
        kettle = Variable("?kettle", kettle_type)
        surface = Variable("?surface", surface_type)
        switch = Variable("?switch", switch_type)
        knob = Variable("?knob", knob_type)

        # Options
        MoveToPrePushOnTop = options["MoveToPrePushOnTop"]
        MoveToPreTurnOff = options["MoveToPreTurnOff"]
        MoveToPreTurnOn = options["MoveToPreTurnOn"]
        MoveToPrePullKettle = options["MoveToPrePullKettle"]
        PullKettle = options["PullKettle"]
        PushObjOnObjForward = options["PushObjOnObjForward"]
        TurnOffSwitch = options["TurnOffSwitch"]
        TurnOnSwitch = options["TurnOnSwitch"]
        TurnOffKnob = options["TurnOffKnob"]
        TurnOnKnob = options["TurnOnKnob"]

        # Predicates
        AtPreTurnOn = predicates["AtPreTurnOn"]
        AtPreTurnOff = predicates["AtPreTurnOff"]
        AtPrePushOnTop = predicates["AtPrePushOnTop"]
        AtPrePullKettle = predicates["AtPrePullKettle"]
        TurnedOn = predicates["TurnedOn"]
        TurnedOff = predicates["TurnedOff"]
        OnTop = predicates["OnTop"]
        NotOnTop = predicates["NotOnTop"]

        nsrts = set()

        # MoveToPreTurnOff
        parameters = [gripper, on_off_obj]
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(AtPreTurnOff, [gripper, on_off_obj])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPreTurnOff
        option_vars = [gripper, on_off_obj]

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
        parameters = [gripper, on_off_obj]
        preconditions = set()
        add_effects = {LiftedAtom(AtPreTurnOn, [gripper, on_off_obj])}
        delete_effects = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPreTurnOn
        option_vars = [gripper, on_off_obj]

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
        parameters = [gripper, kettle]
        preconditions = set()
        add_effects = {LiftedAtom(AtPrePushOnTop, [gripper, kettle])}
        delete_effects = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPrePushOnTop
        option_vars = [gripper, kettle]

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

        # MoveToPrePullKettle
        parameters = [gripper, kettle]
        preconditions = set()
        add_effects = {LiftedAtom(AtPrePullKettle, [gripper, kettle])}
        delete_effects = set()
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = MoveToPrePullKettle
        option_vars = [gripper, kettle]

        def moveto_prepullkettle_sampler(state: State, goal: Set[GroundAtom],
                                         rng: np.random.Generator,
                                         objs: Sequence[Object]) -> Array:
            del state, goal  # unused
            _, obj = objs
            params = np.array(KitchenEnv.get_pre_push_delta_pos(obj, "off"),
                              dtype=np.float32)
            if not CFG.kitchen_use_perfect_samplers:
                params[0] += rng.uniform(-0.05, 0.05)
            return params

        move_to_pre_pull_kettle_nsrt = NSRT("MoveToPrePullKettle", parameters,
                                            preconditions, add_effects,
                                            delete_effects, ignore_effects,
                                            option, option_vars,
                                            moveto_prepullkettle_sampler)
        nsrts.add(move_to_pre_pull_kettle_nsrt)

        # PushObjOnObjForward
        parameters = [gripper, kettle, surface]
        preconditions = {
            LiftedAtom(AtPrePushOnTop, [gripper, kettle]),
            LiftedAtom(NotOnTop, [kettle, surface])
        }
        add_effects = {LiftedAtom(OnTop, [kettle, surface])}
        delete_effects = {LiftedAtom(NotOnTop, [kettle, surface])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = PushObjOnObjForward
        option_vars = [gripper, kettle, surface]

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

        # PullKettle
        parameters = [gripper, kettle, surface]
        preconditions = {
            LiftedAtom(AtPrePullKettle, [gripper, kettle]),
            LiftedAtom(NotOnTop, [kettle, surface])
        }
        add_effects = {LiftedAtom(OnTop, [kettle, surface])}
        delete_effects = {LiftedAtom(AtPrePullKettle, [gripper, kettle])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = PullKettle
        option_vars = [gripper, kettle, surface]

        def pull_kettle_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to pull w.r.t. the negative y axis.
            if CFG.kitchen_use_perfect_samplers:
                pull_angle = np.pi + (np.pi / 16.0)
            else:
                pull_angle = rng.uniform(7 * np.pi / 8, 9 * np.pi / 8)
            return np.array([pull_angle], dtype=np.float32)

        pull_kettle_nsrt = NSRT("PullKettle", parameters, preconditions,
                                add_effects, delete_effects, ignore_effects,
                                option, option_vars, pull_kettle_sampler)
        nsrts.add(pull_kettle_nsrt)

        # TurnOffSwitch
        parameters = [gripper, switch]
        preconditions = {
            LiftedAtom(AtPreTurnOff, [gripper, switch]),
            LiftedAtom(TurnedOn, [switch])
        }
        add_effects = {LiftedAtom(TurnedOff, [switch])}
        delete_effects = {LiftedAtom(TurnedOn, [switch])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOffSwitch
        option_vars = [gripper, switch]

        # The same sampler is used for both on and off, since the option
        # internally takes care of the direction change.
        def switch_turn_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to push w.r.t. the x axis.
            if CFG.kitchen_use_perfect_samplers:
                # Push slightly inward.
                push_angle = np.pi / 4
            else:
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        turn_off_switch_nsrt = NSRT("TurnOffSwitch", parameters, preconditions,
                                    add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    switch_turn_sampler)
        nsrts.add(turn_off_switch_nsrt)

        # TurnOnSwitch
        parameters = [gripper, switch]
        preconditions = {
            LiftedAtom(AtPreTurnOn, [gripper, switch]),
            LiftedAtom(TurnedOff, [switch])
        }
        add_effects = {LiftedAtom(TurnedOn, [switch])}
        delete_effects = {LiftedAtom(TurnedOff, [switch])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOnSwitch
        option_vars = [gripper, switch]

        turn_on_switch_nsrt = NSRT("TurnOnSwitch", parameters, preconditions,
                                   add_effects, delete_effects, ignore_effects,
                                   option, option_vars, switch_turn_sampler)
        nsrts.add(turn_on_switch_nsrt)

        # TurnOnKnob
        parameters = [gripper, knob]
        preconditions = {
            LiftedAtom(AtPreTurnOn, [gripper, knob]),
            LiftedAtom(TurnedOff, [knob])
        }
        add_effects = {LiftedAtom(TurnedOn, [knob])}
        delete_effects = {LiftedAtom(TurnedOff, [knob])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOnKnob
        option_vars = [gripper, knob]

        def knob_turn_on_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to push w.r.t. the x axis.
            if CFG.kitchen_use_perfect_samplers:
                # Push slightly inward.
                push_angle = np.pi / 8
            else:
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        turn_on_knob_nsrt = NSRT("TurnOnKnob", parameters, preconditions,
                                 add_effects, delete_effects, ignore_effects,
                                 option, option_vars, knob_turn_on_sampler)
        nsrts.add(turn_on_knob_nsrt)

        # TurnOffKnob
        parameters = [gripper, knob]
        preconditions = {
            LiftedAtom(AtPreTurnOff, [gripper, knob]),
            LiftedAtom(TurnedOn, [knob])
        }
        add_effects = {LiftedAtom(TurnedOff, [knob])}
        delete_effects = {LiftedAtom(TurnedOn, [knob])}
        ignore_effects = {
            AtPreTurnOn, AtPrePushOnTop, AtPreTurnOff, AtPrePullKettle
        }
        option = TurnOffKnob
        option_vars = [gripper, knob]

        def knob_turn_off_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Sample a direction to push w.r.t. the y-z plane.
            if CFG.kitchen_use_perfect_samplers:
                push_angle = 0.0
            else:
                push_angle = rng.uniform(-np.pi / 3, np.pi / 3)
            return np.array([push_angle], dtype=np.float32)

        turn_off_knob_nsrt = NSRT("TurnOffKnob", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, knob_turn_off_sampler)
        nsrts.add(turn_off_knob_nsrt)

        return nsrts
