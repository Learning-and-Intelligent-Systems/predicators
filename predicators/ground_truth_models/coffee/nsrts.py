"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class CoffeeGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the coffee environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"coffee", "pybullet_coffee"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        cup_type = types["cup"]
        machine_type = types["machine"]

        # Predicates
        CupFilled = predicates["CupFilled"]
        Holding = predicates["Holding"]
        JugInMachine = predicates["JugInMachine"]
        MachineOn = predicates["MachineOn"]
        OnTable = predicates["OnTable"]
        HandEmpty = predicates["HandEmpty"]
        JugFilled = predicates["JugFilled"]
        RobotAboveCup = predicates["RobotAboveCup"]
        JugAboveCup = predicates["JugAboveCup"]
        NotAboveCup = predicates["NotAboveCup"]
        PressingButton = predicates["PressingButton"]
        Twisting = predicates["Twisting"]
        NotSameCup = predicates["NotSameCup"]
        JugPickable = predicates["JugPickable"]

        # Options
        if CFG.coffee_combined_move_and_twist_policy:
            Twist = options["Twist"]
        else:
            MoveToTwistJug = options["MoveToTwistJug"]
            TwistJug = options["TwistJug"]
        PickJug = options["PickJug"]
        PlaceJugInMachine = options["PlaceJugInMachine"]
        TurnMachineOn = options["TurnMachineOn"]
        Pour = options["Pour"]


        nsrts = set()

        if not CFG.coffee_combined_move_and_twist_policy:
            # MoveToTwistJug
            robot = Variable("?robot", robot_type)
            jug = Variable("?jug", jug_type)
            parameters = [robot, jug]
            option_vars = [robot, jug]
            option = MoveToTwistJug
            preconditions = {
                LiftedAtom(OnTable, [jug]),
                LiftedAtom(HandEmpty, [robot]),
            }
            add_effects = {
                LiftedAtom(Twisting, [robot, jug]),
            }
            delete_effects = {
                LiftedAtom(HandEmpty, [robot]),
            }
            ignore_effects: Set[Predicate] = set()
            move_to_twist_jug_nsrt = NSRT("MoveToTwistJug", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, null_sampler)
            nsrts.add(move_to_twist_jug_nsrt)

            # TwistJug
            robot = Variable("?robot", robot_type)
            jug = Variable("?jug", jug_type)
            parameters = [robot, jug]
            option_vars = [robot, jug]
            option = TwistJug
            preconditions = {
                LiftedAtom(OnTable, [jug]),
                LiftedAtom(Twisting, [robot, jug]),
            }
            add_effects = {
                LiftedAtom(HandEmpty, [robot]),
            }
            if CFG.coffee_jug_pickable_pred:
                add_effects.add(LiftedAtom(JugPickable, [jug]))
            delete_effects = {
                LiftedAtom(Twisting, [robot, jug]),
            }
            ignore_effects = set()

            def twist_jug_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
                del state, goal, objs  # unused
                return np.array(rng.uniform(-1, 1, size=(1, )), dtype=np.float32)

            twist_jug_nsrt = NSRT("TwistJug", parameters, preconditions,
                                add_effects, delete_effects, ignore_effects,
                                option, option_vars,
                                twist_jug_sampler if CFG.coffee_twist_sampler \
                                    else null_sampler)
            nsrts.add(twist_jug_nsrt)
        else:
            # Twist
            robot = Variable("?robot", robot_type)
            jug = Variable("?jug", jug_type)
            parameters = [robot, jug]
            option_vars = [robot, jug]
            option = Twist
            preconditions = {
                LiftedAtom(OnTable, [jug]),
                LiftedAtom(HandEmpty, [robot]),
            }
            add_effects = set()
            if CFG.coffee_jug_pickable_pred:
                add_effects.add(LiftedAtom(JugPickable, [jug]))
            delete_effects = set()
            ignore_effects: Set[Predicate] = set()
            twist_nsrt = NSRT("Twist", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, null_sampler)
            nsrts.add(twist_nsrt)


        # PickJugFromTable
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        preconditions = {
            LiftedAtom(OnTable, [jug]),
            LiftedAtom(HandEmpty, [robot]),
        }
        if CFG.coffee_jug_pickable_pred:
            preconditions.add(LiftedAtom(JugPickable, [jug]))
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(OnTable, [jug]),
            LiftedAtom(HandEmpty, [robot])
        }
        ignore_effects = set()
        pick_jug_from_table_nsrt = NSRT("PickJugFromTable", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, null_sampler)
        nsrts.add(pick_jug_from_table_nsrt)

        # PlaceJugInMachine
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        machine = Variable("?machine", machine_type)
        parameters = [robot, jug, machine]
        option_vars = [robot, jug, machine]
        option = PlaceJugInMachine
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugInMachine, [jug, machine]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        ignore_effects = set()
        place_jug_in_machine_nsrt = NSRT("PlaceJugInMachine", parameters,
                                         preconditions, add_effects,
                                         delete_effects, ignore_effects,
                                         option, option_vars, null_sampler)
        nsrts.add(place_jug_in_machine_nsrt)

        # TurnMachineOn
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        machine = Variable("?machine", machine_type)
        parameters = [robot, jug, machine]
        option_vars = [robot, machine]
        option = TurnMachineOn
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugInMachine, [jug, machine]),
        }
        add_effects = {
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(MachineOn, [machine]),
            LiftedAtom(PressingButton, [robot, machine]),
        }
        delete_effects = set()
        ignore_effects = set()
        turn_machine_on_nsrt = NSRT("TurnMachineOn", parameters, preconditions,
                                    add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
        nsrts.add(turn_machine_on_nsrt)

        # PickJugFromMachine
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        machine = Variable("?machine", machine_type)
        parameters = [robot, jug, machine]
        option_vars = [robot, jug]
        option = PickJug
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugInMachine, [jug, machine]),
            LiftedAtom(PressingButton, [robot, machine]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugInMachine, [jug, machine]),
            LiftedAtom(PressingButton, [robot, machine]),
        }
        ignore_effects = set()
        pick_jug_from_machine_nsrt = NSRT("PickJugFromMachine", parameters,
                                          preconditions, add_effects,
                                          delete_effects, ignore_effects,
                                          option, option_vars, null_sampler)
        nsrts.add(pick_jug_from_machine_nsrt)

        # PourFromNowhere
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        parameters = [robot, jug, cup]
        option_vars = [robot, jug, cup]
        option = Pour
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(NotAboveCup, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(JugAboveCup, [jug, cup]),
            LiftedAtom(RobotAboveCup, [robot, cup]),
            LiftedAtom(CupFilled, [cup]),
        }
        delete_effects = {
            LiftedAtom(NotAboveCup, [robot, jug]),
        }
        ignore_effects = set()
        pour_from_nowhere_nsrt = NSRT("PourFromNowhere", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects, option,
                                      option_vars, null_sampler)
        nsrts.add(pour_from_nowhere_nsrt)

        # PourFromOtherCup
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        cup = Variable("?cup", cup_type)
        other_cup = Variable("?other_cup", cup_type)
        parameters = [robot, jug, cup, other_cup]
        option_vars = [robot, jug, cup]
        option = Pour
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(JugAboveCup, [jug, other_cup]),
            LiftedAtom(RobotAboveCup, [robot, other_cup]),
            LiftedAtom(NotSameCup, [cup, other_cup]),
        }
        add_effects = {
            LiftedAtom(JugAboveCup, [jug, cup]),
            LiftedAtom(RobotAboveCup, [robot, cup]),
            LiftedAtom(CupFilled, [cup]),
        }
        delete_effects = {
            LiftedAtom(JugAboveCup, [jug, other_cup]),
            LiftedAtom(RobotAboveCup, [robot, other_cup]),
        }
        ignore_effects = set()
        pour_from_other_cup_nsrt = NSRT("PourFromOtherCup", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, null_sampler)
        nsrts.add(pour_from_other_cup_nsrt)

        return nsrts
