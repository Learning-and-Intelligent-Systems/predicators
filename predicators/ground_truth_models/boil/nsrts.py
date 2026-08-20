"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class PyBulletBoilGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the boil environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_boil"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        burner_type = types["burner"]
        faucet_type = types["faucet"]
        _ = types["human"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        JugAtBurner = predicates["JugAtBurner"]
        JugAtFaucet = predicates["JugAtFaucet"]
        NoJugAtFaucet = predicates["NoJugAtFaucet"]
        NoJugAtBurner = predicates["NoJugAtBurner"]
        JugNotAtBurnerOrFaucet = predicates["JugNotAtBurnerOrFaucet"]
        if CFG.boil_add_jug_reached_capacity_predicate:
            _NoJugAtFaucetOrAtFaucetAndReachedCapacity = predicates[
                "NoJugAtFaucetOrAtFaucetAndReachedCapacity"]
            _ = predicates["JugAtCapacity"]
        else:
            _NoJugAtFaucetOrJugAtFaucetAndFilled = predicates[
                "NoJugAtFaucetOrAtFaucetAndFilled"]
        JugFilled = predicates["JugFilled"]
        # JugNotFilled = predicates["JugNotFilled"]
        # WaterSpilled = predicates["WaterSpilled"]
        NoWaterSpilled = predicates["NoWaterSpilled"]
        FaucetOn = predicates["FaucetOn"]
        FaucetOff = predicates["FaucetOff"]
        BurnerOn = predicates["BurnerOn"]
        BurnerOff = predicates["BurnerOff"]
        WaterBoiled = predicates["WaterBoiled"]
        if CFG.boil_goal == "human_happy":
            _ = predicates["HumanHappy"]
        elif CFG.boil_goal == "task_completed":
            TaskCompleted = predicates["TaskCompleted"]

        # Options
        PickJug = options["PickJug"]
        PlaceOnBurner = options["PlaceOnBurner"]
        PlaceUnderFaucet = options["PlaceUnderFaucet"]
        PlaceOutsideBurnerAndFaucet = options["PlaceOutsideBurnerAndFaucet"]
        # Having swtich for each because of the type
        SwitchFaucetOn = options["SwitchFaucetOn"]
        SwitchFaucetOff = options["SwitchFaucetOff"]
        SwitchBurnerOn = options["SwitchBurnerOn"]
        SwitchBurnerOff = options["SwitchBurnerOff"]
        Wait = options["Wait"]
        if CFG.boil_goal == "task_completed":
            DeclareComplete = options["DeclareComplete"]

        nsrts = set()

        # PickJug
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        pick_jug_from_table_nsrt = NSRT("PickJugFromTable", parameters,
                                        preconditions, add_effects,
                                        delete_effects, set(), option,
                                        option_vars, null_sampler)
        nsrts.add(pick_jug_from_table_nsrt)

        # Place
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, jug, burner]
        option_vars = [robot, burner]
        option = PlaceOnBurner
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(JugAtBurner, [jug, burner]),
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }

        place_on_burner_nsrt = NSRT("PlaceOnBurner", parameters,
                                    preconditions, add_effects, delete_effects,
                                    set(), option, option_vars, null_sampler)
        nsrts.add(place_on_burner_nsrt)

        # PickJugFromFaucet
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, jug, faucet]
        option_vars = [robot, jug]
        option = PickJug
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtFaucet, [faucet]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        pick_jug_from_faucet_nsrt = NSRT("PickJugFromFaucet", parameters,
                                         preconditions, add_effects,
                                         delete_effects, set(), option,
                                         option_vars, null_sampler)
        nsrts.add(pick_jug_from_faucet_nsrt)

        # PickJugFromBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, jug, burner]
        option_vars = [robot, jug]
        option = PickJug
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtBurner, [jug, burner]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtBurner, [burner]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtBurner, [jug, burner]),
        }
        pick_jug_from_burner_nsrt = NSRT("PickJugFromBurner", parameters,
                                         preconditions, add_effects,
                                         delete_effects, set(), option,
                                         option_vars, null_sampler)
        nsrts.add(pick_jug_from_burner_nsrt)

        # PickJugFromOutsideFaucetAndBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        pick_jug_outside_faucet_burner_nsrt = NSRT(
            "PickJugFromOutsideFaucetAndBurner", parameters, preconditions,
            add_effects, delete_effects, set(), option, option_vars,
            null_sampler)
        nsrts.add(pick_jug_outside_faucet_burner_nsrt)

        # PlaceUnderFaucet
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, jug, faucet]
        option_vars = [robot, faucet]
        option = PlaceUnderFaucet
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtFaucet, [faucet]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtFaucet, [faucet]),
        }
        place_under_faucet_nsrt = NSRT("PlaceUnderFaucet", parameters,
                                       preconditions, add_effects,
                                       delete_effects, set(), option,
                                       option_vars, null_sampler)
        nsrts.add(place_under_faucet_nsrt)

        # PlaceOutsideFaucetAndBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot]
        option = PlaceOutsideBurnerAndFaucet
        preconditions = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        place_outside_faucet_burner_nsrt = NSRT("PlaceOutsideFaucetAndBurner",
                                                parameters, preconditions,
                                                add_effects, delete_effects,
                                                set(), option, option_vars,
                                                null_sampler)
        nsrts.add(place_outside_faucet_burner_nsrt)

        # SwitchFaucetOn
        robot = Variable("?robot", robot_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, faucet]
        option_vars = [robot, faucet]
        option = SwitchFaucetOn
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(FaucetOff, [faucet]),
        }
        add_effects = {
            LiftedAtom(FaucetOn, [faucet]),
        }
        delete_effects = {
            LiftedAtom(FaucetOff, [faucet]),
        }
        switch_faucet_on_nsrt = NSRT("SwitchFaucetOn", parameters,
                                     preconditions, add_effects,
                                     delete_effects, set(), option,
                                     option_vars, null_sampler)
        nsrts.add(switch_faucet_on_nsrt)

        # SwitchFaucetOff
        robot = Variable("?robot", robot_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, faucet]
        option_vars = [robot, faucet]
        option = SwitchFaucetOff
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(FaucetOn, [faucet]),
        }
        add_effects = {
            LiftedAtom(FaucetOff, [faucet]),
        }
        delete_effects = {
            LiftedAtom(FaucetOn, [faucet]),
        }
        switch_faucet_off_nsrt = NSRT("SwitchFaucetOff", parameters,
                                      preconditions, add_effects,
                                      delete_effects, set(), option,
                                      option_vars, null_sampler)
        nsrts.add(switch_faucet_off_nsrt)

        # SwitchBurnerOn
        robot = Variable("?robot", robot_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, burner]
        option_vars = [robot, burner]
        option = SwitchBurnerOn
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(BurnerOff, [burner]),
        }
        add_effects = {
            LiftedAtom(BurnerOn, [burner]),
        }
        delete_effects = {
            LiftedAtom(BurnerOff, [burner]),
        }
        switch_burner_on_nsrt = NSRT("SwitchBurnerOn", parameters,
                                     preconditions, add_effects,
                                     delete_effects, set(), option,
                                     option_vars, null_sampler)
        nsrts.add(switch_burner_on_nsrt)

        # SwitchBurnerOff
        robot = Variable("?robot", robot_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, burner]
        option_vars = [robot, burner]
        option = SwitchBurnerOff
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(BurnerOn, [burner]),
        }
        add_effects = {
            LiftedAtom(BurnerOff, [burner]),
        }
        delete_effects = {
            LiftedAtom(BurnerOn, [burner]),
        }
        switch_burner_off_nsrt = NSRT("SwitchBurnerOff", parameters,
                                      preconditions, add_effects,
                                      delete_effects, set(), option,
                                      option_vars, null_sampler)
        nsrts.add(switch_burner_off_nsrt)

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = Wait
        preconditions = set()
        add_effects = set()
        delete_effects = set()
        wait_nsrt = NSRT("Wait", parameters, preconditions, add_effects,
                         delete_effects, set(), option, option_vars,
                         null_sampler)
        nsrts.add(wait_nsrt)

        # DeclareComplete (only if task_completed goal)
        if CFG.boil_goal == "task_completed":
            robot = Variable("?robot", robot_type)
            jug = Variable("?jug", jug_type)
            burner = Variable("?burner", burner_type)
            parameters = [robot, jug, burner]
            option_vars = [robot]
            option = DeclareComplete
            preconditions = {
                LiftedAtom(NoWaterSpilled, []),
                LiftedAtom(WaterBoiled, [jug]),
                LiftedAtom(JugFilled, [jug]),
                LiftedAtom(BurnerOff, [burner]),
            }
            add_effects = {
                LiftedAtom(TaskCompleted, []),
            }
            delete_effects = set()
            declare_complete_nsrt = NSRT("DeclareComplete", parameters,
                                         preconditions, add_effects,
                                         delete_effects, set(), option,
                                         option_vars, null_sampler)
            nsrts.add(declare_complete_nsrt)

        return nsrts
