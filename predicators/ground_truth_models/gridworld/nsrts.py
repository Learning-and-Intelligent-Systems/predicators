"""Ground-truth NSRTs for the gridworld environment."""

from typing import Dict, List, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class GridWorldGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the GridWorld environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"gridworld"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        bottom_bun_type = types["bottom_bun"]
        top_bun_type = types["top_bun"]
        patty = types["patty"]
        cheese_type = types["cheese"]
        tomato_type = types["tomato"]
        grill_type = types["grill"]
        cutting_board_type = types["cutting_board"]
        robot_type = types["robot"]
        item_type = types["item"]
        station_type = types["station"]
        object_type = types["object"]

        # Objects
        bottom_bun = Variable("?bottom_bun", bottom_bun_type)
        top_bun = Variable("?top_bun", top_bun_type)
        cheese = Variable("?cheese", cheese_type)
        tomato = Variable("?tomato", tomato_type)
        grill = Variable("?grill", grill_type)
        cutting_board = Variable("?cutting_board", cutting_board_type)
        robot = Variable("?robot", robot_type)
        item = Variable("?item", item_type)
        station = Variable("?station", station_type)
        from_obj = Variable("?from_obj", object_type)
        to_obj = Variable("?to_obj", object_type)

        # Predicates
        Adjacent = predicates["Adjacent"]
        Facing = predicates["Facing"]
        IsCooked = predicates["IsCooked"]
        IsSliced = predicates["IsSliced"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]
        # GoalHack = predicates["GoalHack"]

        # Options
        Move = options["Move"]
        # Pick = options["Pick"]
        # Place = options["Place"]
        # Cook = options["Cook"]
        # Slice = options["Slice"]

        nsrts = set()

        # Move
        parameters = [robot, to_obj, from_obj]
        option_vars = [robot, to_obj, from_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj])
        }
        ignore_effects = set()
        move_nsrt = NSRT(
            "Move",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_nsrt)
        return nsrts

        # # Pick
        # parameters = [robot, item]
        # option_vars = [robot, item]
        # option = Pick
        # preconditions = {
        #     LiftedAtom(HandEmpty, [robot]),
        # }
        # add_effects = {
        #     LiftedAtom(Holding, [robot, item])
        # }
        # delete_effects = {
        #     LiftedAtom(HandEmpty, [robot])
        # }
        #
        # ignore_effects: Set[Predicate] = set()
        # pick_nsrt = NSRT(
        #     "Pick",
        #     parameters,
        #     preconditions,
        #     add_effects,
        #     delete_effects,
        #     ignore_effects,
        #     option,
        #     option_vars,
        #     null_sampler
        # )
        # nsrts.add(pick_nsrt)
        #
        # # Place
        # parameters = [robot, item, station]
        # option_vars = [robot, item, station]
        # option = Place
        # preconditions = {
        #     LiftedAtom(Holding, [robot, item]),
        #     LiftedAtom(Facing, [robot, station])
        # }
        # add_effects = {
        #     LiftedAtom(HandEmpty, [robot]),
        #     LiftedAtom(On, [item, station])
        # }
        # delete_effects = {
        #     LiftedAtom(Holding, [robot, item])
        # }
        # ignore_effects: Set[Predicate] = set()
        # place_nsrt = NSRT(
        #     "Place",
        #     parameters,
        #     preconditions,
        #     add_effects,
        #     delete_effects,
        #     ignore_effects,
        #     option,
        #     option_vars,
        #     null_sampler
        # )
        # nsrts.add(place_nsrt)
        #
        # return nsrts
