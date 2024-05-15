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
        patty_type = types["patty"]
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
        patty = Variable("?patty", patty_type)
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
        AdjacentToNothing = predicates["AdjacentToNothing"]
        Facing = predicates["Facing"]
        AdjacentNotFacing = predicates["AdjacentNotFacing"]
        IsCooked = predicates["IsCooked"]
        IsSliced = predicates["IsSliced"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]
        # GoalHack = predicates["GoalHack"]

        # Options
        Move = options["Move"]
        Pick = options["Pick"]
        Place = options["Place"]
        Cook = options["Cook"]
        # Slice = options["Slice"]

        nsrts = set()

        # Cook
        parameters = [robot, patty, grill]
        option_vars = [robot, patty, grill]
        option = Cook
        preconditions = {
            LiftedAtom(On, [patty, grill]),
            LiftedAtom(Facing, [robot, patty])
        }
        add_effects = {
            LiftedAtom(IsCooked, [patty])
        }
        delete_effects = set()
        ignore_effects = set()
        cook_nsrt = NSRT(
            "Cook",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(cook_nsrt)

        # MoveWhenAlreadyAdjacent
        parameters = [robot, to_obj, from_obj]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj]),
            LiftedAtom(Adjacent, [robot, to_obj]),
        }
        add_effects = {
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = set()
        ignore_effects = set()
        move_when_already_adjacent_nsrt = NSRT(
            "MoveWhenAlreadyAdjacent",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_when_already_adjacent_nsrt)

        # MoveFromNothing
        parameters = [robot, to_obj]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(AdjacentToNothing, [robot])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(AdjacentToNothing, [robot])
        }
        ignore_effects = set()
        move_from_nothing_nsrt = NSRT(
            "MoveFromNothing",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_from_nothing_nsrt)

        # MoveWhenFacingStart
        parameters = [robot, to_obj, from_obj]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj]),
            LiftedAtom(Facing, [robot, from_obj])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj]),
            LiftedAtom(Facing, [robot, from_obj])
        }
        ignore_effects = set()
        move_when_facing_start_nsrt = NSRT(
            "MoveWhenFacingStart",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_when_facing_start_nsrt)

        # MoveWhenNotFacingStart
        parameters = [robot, to_obj, from_obj]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj]),
            LiftedAtom(AdjacentNotFacing, [robot, from_obj])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj]),
            LiftedAtom(AdjacentNotFacing, [robot, from_obj])
        }
        ignore_effects = set()
        move_when_not_facing_start_nsrt = NSRT(
            "MoveWhenNotFacingStart",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_when_not_facing_start_nsrt)

        # PickMultipleAdjacent
        parameters = [robot, item]
        option_vars = [robot, item]
        option = Pick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item])
        }
        add_effects = {
            LiftedAtom(Holding, [robot, item])
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item])
        }

        ignore_effects: Set[Predicate] = set()
        pick_multiple_adjacent_nsrt = NSRT(
            "PickMultipleAdjacent",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(pick_multiple_adjacent_nsrt)

        # PickSingleAdjacent
        parameters = [robot, item]
        option_vars = [robot, item]
        option = Pick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item])
        }
        add_effects = {
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(AdjacentToNothing, [robot])
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item])
        }

        ignore_effects: Set[Predicate] = set()
        pick_single_adjacent_nsrt = NSRT(
            "PickSingleAdjacent",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(pick_single_adjacent_nsrt)


        # Place
        parameters = [robot, item, station]
        option_vars = [robot, item, station]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(Adjacent, [robot, station]),
            LiftedAtom(Facing, [robot, station])
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(On, [item, station]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item])
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, item])
        }
        ignore_effects: Set[Predicate] = set()
        place_nsrt = NSRT(
            "Place",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(place_nsrt)

        return nsrts
