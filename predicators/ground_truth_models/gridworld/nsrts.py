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
        from_obj = Variable("?from_obj", item_type)

        to_obj = Variable("?to_obj", object_type)
        object = Variable("?object", object_type)

        from_obj1 = Variable("?from_obj1", item_type)
        from_obj2 = Variable("?from_obj2", item_type)
        from_obj3 = Variable("?from_obj3", item_type)
        from_obj4 = Variable("?from_obj4", object_type)

        to_obj1 = Variable("?to_obj1", item_type)
        to_obj2 = Variable("?to_obj2", item_type)
        to_obj3 = Variable("?to_obj3", item_type)
        to_obj4 = Variable("?to_obj4", object_type)

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
        OnNothing = predicates["OnNothing"]
        Clear = predicates["Clear"]
        # GoalHack = predicates["GoalHack"]

        # Options
        Move = options["Move"]
        Pick = options["Pick"]
        Place = options["Place"]
        Cook = options["Cook"]
        Slice = options["Slice"]

        nsrts = set()

        # Slice
        parameters = [robot, tomato, cutting_board]
        option_vars = [robot, tomato, cutting_board]
        option = Slice
        preconditions = {
            LiftedAtom(Clear, [tomato]),
            LiftedAtom(On, [tomato, cutting_board]),
            LiftedAtom(Facing, [robot, tomato])
        }
        add_effects = {
            LiftedAtom(IsSliced, [tomato])
        }
        delete_effects = set()
        ignore_effects = set()
        slice_nsrt = NSRT(
            "Slice",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(slice_nsrt)

        # Cook
        parameters = [robot, patty, grill]
        option_vars = [robot, patty, grill]
        option = Cook
        preconditions = {
            LiftedAtom(Clear, [patty]),
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
        parameters = [robot, to_obj, from_obj3]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj3]),
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, from_obj3])
        }
        add_effects = {
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Facing, [robot, from_obj3])
        }
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

        # MoveFromNothingToOneStack
        parameters = [robot, to_obj]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(AdjacentToNothing, [robot]),
            LiftedAtom(Clear, [to_obj]),
            LiftedAtom(OnNothing, [to_obj])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(AdjacentToNothing, [robot])
        }
        ignore_effects = set()
        move_from_nothing_to_one_stack_nsrt = NSRT(
            "MoveFromNothingToOneStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_from_nothing_to_one_stack_nsrt)

        # MoveFromNothingToTwoStack
        parameters = [robot, to_obj1, to_obj4]
        option_vars = [robot, to_obj1]
        option = Move
        preconditions = {
            LiftedAtom(AdjacentToNothing, [robot]),
            LiftedAtom(Clear, [to_obj1]),
            LiftedAtom(On, [to_obj1, to_obj4]),
            LiftedAtom(OnNothing, [to_obj4]),
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj1]),
            LiftedAtom(Facing, [robot, to_obj1]),
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4]),
        }
        delete_effects = {
            LiftedAtom(AdjacentToNothing, [robot])
        }
        ignore_effects = set()
        move_from_nothing_to_two_stack_nsrt = NSRT(
            "MoveFromNothingToTwoStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_from_nothing_to_two_stack_nsrt)

        # MoveFromNothingToThreeStack
        parameters = [robot, to_obj1, to_obj2, to_obj4]
        option_vars = [robot, to_obj1]
        option = Move
        preconditions = {
            LiftedAtom(AdjacentToNothing, [robot]),
            LiftedAtom(Clear, [to_obj1]),
            LiftedAtom(On, [to_obj1, to_obj2]),
            LiftedAtom(On, [to_obj2, to_obj4]),
            LiftedAtom(OnNothing, [to_obj4]),
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj1]),
            LiftedAtom(Facing, [robot, to_obj1]),
            LiftedAtom(Adjacent, [robot, to_obj2]),
            LiftedAtom(Facing, [robot, to_obj2]),
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4]),
        }
        delete_effects = {
            LiftedAtom(AdjacentToNothing, [robot])
        }
        ignore_effects = set()
        move_from_nothing_to_three_stack_nsrt = NSRT(
            "MoveFromNothingToThreeStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_from_nothing_to_three_stack_nsrt)

        # MoveFromNothingToFourStack
        parameters = [robot, to_obj1, to_obj2, to_obj3, to_obj4]
        option_vars = [robot, to_obj1]
        option = Move
        preconditions = {
            LiftedAtom(AdjacentToNothing, [robot]),
            LiftedAtom(Clear, [to_obj1]),
            LiftedAtom(On, [to_obj1, to_obj2]),
            LiftedAtom(On, [to_obj2, to_obj3]),
            LiftedAtom(On, [to_obj3, to_obj4]),
            LiftedAtom(OnNothing, [to_obj4]),
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj1]),
            LiftedAtom(Facing, [robot, to_obj1]),
            LiftedAtom(Adjacent, [robot, to_obj2]),
            LiftedAtom(Facing, [robot, to_obj2]),
            LiftedAtom(Adjacent, [robot, to_obj3]),
            LiftedAtom(Facing, [robot, to_obj3]),
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4])
        }
        delete_effects = {
            LiftedAtom(AdjacentToNothing, [robot])
        }
        ignore_effects = set()
        move_from_nothing_to_four_stack_nsrt = NSRT(
            "MoveFromNothingToFourStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_from_nothing_to_four_stack_nsrt)

        # MoveWhenFacingOneStack
        parameters = [robot, to_obj, from_obj4]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
            LiftedAtom(Clear, [from_obj4]),
            LiftedAtom(OnNothing, [from_obj4])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
        }
        ignore_effects = set()
        move_when_facing_one_stack_nsrt = NSRT(
            "MoveWhenFacingOneStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_when_facing_one_stack_nsrt)

        # MoveWhenFacingTwoStack
        parameters = [robot, to_obj, from_obj1, from_obj4]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
            LiftedAtom(Clear, [from_obj1]),
            LiftedAtom(On, [from_obj1, from_obj4]),
            LiftedAtom(OnNothing, [from_obj4])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4])
        }
        ignore_effects = set()
        move_when_facing_two_stack_nsrt = NSRT(
            "MoveWhenFacingTwoStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_when_facing_two_stack_nsrt)

        # MoveWhenFacingThreeStack
        parameters = [robot, to_obj, from_obj1, from_obj2, from_obj4]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj2]),
            LiftedAtom(Facing, [robot, from_obj2]),
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
            LiftedAtom(Clear, [from_obj1]),
            LiftedAtom(On, [from_obj1, from_obj2]),
            LiftedAtom(On, [from_obj2, from_obj4]),
            LiftedAtom(OnNothing, [from_obj4])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj2]),
            LiftedAtom(Facing, [robot, from_obj2]),
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4])
        }
        ignore_effects = set()
        move_when_facing_three_stack_nsrt = NSRT(
            "MoveWhenFacingThreeStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_when_facing_three_stack_nsrt)

        # MoveWhenFacingFourStack
        parameters = [robot, to_obj, from_obj1, from_obj2, from_obj3, from_obj4]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj2]),
            LiftedAtom(Facing, [robot, from_obj2]),
            LiftedAtom(Adjacent, [robot, from_obj3]),
            LiftedAtom(Facing, [robot, from_obj3]),
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
            LiftedAtom(Clear, [from_obj1]),
            LiftedAtom(On, [from_obj1, from_obj2]),
            LiftedAtom(On, [from_obj2, from_obj3]),
            LiftedAtom(On, [from_obj3, from_obj4]),
            LiftedAtom(OnNothing, [from_obj4])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj2]),
            LiftedAtom(Facing, [robot, from_obj2]),
            LiftedAtom(Adjacent, [robot, from_obj3]),
            LiftedAtom(Facing, [robot, from_obj3]),
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4])
        }
        ignore_effects = set()
        move_when_facing_four_stack_nsrt = NSRT(
            "MoveWhenFacingFourStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_when_facing_four_stack_nsrt)

        # MoveWhenFacingThreeStack
        parameters = [robot, to_obj, from_obj1, from_obj2, from_obj3]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj2]),
            LiftedAtom(Facing, [robot, from_obj2]),
            LiftedAtom(Adjacent, [robot, from_obj3]),
            LiftedAtom(Facing, [robot, from_obj3]),
            LiftedAtom(Clear, [from_obj1]),
            LiftedAtom(On, [from_obj1, from_obj2]),
            LiftedAtom(On, [from_obj2, from_obj3]),
            LiftedAtom(OnNothing, [from_obj3])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj2]),
            LiftedAtom(Facing, [robot, from_obj2]),
            LiftedAtom(Adjacent, [robot, from_obj3]),
            LiftedAtom(Facing, [robot, from_obj3])
        }
        ignore_effects = set()
        move_when_facing_three_stack_nsrt = NSRT(
            "MoveWhenFacingThreeStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_when_facing_three_stack_nsrt)

        # MoveFromOneStackToThreeStack
        parameters = [robot, to_obj1, to_obj2, to_obj4, from_obj4]
        option_vars = [robot, to_obj1]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
            LiftedAtom(Clear, [from_obj4]),
            LiftedAtom(OnNothing, [from_obj4]),
            LiftedAtom(Clear, [to_obj1]),
            LiftedAtom(On, [to_obj1, to_obj2]),
            LiftedAtom(On, [to_obj2, to_obj4]),
            LiftedAtom(OnNothing, [to_obj4]),
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj1]),
            LiftedAtom(Facing, [robot, to_obj1]),
            LiftedAtom(Adjacent, [robot, to_obj2]),
            LiftedAtom(Facing, [robot, to_obj2]),
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4]),
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
        }
        ignore_effects = set()
        move_from_one_stack_to_three_stack_nsrt = NSRT(
            "MoveFromOneStackToThreeStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(move_from_one_stack_to_three_stack_nsrt)

        # MoveWhenNotFacingStart
        parameters = [robot, to_obj, from_obj3]
        option_vars = [robot, to_obj]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj3]),
            LiftedAtom(AdjacentNotFacing, [robot, from_obj3])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj]),
            LiftedAtom(Facing, [robot, to_obj])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj3]),
            LiftedAtom(AdjacentNotFacing, [robot, from_obj3])
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
            LiftedAtom(Facing, [robot, item]),
            LiftedAtom(OnNothing, [item]),
            LiftedAtom(Clear, [item])
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
            LiftedAtom(Facing, [robot, item]),
            LiftedAtom(OnNothing, [item]),
            LiftedAtom(Clear, [item])
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

        # PickFromStack
        parameters = [robot, item, object]
        option_vars = [robot, item]
        option = Pick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item]),
            LiftedAtom(Adjacent, [robot, object]),
            LiftedAtom(Facing, [robot, object]),
            LiftedAtom(On, [item, object]),
            LiftedAtom(Clear, [item])
        }
        add_effects = {
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(Clear, [object])
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item]),
            LiftedAtom(On, [item, object]),
        }
        ignore_effects: Set[Predicate] = set()
        pick_from_stack_nsrt = NSRT(
            "PickFromStack",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )
        nsrts.add(pick_from_stack_nsrt)

        # Place
        parameters = [robot, item, object]
        option_vars = [robot, item, object]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(Adjacent, [robot, object]),
            LiftedAtom(Facing, [robot, object]),
            LiftedAtom(Clear, [object])
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(On, [item, object]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item])
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(OnNothing, [item]),
            LiftedAtom(Clear, [object])
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

        # # Place
        # parameters = [robot, item, station]
        # option_vars = [robot, item, station]
        # option = Place
        # preconditions = {
        #     LiftedAtom(Holding, [robot, item]),
        #     LiftedAtom(Adjacent, [robot, station]),
        #     LiftedAtom(Facing, [robot, station])
        # }
        # add_effects = {
        #     LiftedAtom(HandEmpty, [robot]),
        #     LiftedAtom(On, [item, station]),
        #     LiftedAtom(Adjacent, [robot, item]),
        #     LiftedAtom(Facing, [robot, item])
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

        return nsrts
