"""Ground-truth NSRTs for the burger environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class BurgerGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Burger environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"burger"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        tomato_type = types["tomato"]
        patty_type = types["patty"]

        grill_type = types["grill"]
        cutting_board_type = types["cutting_board"]
        robot_type = types["robot"]

        item_type = types["item"]
        object_type = types["object"]

        # Variables
        tomato = Variable("?tomato", tomato_type)
        patty = Variable("?patty", patty_type)

        grill = Variable("?grill", grill_type)
        cutting_board = Variable("?cutting_board", cutting_board_type)
        robot = Variable("?robot", robot_type)

        item = Variable("?item", item_type)
        obj = Variable("?object", object_type)

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
        IsCooked = predicates["IsCooked"]
        IsSliced = predicates["IsSliced"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]
        OnNothing = predicates["OnNothing"]
        Clear = predicates["Clear"]

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
        add_effects = {LiftedAtom(IsSliced, [tomato])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects: Set[Predicate] = set()
        slice_nsrt = NSRT("Slice", parameters, preconditions, add_effects,
                          delete_effects, ignore_effects, option, option_vars,
                          null_sampler)

        # Cook
        parameters = [robot, patty, grill]
        option_vars = [robot, patty, grill]
        option = Cook
        preconditions = {
            LiftedAtom(Clear, [patty]),
            LiftedAtom(On, [patty, grill]),
            LiftedAtom(Facing, [robot, patty])
        }
        add_effects = {LiftedAtom(IsCooked, [patty])}
        delete_effects = set()
        ignore_effects = set()
        cook_nsrt = NSRT("Cook", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         null_sampler)

        # NOTE: this nsrt will be relevant after the environment is updated to
        # have more variation in the tasks' initial states.
        # # MoveWhenAlreadyAdjacent
        # parameters = [robot, to_obj, from_obj3]
        # option_vars = [robot, to_obj]
        # option = Move
        # preconditions = {
        #     LiftedAtom(Adjacent, [robot, from_obj3]),
        #     LiftedAtom(Adjacent, [robot, to_obj]),
        #     LiftedAtom(Facing, [robot, from_obj3])
        # }
        # add_effects = {
        #     LiftedAtom(Facing, [robot, to_obj])
        # }
        # delete_effects = {
        #     LiftedAtom(Facing, [robot, from_obj3])
        # }
        # ignore_effects = set()
        # move_when_already_adjacent_nsrt = NSRT(
        #     "MoveWhenAlreadyAdjacent",
        #     parameters,
        #     preconditions,
        #     add_effects,
        #     delete_effects,
        #     ignore_effects,
        #     option,
        #     option_vars,
        #     null_sampler
        # )
        # nsrts.add(move_when_already_adjacent_nsrt)

        # MoveFromNothingToOneStack
        parameters = [robot, to_obj4]
        option_vars = [robot, to_obj4]
        option = Move
        preconditions = {
            LiftedAtom(AdjacentToNothing, [robot]),
            LiftedAtom(Clear, [to_obj4]),
            LiftedAtom(OnNothing, [to_obj4])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4])
        }
        delete_effects = {LiftedAtom(AdjacentToNothing, [robot])}
        ignore_effects = set()
        move_from_nothing_to_one_stack_nsrt = NSRT("MoveFromNothingToOneStack",
                                                   parameters, preconditions,
                                                   add_effects, delete_effects,
                                                   ignore_effects, option,
                                                   option_vars, null_sampler)

        # MoveFromNothing
        parameters = [robot, to_obj4]
        option_vars = [robot, to_obj4]
        option = Move
        # okay to be less specific
        preconditions = set()
        # okay to be less specific
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4])
        }
        # okay to be less specific
        delete_effects = set()
        ignore_effects = set()
        move_from_nothing = NSRT("Move", parameters, preconditions,
                                 add_effects, delete_effects, ignore_effects,
                                 option, option_vars, null_sampler)

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
        delete_effects = {LiftedAtom(AdjacentToNothing, [robot])}
        ignore_effects = set()
        move_from_nothing_to_two_stack_nsrt = NSRT("MoveFromNothingToTwoStack",
                                                   parameters, preconditions,
                                                   add_effects, delete_effects,
                                                   ignore_effects, option,
                                                   option_vars, null_sampler)

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
        delete_effects = {LiftedAtom(AdjacentToNothing, [robot])}
        ignore_effects = set()
        move_from_nothing_to_three_stack_nsrt = NSRT(
            "MoveFromNothingToThreeStack", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            null_sampler)

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
        delete_effects = {LiftedAtom(AdjacentToNothing, [robot])}
        ignore_effects = set()
        move_from_nothing_to_four_stack_nsrt = NSRT(
            "MoveFromNothingToFourStack", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            null_sampler)

        # MoveWhenFacingOneStack
        parameters = [robot, to_obj4, from_obj4]
        option_vars = [robot, to_obj4]
        option = Move
        preconditions = {
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
            LiftedAtom(Clear, [from_obj4]),
            LiftedAtom(OnNothing, [from_obj4])
        }
        add_effects = {
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4]),
        }
        ignore_effects = set()
        move_when_facing_one_stack_nsrt = NSRT("MoveWhenFacingOneStack",
                                               parameters, preconditions,
                                               add_effects, delete_effects,
                                               ignore_effects, option,
                                               option_vars, null_sampler)

        # MoveWhenFacingTwoStack
        parameters = [robot, to_obj4, from_obj1, from_obj4]
        option_vars = [robot, to_obj4]
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
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4])
        }
        delete_effects = {
            LiftedAtom(Adjacent, [robot, from_obj1]),
            LiftedAtom(Facing, [robot, from_obj1]),
            LiftedAtom(Adjacent, [robot, from_obj4]),
            LiftedAtom(Facing, [robot, from_obj4])
        }
        ignore_effects = set()
        move_when_facing_two_stack_nsrt = NSRT("MoveWhenFacingTwoStack",
                                               parameters, preconditions,
                                               add_effects, delete_effects,
                                               ignore_effects, option,
                                               option_vars, null_sampler)

        # MoveWhenFacingThreeStack
        parameters = [robot, to_obj4, from_obj1, from_obj2, from_obj4]
        option_vars = [robot, to_obj4]
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
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4])
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
        move_when_facing_three_stack_nsrt = NSRT("MoveWhenFacingThreeStack",
                                                 parameters, preconditions,
                                                 add_effects, delete_effects,
                                                 ignore_effects, option,
                                                 option_vars, null_sampler)

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
            "MoveFromOneStackToThreeStack", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            null_sampler)

        # MoveWhenFacingFourStack
        parameters = [
            robot, to_obj4, from_obj1, from_obj2, from_obj3, from_obj4
        ]
        option_vars = [robot, to_obj4]
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
            LiftedAtom(Adjacent, [robot, to_obj4]),
            LiftedAtom(Facing, [robot, to_obj4])
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
        move_when_facing_four_stack_nsrt = NSRT("MoveWhenFacingFourStack",
                                                parameters, preconditions,
                                                add_effects, delete_effects,
                                                ignore_effects, option,
                                                option_vars, null_sampler)

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
            "MoveFromOneStackToThreeStack", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            null_sampler)

        # NOTE: this nsrt will be relevant after the environment is updated to
        # have more variation in the tasks' initial states.
        # # MoveWhenNotFacingStart
        # parameters = [robot, to_obj, from_obj3]
        # option_vars = [robot, to_obj]
        # option = Move
        # preconditions = {
        #     LiftedAtom(Adjacent, [robot, from_obj3]),
        #     LiftedAtom(AdjacentNotFacing, [robot, from_obj3])
        # }
        # add_effects = {
        #     LiftedAtom(Adjacent, [robot, to_obj]),
        #     LiftedAtom(Facing, [robot, to_obj])
        # }
        # delete_effects = {
        #     LiftedAtom(Adjacent, [robot, from_obj3]),
        #     LiftedAtom(AdjacentNotFacing, [robot, from_obj3])
        # }
        # ignore_effects = set()
        # move_when_not_facing_start_nsrt = NSRT(
        #     "MoveWhenNotFacingStart",
        #     parameters,
        #     preconditions,
        #     add_effects,
        #     delete_effects,
        #     ignore_effects,
        #     option,
        #     option_vars,
        #     null_sampler
        # )
        # nsrts.add(move_when_not_facing_start_nsrt)

        # NOTE: this nsrt will be relevant after the environment is updated to
        # have more variation in the tasks' initial states.
        # # PickMultipleAdjacent
        # parameters = [robot, item]
        # option_vars = [robot, item]
        # option = Pick
        # preconditions = {
        #     LiftedAtom(HandEmpty, [robot]),
        #     LiftedAtom(Adjacent, [robot, item]),
        #     LiftedAtom(Facing, [robot, item]),
        #     LiftedAtom(OnNothing, [item]),
        #     LiftedAtom(Clear, [item])
        # }
        # add_effects = {
        #     LiftedAtom(Holding, [robot, item])
        # }
        # delete_effects = {
        #     LiftedAtom(HandEmpty, [robot]),
        #     LiftedAtom(Adjacent, [robot, item]),
        #     LiftedAtom(Facing, [robot, item])
        # }
        # ignore_effects: Set[Predicate] = set()
        # pick_multiple_adjacent_nsrt = NSRT(
        #     "PickMultipleAdjacent",
        #     parameters,
        #     preconditions,
        #     add_effects,
        #     delete_effects,
        #     ignore_effects,
        #     option,
        #     option_vars,
        #     null_sampler
        # )
        # nsrts.add(pick_multiple_adjacent_nsrt)

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
        ignore_effects: Set[Predicate] = set()  # type: ignore[no-redef]
        pick_single_adjacent_nsrt = NSRT("PickSingleAdjacent", parameters,
                                         preconditions, add_effects,
                                         delete_effects, ignore_effects,
                                         option, option_vars, null_sampler)

        # PickFromStack
        parameters = [robot, item, obj]
        option_vars = [robot, item]
        option = Pick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item]),
            LiftedAtom(Adjacent, [robot, obj]),
            LiftedAtom(Facing, [robot, obj]),
            LiftedAtom(On, [item, obj]),
            LiftedAtom(Clear, [item])
        }
        add_effects = {
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(Clear, [obj])
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item]),
            LiftedAtom(On, [item, obj]),
        }
        ignore_effects: Set[Predicate] = set()  # type: ignore[no-redef]
        pick_from_stack_nsrt = NSRT("PickFromStack", parameters, preconditions,
                                    add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)

        # Place
        parameters = [robot, item, obj]
        option_vars = [robot, item, obj]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(Adjacent, [robot, obj]),
            LiftedAtom(Facing, [robot, obj]),
            LiftedAtom(Clear, [obj])
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(On, [item, obj]),
            LiftedAtom(Adjacent, [robot, item]),
            LiftedAtom(Facing, [robot, item])
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, item]),
            LiftedAtom(OnNothing, [item]),
            LiftedAtom(Clear, [obj])
        }
        ignore_effects: Set[Predicate] = set()  # type: ignore[no-redef]
        place_nsrt = NSRT("Place", parameters, preconditions, add_effects,
                          delete_effects, ignore_effects, option, option_vars,
                          null_sampler)

        nsrts.add(slice_nsrt)
        nsrts.add(cook_nsrt)
        # nsrts.add(move_from_nothing) # yichao add
        nsrts.add(move_from_nothing_to_one_stack_nsrt)
        nsrts.add(move_from_nothing_to_two_stack_nsrt)
        nsrts.add(move_from_nothing_to_three_stack_nsrt)
        nsrts.add(move_from_nothing_to_four_stack_nsrt)
        nsrts.add(move_when_facing_one_stack_nsrt)
        nsrts.add(move_when_facing_two_stack_nsrt)
        nsrts.add(move_when_facing_three_stack_nsrt)
        nsrts.add(move_when_facing_four_stack_nsrt)
        nsrts.add(move_from_one_stack_to_three_stack_nsrt)
        nsrts.add(pick_single_adjacent_nsrt)
        nsrts.add(pick_from_stack_nsrt)
        nsrts.add(place_nsrt)

        return nsrts
