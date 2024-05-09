"""Ground-truth NSRTs for the cover environment."""

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

        # Objects
        bottom_bun = Variable("?bottom_bun", bottom_bun_type)
        top_bun = Variable("?top_bun", top_bun_type)
        cheese = Variable("?cheese", cheese_type)
        tomato = Variable("?tomato", tomato_type)
        grill = Variable("?grill", grill_type)
        cutting_board = Variable("?cutting_board", cutting_board_type)
        robot = Variable("?robot", robot_type)
        item = Variable("?item", item_type)
        station = Variable(?"station", station_type)

        # Predicates
        Facing = predicates["Facing"]
        IsCooked = predicates["IsCooked"]
        IsSliced = predicates["IsSliced"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        On = predicates["On"]
        # GoalHack = predicates["GoalHack"]

        # Options
        Pick = options["Pick"]
        Place = options["Place"]
        Cook = options["Cook"]
        Slice = options["Slice"]

        nsrts = set()

        # Pick
        parameters = [robot, item]
        option_vars = [robot, item]
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(Facing, [robot, item])
        }
        add_effects = {
            LiftedAtom(Holding, [robot, item])
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot])
        }

        ignore_effects: Set[Predicate] = set()
        pick_nsrt = NSRT(
            "Pick",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            null_sampler
        )

        nsrts.add(pick_nsrt)

        return nsrts

        # # Types
        # object_type = types["obj"]
        #
        # # Objects
        # obj1 = Variable("?obj1", object_type)
        # obj2 = Variable("?obj2", object_type)
        # obj3 = Variable("?obj3", object_type)
        # obj4 = Variable("?obj4", object_type)
        # obj5 = Variable("?obj5", object_type)
        #
        # # Predicates
        # At = predicates["At"]
        # GoalCovered = predicates["GoalCovered"]
        # IsLoc = predicates["IsLoc"]
        # NoBoxAtLoc = predicates["NoBoxAtLoc"]
        # Above = predicates["Above"]
        # Below = predicates["Below"]
        # RightOf = predicates["RightOf"]
        # LeftOf = predicates["LeftOf"]
        # IsBox = predicates["IsBox"]
        # IsPlayer = predicates["IsPlayer"]
        # IsGoal = predicates["IsGoal"]
        # IsNonGoalLoc = predicates["IsNonGoalLoc"]
        #
        # # Options
        # PushUp = options["PushUp"]
        # PushDown = options["PushDown"]
        # PushLeft = options["PushLeft"]
        # PushRight = options["PushRight"]
        # MoveUp = options["MoveUp"]
        # MoveDown = options["MoveDown"]
        # MoveLeft = options["MoveLeft"]
        # MoveRight = options["MoveRight"]
        #
        # nsrts = set()
        #
        # # MoveUp
        # # Player, from_loc, to_loc
        # parameters = [obj1, obj2, obj3]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj2]),
        #     LiftedAtom(NoBoxAtLoc, [obj3]),
        #     LiftedAtom(Above, [obj3, obj2]),
        #     LiftedAtom(At, [obj1, obj2]),
        # }
        # add_effects = {LiftedAtom(At, [obj1, obj3])}
        # delete_effects = {LiftedAtom(At, [obj1, obj2])}
        # option = MoveUp
        # option_vars: List[Variable] = []  # dummy - not used
        # move_up_nsrt = NSRT("MoveUp", parameters, preconditions, add_effects,
        #                     delete_effects, set(), option, option_vars,
        #                     null_sampler)
        # nsrts.add(move_up_nsrt)
        #
        # # MoveDown
        # # Player, from_loc, to_loc
        # parameters = [obj1, obj2, obj3]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj2]),
        #     LiftedAtom(NoBoxAtLoc, [obj3]),
        #     LiftedAtom(Below, [obj3, obj2]),
        #     LiftedAtom(At, [obj1, obj2]),
        # }
        # add_effects = {LiftedAtom(At, [obj1, obj3])}
        # delete_effects = {LiftedAtom(At, [obj1, obj2])}
        # option = MoveDown
        # option_vars = []  # dummy - not used
        # move_down_nsrt = NSRT("MoveDown", parameters, preconditions,
        #                       add_effects, delete_effects, set(), option,
        #                       option_vars, null_sampler)
        # nsrts.add(move_down_nsrt)
        #
        # # MoveRight
        # # Player, from_loc, to_loc
        # parameters = [obj1, obj2, obj3]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj2]),
        #     LiftedAtom(NoBoxAtLoc, [obj3]),
        #     LiftedAtom(RightOf, [obj3, obj2]),
        #     LiftedAtom(At, [obj1, obj2]),
        # }
        # add_effects = {LiftedAtom(At, [obj1, obj3])}
        # delete_effects = {LiftedAtom(At, [obj1, obj2])}
        # option = MoveRight
        # option_vars = []  # dummy - not used
        # move_right_nsrt = NSRT("MoveRight", parameters, preconditions,
        #                        add_effects, delete_effects, set(), option,
        #                        option_vars, null_sampler)
        # nsrts.add(move_right_nsrt)
        #
        # # MoveLeft
        # # Player, from_loc, to_loc
        # parameters = [obj1, obj2, obj3]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj2]),
        #     LiftedAtom(NoBoxAtLoc, [obj3]),
        #     LiftedAtom(LeftOf, [obj3, obj2]),
        #     LiftedAtom(At, [obj1, obj2]),
        # }
        # add_effects = {LiftedAtom(At, [obj1, obj3])}
        # delete_effects = {LiftedAtom(At, [obj1, obj2])}
        # option = MoveLeft
        # option_vars = []  # dummy - not used
        # move_left_nsrt = NSRT("MoveLeft", parameters, preconditions,
        #                       add_effects, delete_effects, set(), option,
        #                       option_vars, null_sampler)
        # nsrts.add(move_left_nsrt)
        #
        # # PushUp
        # # Player, Box, player_loc, box_loc, pushto_loc
        # parameters = [obj1, obj2, obj3, obj4, obj5]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsBox, [obj2]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj4]),
        #     LiftedAtom(IsLoc, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(Above, [obj4, obj3]),
        #     LiftedAtom(Above, [obj5, obj4]),
        #     LiftedAtom(IsNonGoalLoc, [obj5]),
        # }
        # add_effects = {
        #     LiftedAtom(At, [obj2, obj5]),
        #     LiftedAtom(At, [obj1, obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj4]),
        # }
        # delete_effects = {
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(GoalCovered, [obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj5])
        # }
        # option = PushUp
        # option_vars = []  # dummy - not used
        # push_up_nsrt = NSRT("PushUp", parameters, preconditions, add_effects,
        #                     delete_effects, set(), option, option_vars,
        #                     null_sampler)
        # nsrts.add(push_up_nsrt)
        #
        # # PushDown
        # # Player, Box, player_loc, box_loc, pushto_loc
        # parameters = [obj1, obj2, obj3, obj4, obj5]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsBox, [obj2]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj4]),
        #     LiftedAtom(IsLoc, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(Below, [obj4, obj3]),
        #     LiftedAtom(Below, [obj5, obj4]),
        #     LiftedAtom(IsNonGoalLoc, [obj5]),
        # }
        # add_effects = {
        #     LiftedAtom(At, [obj2, obj5]),
        #     LiftedAtom(At, [obj1, obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj4]),
        # }
        # delete_effects = {
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(GoalCovered, [obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        # }
        # option = PushDown
        # option_vars = []  # dummy - not used
        # push_down_nsrt = NSRT("PushDown", parameters, preconditions,
        #                       add_effects, delete_effects, set(), option,
        #                       option_vars, null_sampler)
        # nsrts.add(push_down_nsrt)
        #
        # # PushRight
        # # Player, Box, player_loc, box_loc, pushto_loc
        # parameters = [obj1, obj2, obj3, obj4, obj5]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsBox, [obj2]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj4]),
        #     LiftedAtom(IsLoc, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(RightOf, [obj4, obj3]),
        #     LiftedAtom(RightOf, [obj5, obj4]),
        #     LiftedAtom(IsNonGoalLoc, [obj5]),
        # }
        # add_effects = {
        #     LiftedAtom(At, [obj2, obj5]),
        #     LiftedAtom(At, [obj1, obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj4]),
        # }
        # delete_effects = {
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(GoalCovered, [obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        # }
        # option = PushRight
        # option_vars = []  # dummy - not used
        # push_right_nsrt = NSRT("PushRight", parameters, preconditions,
        #                        add_effects, delete_effects, set(), option,
        #                        option_vars, null_sampler)
        # nsrts.add(push_right_nsrt)
        #
        # # PushLeft
        # # Player, Box, player_loc, box_loc, pushto_loc
        # parameters = [obj1, obj2, obj3, obj4, obj5]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsBox, [obj2]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj4]),
        #     LiftedAtom(IsLoc, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(LeftOf, [obj4, obj3]),
        #     LiftedAtom(LeftOf, [obj5, obj4]),
        #     LiftedAtom(IsNonGoalLoc, [obj5]),
        # }
        # add_effects = {
        #     LiftedAtom(At, [obj2, obj5]),
        #     LiftedAtom(At, [obj1, obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj4]),
        # }
        # delete_effects = {
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(GoalCovered, [obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        # }
        # option = PushLeft
        # option_vars = []  # dummy - not used
        # push_left_nsrt = NSRT("PushLeft", parameters, preconditions,
        #                       add_effects, delete_effects, set(), option,
        #                       option_vars, null_sampler)
        # nsrts.add(push_left_nsrt)
        #
        # # PushUpGoal
        # # Player, Box, player_loc, box_loc, goal_loc
        # parameters = [obj1, obj2, obj3, obj4, obj5]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsBox, [obj2]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj4]),
        #     LiftedAtom(IsGoal, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(Above, [obj4, obj3]),
        #     LiftedAtom(Above, [obj5, obj4]),
        # }
        # add_effects = {
        #     LiftedAtom(At, [obj2, obj5]),
        #     LiftedAtom(At, [obj1, obj4]),
        #     LiftedAtom(GoalCovered, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj4]),
        # }
        # delete_effects = {
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(GoalCovered, [obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        # }
        # option = PushUp
        # option_vars = []  # dummy - not used
        # push_up_goal_nsrt = NSRT("PushUpGoal", parameters,
        #                          preconditions, add_effects, delete_effects,
        #                          set(), option, option_vars, null_sampler)
        # nsrts.add(push_up_goal_nsrt)
        #
        # # PushDownGoal
        # # Player, Box, player_loc, box_loc, goal_loc
        # parameters = [obj1, obj2, obj3, obj4, obj5]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsBox, [obj2]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj4]),
        #     LiftedAtom(IsGoal, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(Below, [obj4, obj3]),
        #     LiftedAtom(Below, [obj5, obj4]),
        # }
        # add_effects = {
        #     LiftedAtom(At, [obj2, obj5]),
        #     LiftedAtom(At, [obj1, obj4]),
        #     LiftedAtom(GoalCovered, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj4]),
        # }
        # delete_effects = {
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(GoalCovered, [obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        # }
        # option = PushDown
        # option_vars = []  # dummy - not used
        # push_down_goal_nsrt = NSRT("PushDownGoal", parameters,
        #                            preconditions, add_effects, delete_effects,
        #                            set(), option, option_vars, null_sampler)
        # nsrts.add(push_down_goal_nsrt)
        #
        # # PushRightGoal
        # # Player, Box, player_loc, box_loc, goal_loc
        # parameters = [obj1, obj2, obj3, obj4, obj5]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsBox, [obj2]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj4]),
        #     LiftedAtom(IsGoal, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(RightOf, [obj4, obj3]),
        #     LiftedAtom(RightOf, [obj5, obj4]),
        # }
        # add_effects = {
        #     LiftedAtom(At, [obj2, obj5]),
        #     LiftedAtom(At, [obj1, obj4]),
        #     LiftedAtom(GoalCovered, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj4]),
        # }
        # delete_effects = {
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(GoalCovered, [obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        # }
        # option = PushRight
        # option_vars = []  # dummy - not used
        # push_right_goal_nsrt = NSRT("PushRightGoal", parameters,
        #                             preconditions, add_effects, delete_effects,
        #                             set(), option, option_vars, null_sampler)
        # nsrts.add(push_right_goal_nsrt)
        #
        # # PushLeftGoal
        # # Player, Box, player_loc, box_loc, goal_loc
        # parameters = [obj1, obj2, obj3, obj4, obj5]
        # preconditions = {
        #     LiftedAtom(IsPlayer, [obj1]),
        #     LiftedAtom(IsBox, [obj2]),
        #     LiftedAtom(IsLoc, [obj3]),
        #     LiftedAtom(IsLoc, [obj4]),
        #     LiftedAtom(IsGoal, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(LeftOf, [obj4, obj3]),
        #     LiftedAtom(LeftOf, [obj5, obj4]),
        # }
        # add_effects = {
        #     LiftedAtom(At, [obj2, obj5]),
        #     LiftedAtom(At, [obj1, obj4]),
        #     LiftedAtom(GoalCovered, [obj5]),
        #     LiftedAtom(NoBoxAtLoc, [obj4]),
        # }
        # delete_effects = {
        #     LiftedAtom(At, [obj1, obj3]),
        #     LiftedAtom(At, [obj2, obj4]),
        #     LiftedAtom(GoalCovered, [obj4]),
        #     LiftedAtom(NoBoxAtLoc, [obj5]),
        # }
        # option = PushLeft
        # option_vars = []  # dummy - not used
        # push_left_goal_nsrt = NSRT("PushLeftGoal", parameters,
        #                            preconditions, add_effects, delete_effects,
        #                            set(), option, option_vars, null_sampler)
        # nsrts.add(push_left_goal_nsrt)
        #
        # return nsrts
