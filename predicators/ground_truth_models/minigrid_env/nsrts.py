"""Ground-truth NSRTs for the cover environment."""

from typing import Dict, List, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class MiniGridGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the MiniGrid environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"minigrid_env"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        object_type = types["obj"]

        # Objects
        obj1 = Variable("?obj1", object_type)
        obj2 = Variable("?obj2", object_type)
        obj3 = Variable("?obj3", object_type)
        obj3 = Variable("?obj4", object_type)

        # Predicates
        At = predicates["At"]
        IsLoc = predicates["IsLoc"]
        Above = predicates["Above"]
        Below = predicates["Below"]
        RightOf = predicates["RightOf"]
        LeftOf = predicates["LeftOf"]
        IsAgent = predicates["IsAgent"]
        IsFacingUp = predicates["IsFacingUp"]
        IsFacingDown = predicates["IsFacingDown"]
        IsFacingLeft = predicates["IsFacingLeft"]
        IsFacingRight = predicates["IsFacingRight"]

        # Options
        MoveForward = options["Forward"]
        TurnLeft = options["Left"]
        TurnRight = options["Right"]

        nsrts = set()

        # MoveUp
        # Agent, from_loc, to_loc
        parameters = [obj1, obj2, obj3]
        preconditions = {
            LiftedAtom(IsAgent, [obj1]),
            LiftedAtom(IsLoc, [obj3]),
            LiftedAtom(IsLoc, [obj2]),
            LiftedAtom(Above, [obj3, obj2]),
            LiftedAtom(At, [obj1, obj2]),
            LiftedAtom(IsFacingUp, [obj1]),
        }
        add_effects = {LiftedAtom(At, [obj1, obj3])}
        delete_effects = {LiftedAtom(At, [obj1, obj2])}
        option = MoveForward
        option_vars: List[Variable] = []  # dummy - not used
        move_up_nsrt = NSRT("MoveUp", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            null_sampler)
        nsrts.add(move_up_nsrt)

        # MoveDown
        # Agent, from_loc, to_loc
        parameters = [obj1, obj2, obj3]
        preconditions = {
            LiftedAtom(IsAgent, [obj1]),
            LiftedAtom(IsLoc, [obj3]),
            LiftedAtom(IsLoc, [obj2]),
            LiftedAtom(Below, [obj3, obj2]),
            LiftedAtom(At, [obj1, obj2]),
            LiftedAtom(IsFacingDown, [obj1]),
        }
        add_effects = {LiftedAtom(At, [obj1, obj3])}
        delete_effects = {LiftedAtom(At, [obj1, obj2])}
        option = MoveForward
        option_vars = []  # dummy - not used
        move_down_nsrt = NSRT("MoveDown", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(move_down_nsrt)

        # MoveRight
        # Agent, from_loc, to_loc
        parameters = [obj1, obj2, obj3]
        preconditions = {
            LiftedAtom(IsAgent, [obj1]),
            LiftedAtom(IsLoc, [obj3]),
            LiftedAtom(IsLoc, [obj2]),
            LiftedAtom(RightOf, [obj3, obj2]),
            LiftedAtom(At, [obj1, obj2]),
            LiftedAtom(IsFacingRight, [obj1]),
        }
        add_effects = {LiftedAtom(At, [obj1, obj3])}
        delete_effects = {LiftedAtom(At, [obj1, obj2])}
        option = MoveForward
        option_vars = []  # dummy - not used
        move_right_nsrt = NSRT("MoveRight", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, null_sampler)
        nsrts.add(move_right_nsrt)

        # MoveLeft
        # Agent, from_loc, to_loc
        parameters = [obj1, obj2, obj3]
        preconditions = {
            LiftedAtom(IsAgent, [obj1]),
            LiftedAtom(IsLoc, [obj3]),
            LiftedAtom(IsLoc, [obj2]),
            LiftedAtom(LeftOf, [obj3, obj2]),
            LiftedAtom(At, [obj1, obj2]),
            LiftedAtom(IsFacingLeft, [obj1]),
        }
        add_effects = {LiftedAtom(At, [obj1, obj3])}
        delete_effects = {LiftedAtom(At, [obj1, obj2])}
        option = MoveForward
        option_vars = []  # dummy - not used
        move_left_nsrt = NSRT("MoveLeft", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(move_left_nsrt)

        # TurnRight
        turn_right_from_up_nsrt = NSRT("TurnRightFromUp", [obj1],
                               {LiftedAtom(IsFacingUp, [obj1])},
                               {LiftedAtom(IsFacingRight, [obj1])},
                               {LiftedAtom(IsFacingUp, [obj1])},
                                set(),
                                TurnRight, [], null_sampler)
        nsrts.add(turn_right_from_up_nsrt)

        turn_right_from_down_nsrt = NSRT("TurnRightFromDown", [obj1],
                               {LiftedAtom(IsFacingDown, [obj1])},
                               {LiftedAtom(IsFacingLeft, [obj1])},
                               {LiftedAtom(IsFacingDown, [obj1])},
                                set(),
                                TurnRight, [], null_sampler)
        nsrts.add(turn_right_from_down_nsrt)

        turn_right_from_left_nsrt = NSRT("TurnRightFromLeft", [obj1],
                               {LiftedAtom(IsFacingLeft, [obj1])},
                               {LiftedAtom(IsFacingUp, [obj1])},
                               {LiftedAtom(IsFacingLeft, [obj1])},
                                set(),
                                TurnRight, [], null_sampler)
        nsrts.add(turn_right_from_left_nsrt)

        turn_right_from_right_nsrt = NSRT("TurnRightFromRight", [obj1],
                                 {LiftedAtom(IsFacingRight, [obj1])},
                                 {LiftedAtom(IsFacingDown, [obj1])},
                                 {LiftedAtom(IsFacingRight, [obj1])},
                                  set(),
                                  TurnRight, [], null_sampler)
        nsrts.add(turn_right_from_right_nsrt)

        # TurnLeft
        turn_left_from_up_nsrt = NSRT("TurnLeftFromUp", [obj1],
                               {LiftedAtom(IsFacingUp, [obj1])},
                               {LiftedAtom(IsFacingLeft, [obj1])},
                               {LiftedAtom(IsFacingUp, [obj1])},
                                set(),
                                TurnLeft, [], null_sampler)
        nsrts.add(turn_left_from_up_nsrt)

        turn_left_from_down_nsrt = NSRT("TurnLeftFromDown", [obj1],
                                 {LiftedAtom(IsFacingDown, [obj1])},
                                 {LiftedAtom(IsFacingRight, [obj1])},
                                 {LiftedAtom(IsFacingDown, [obj1])},
                                  set(),
                                  TurnLeft, [], null_sampler)
        nsrts.add(turn_left_from_down_nsrt)

        turn_left_from_left_nsrt = NSRT("TurnLeftFromLeft", [obj1],
                                    {LiftedAtom(IsFacingLeft, [obj1])},
                                    {LiftedAtom(IsFacingDown, [obj1])},
                                    {LiftedAtom(IsFacingLeft, [obj1])},
                                    set(),
                                    TurnLeft, [], null_sampler)
        nsrts.add(turn_left_from_left_nsrt)

        turn_left_from_right_nsrt = NSRT("TurnLeftFromRight", [obj1],
                                    {LiftedAtom(IsFacingRight, [obj1])},
                                    {LiftedAtom(IsFacingUp, [obj1])},
                                    {LiftedAtom(IsFacingRight, [obj1])},
                                    set(),
                                    TurnLeft, [], null_sampler)
        nsrts.add(turn_left_from_right_nsrt)

        return nsrts
