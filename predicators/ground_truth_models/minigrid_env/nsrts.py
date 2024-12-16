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
        location_type = types["loc"]

        # Objects
        obj1 = Variable("?obj1", location_type)
        obj2 = Variable("?obj2", location_type)
        obj = Variable("?obj3", object_type)
        obj3 = Variable("?obj4", object_type)

        # Predicates
        At = predicates["At"]
        Above = predicates["Above"]
        Below = predicates["Below"]
        RightOf = predicates["RightOf"]
        LeftOf = predicates["LeftOf"]
        ObjAbove = predicates["ObjAbove"]
        ObjBelow = predicates["ObjBelow"]
        ObjRightOf = predicates["ObjRightOf"]
        ObjLeftOf = predicates["ObjLeftOf"]
        AgentAt = predicates["AgentAt"]
        IsGoal = predicates["IsGoal"]
        IsFacingUp = predicates["IsFacingUp"]
        IsFacingDown = predicates["IsFacingDown"]
        IsFacingLeft = predicates["IsFacingLeft"]
        IsFacingRight = predicates["IsFacingRight"]
        Unknown = predicates["Unknown"]
        Found = predicates["Found"]
        ObjUnknown = predicates["ObjUnknown"]
        ObjFound = predicates["ObjFound"]
        Holding = predicates["Holding"]
        Near = predicates["Near"]

        # Options
        MoveForward = options["Forward"]
        TurnLeft = options["Left"]
        TurnRight = options["Right"]
        Pickup = options["Pickup"]
        Drop = options["Drop"]
        Toggle = options["Toggle"]
        Done = options["Done"]
        FindObj = options["FindObj"]
        ReplanToObj = options["ReplanToObj"]

        nsrts = set()

        # MoveUp
        # from_loc, to_loc
        parameters = [obj1, obj2]
        preconditions = {
            LiftedAtom(Above, [obj2, obj1]),
            LiftedAtom(AgentAt, [obj1]),
            LiftedAtom(IsFacingUp, []),
        }
        add_effects = {LiftedAtom(AgentAt, [obj2])}
        delete_effects = {LiftedAtom(AgentAt, [obj1])}
        option = MoveForward
        option_vars: List[Variable] = []  # dummy - not used
        move_up_nsrt = NSRT("MoveUp", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            null_sampler)
        nsrts.add(move_up_nsrt)

        # MoveDown
        # from_loc, to_loc
        parameters = [obj1, obj2]
        preconditions = {
            LiftedAtom(Below, [obj2, obj1]),
            LiftedAtom(AgentAt, [obj1]),
            LiftedAtom(IsFacingDown, []),
        }
        add_effects = {LiftedAtom(AgentAt, [obj2])}
        delete_effects = {LiftedAtom(AgentAt, [obj1])}
        option = MoveForward
        option_vars = []  # dummy - not used
        move_down_nsrt = NSRT("MoveDown", parameters, preconditions,
                                add_effects, delete_effects, set(), option,
                                option_vars, null_sampler)
        nsrts.add(move_down_nsrt)

        # MoveRight
        # from_loc, to_loc
        parameters = [obj1, obj2]
        preconditions = {
            LiftedAtom(RightOf, [obj2, obj1]),
            LiftedAtom(AgentAt, [obj1]),
            LiftedAtom(IsFacingRight, []),
        }
        add_effects = {LiftedAtom(AgentAt, [obj2])}
        delete_effects = {LiftedAtom(AgentAt, [obj1])}
        option = MoveForward
        option_vars = []  # dummy - not used
        move_right_nsrt = NSRT("MoveRight", parameters, preconditions,
                                add_effects, delete_effects, set(), option,
                                option_vars, null_sampler)
        nsrts.add(move_right_nsrt)

        # MoveLeft
        # from_loc, to_loc
        parameters = [obj1, obj2]
        preconditions = {
            LiftedAtom(LeftOf, [obj2, obj1]),
            LiftedAtom(AgentAt, [obj1]),
            LiftedAtom(IsFacingLeft, []),
        }
        add_effects = {LiftedAtom(AgentAt, [obj2])}
        delete_effects = {LiftedAtom(AgentAt, [obj1])}
        option = MoveForward
        option_vars = []  # dummy - not used
        move_left_nsrt = NSRT("MoveLeft", parameters, preconditions,
                                add_effects, delete_effects, set(), option,
                                option_vars, null_sampler)
        nsrts.add(move_left_nsrt)

        # TurnRight
        turn_right_from_up_nsrt = NSRT("TurnRightFromUp", [],
                               {LiftedAtom(IsFacingUp, [])},
                               {LiftedAtom(IsFacingRight, [])},
                               {LiftedAtom(IsFacingUp, [])},
                                set(),
                                TurnRight, [], null_sampler)
        nsrts.add(turn_right_from_up_nsrt)

        turn_right_from_down_nsrt = NSRT("TurnRightFromDown", [],
                               {LiftedAtom(IsFacingDown, [])},
                               {LiftedAtom(IsFacingLeft, [])},
                               {LiftedAtom(IsFacingDown, [])},
                                set(),
                                TurnRight, [], null_sampler)
        nsrts.add(turn_right_from_down_nsrt)

        turn_right_from_left_nsrt = NSRT("TurnRightFromLeft", [],
                               {LiftedAtom(IsFacingLeft, [])},
                               {LiftedAtom(IsFacingUp, [])},
                               {LiftedAtom(IsFacingLeft, [])},
                                set(),
                                TurnRight, [], null_sampler)
        nsrts.add(turn_right_from_left_nsrt)

        turn_right_from_right_nsrt = NSRT("TurnRightFromRight", [],
                                 {LiftedAtom(IsFacingRight, [])},
                                 {LiftedAtom(IsFacingDown, [])},
                                 {LiftedAtom(IsFacingRight, [])},
                                  set(),
                                  TurnRight, [], null_sampler)
        nsrts.add(turn_right_from_right_nsrt)

        # TurnLeft
        turn_left_from_up_nsrt = NSRT("TurnLeftFromUp", [],
                               {LiftedAtom(IsFacingUp, [])},
                               {LiftedAtom(IsFacingLeft, [])},
                               {LiftedAtom(IsFacingUp, [])},
                                set(),
                                TurnLeft, [], null_sampler)
        nsrts.add(turn_left_from_up_nsrt)

        turn_left_from_down_nsrt = NSRT("TurnLeftFromDown", [],
                                 {LiftedAtom(IsFacingDown, [])},
                                 {LiftedAtom(IsFacingRight, [])},
                                 {LiftedAtom(IsFacingDown, [])},
                                  set(),
                                  TurnLeft, [], null_sampler)
        nsrts.add(turn_left_from_down_nsrt)

        turn_left_from_left_nsrt = NSRT("TurnLeftFromLeft", [],
                                    {LiftedAtom(IsFacingLeft, [])},
                                    {LiftedAtom(IsFacingDown, [])},
                                    {LiftedAtom(IsFacingLeft, [])},
                                    set(),
                                    TurnLeft, [], null_sampler)
        nsrts.add(turn_left_from_left_nsrt)

        turn_left_from_right_nsrt = NSRT("TurnLeftFromRight", [],
                                    {LiftedAtom(IsFacingRight, [])},
                                    {LiftedAtom(IsFacingUp, [])},
                                    {LiftedAtom(IsFacingRight, [])},
                                    set(),
                                    TurnLeft, [], null_sampler)
        nsrts.add(turn_left_from_right_nsrt)

        # Pickup Left
        # obj, agent_loc
        parameters = [obj, obj2]
        preconditions = {
            LiftedAtom(AgentAt, [obj2]),
            LiftedAtom(ObjLeftOf, [obj, obj2]),
            LiftedAtom(IsFacingLeft, []),
            LiftedAtom(ObjFound, [obj])
        }
        add_effects = {LiftedAtom(Holding, [obj])}
        delete_effects = {LiftedAtom(ObjLeftOf, [obj, obj2])}
        option = Pickup
        option_vars: List[Variable] = []
        pickup_left_nsrt = NSRT("Pickup_Left", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(pickup_left_nsrt)

        # Pickup Right
        # obj, agent_loc
        parameters = [obj, obj2]
        preconditions = {
            LiftedAtom(AgentAt, [obj2]),
            LiftedAtom(ObjRightOf, [obj, obj2]),
            LiftedAtom(IsFacingRight, []),
            LiftedAtom(ObjFound, [obj])
        }
        add_effects = {LiftedAtom(Holding, [obj])}
        delete_effects = {LiftedAtom(ObjRightOf, [obj, obj2])}
        option = Pickup
        option_vars: List[Variable] = []
        pickup_right_nsrt = NSRT("Pickup_Right", parameters, preconditions,
                                add_effects, delete_effects, set(), option,
                                option_vars, null_sampler)
        nsrts.add(pickup_right_nsrt)

        # Pickup Up
        # obj, agent_loc
        parameters = [obj, obj2]
        preconditions = {
            LiftedAtom(AgentAt, [obj2]),
            LiftedAtom(ObjAbove, [obj, obj2]),
            LiftedAtom(IsFacingUp, []),
            LiftedAtom(ObjFound, [obj])
        }
        add_effects = {LiftedAtom(Holding, [obj])}
        delete_effects = {LiftedAtom(ObjAbove, [obj, obj2])}
        option = Pickup
        option_vars: List[Variable] = []
        pickup_up_nsrt = NSRT("Pickup_Up", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(pickup_up_nsrt)

        # Pickup Down
        # obj, agent_loc
        parameters = [obj, obj2]
        preconditions = {
            LiftedAtom(AgentAt, [obj2]),
            LiftedAtom(ObjBelow, [obj, obj2]),
            LiftedAtom(IsFacingDown, []),
            LiftedAtom(ObjFound, [obj])
        }
        add_effects = {LiftedAtom(Holding, [obj])}
        delete_effects = {LiftedAtom(ObjBelow, [obj, obj2])}
        option = Pickup
        option_vars: List[Variable] = []
        pickup_down_nsrt = NSRT("Pickup_Down", parameters, preconditions,
                                add_effects, delete_effects, set(), option,
                                option_vars, null_sampler)
        nsrts.add(pickup_down_nsrt)

        # Drop
        # TODO

        # Toggle
        # TODO

        # For Partial Observability
        # Find Object
        find_obj_nsrt = NSRT("FindObj", [obj],
                                    {LiftedAtom(ObjUnknown, [obj])},
                                    {LiftedAtom(ObjFound, [obj])},
                                    set(),
                                    {LeftOf, RightOf, Above, Below, ObjAbove, ObjBelow, ObjRightOf, ObjLeftOf},
                                    FindObj, [obj], null_sampler)
        nsrts.add(find_obj_nsrt)

        # Find Location
        find_loc_nsrt = NSRT("FindLoc", [obj1],
                                {LiftedAtom(Unknown, [obj1])},
                                {LiftedAtom(Found, [obj1])},
                                set(),
                                {LeftOf, RightOf, Above, Below, ObjAbove, ObjBelow, ObjRightOf, ObjLeftOf},
                                FindObj, [obj1], null_sampler)

        # Replan to location
        replan_to_loc_nsrt = NSRT("ReplanToLoc", [obj1],
                                {LiftedAtom(Unknown, [obj1]), LiftedAtom(Found, [obj1])},
                                {LiftedAtom(AgentAt, [obj1])},
                                set(),
                                {LeftOf, RightOf, Above, Below, ObjAbove, ObjBelow, ObjRightOf, ObjLeftOf},
                                ReplanToObj, [], null_sampler)
        nsrts.add(replan_to_loc_nsrt)

        # Replan With Obj Known
        replan_to_obj_nsrt = NSRT("ReplanToObj", [obj, obj1],
                                {LiftedAtom(ObjUnknown, [obj]), LiftedAtom(ObjFound, [obj])},
                                {LiftedAtom(At, [obj, obj1])},
                                set(),
                                {LeftOf, RightOf, Above, Below, ObjAbove, ObjBelow, ObjRightOf, ObjLeftOf},
                                ReplanToObj, [], null_sampler)
        nsrts.add(replan_to_obj_nsrt)

        replan_to_pickable_obj_nsrt = NSRT("ReplanToPickableObj", [obj],
                                {LiftedAtom(ObjUnknown, [obj]), LiftedAtom(ObjFound, [obj])},
                                {LiftedAtom(Holding, [obj])},
                                set(),
                                {LeftOf, RightOf, Above, Below, ObjAbove, ObjBelow, ObjRightOf, ObjLeftOf},
                                ReplanToObj, [], null_sampler)
        nsrts.add(replan_to_pickable_obj_nsrt)

        return nsrts
