"""Ground-truth Operators for the Kitchen environment."""

from typing import Dict, Set

from predicators.structs import LiftedAtom, Predicate, STRIPSOperator, Type, \
    Variable


class KitchenGroundTruthOperatorFactory():
    """Ground-truth Operators for the Kitchen environment."""

    @staticmethod
    def get_operators(env_name: str, types: Dict[str, Type],
                      predicates: Dict[str, Predicate]) -> Set[STRIPSOperator]:
        """Creates Operators for Mujoco Kitchen Env."""
        assert env_name == "kitchen"
        # Types
        gripper_type = types["gripper"]
        object_type = types["obj"]

        # Objects
        gripper = Variable("?gripper", gripper_type)
        obj = Variable("?obj", object_type)
        obj2 = Variable("?obj2", object_type)

        # Predicates
        At = predicates["At"]
        TurnedOn = predicates["TurnedOn"]
        OnTop = predicates["OnTop"]
        CanTurnDial = predicates["CanTurnDial"]

        operators = set()

        # MoveTo
        parameters = [gripper, obj]
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(At, [gripper, obj])}
        delete_effects: Set[LiftedAtom] = set()
        move_to_operator = STRIPSOperator("MoveTo", parameters, preconditions,
                                          add_effects, delete_effects, {At})
        operators.add(move_to_operator)

        # PushObjOnObjForward
        parameters = [gripper, obj, obj2]
        preconditions = {LiftedAtom(At, [gripper, obj])}
        add_effects = {LiftedAtom(OnTop, [obj, obj2])}
        delete_effects = {LiftedAtom(CanTurnDial, [gripper])}
        push_obj_on_obj_forward_operator = STRIPSOperator(
            "PushObjOnObjForward", parameters, preconditions, add_effects,
            delete_effects, {OnTop})
        operators.add(push_obj_on_obj_forward_operator)

        # PushObjTurnOnRight
        parameters = [gripper, obj]
        preconditions = {
            LiftedAtom(At, [gripper, obj]),
            LiftedAtom(CanTurnDial, [gripper])
        }
        add_effects = {LiftedAtom(TurnedOn, [obj])}
        delete_effects = set()
        push_obj_turn_on_right_operator = STRIPSOperator(
            "PushObjTurnOnRight", parameters, preconditions, add_effects,
            delete_effects, {TurnedOn})
        operators.add(push_obj_turn_on_right_operator)

        return operators
