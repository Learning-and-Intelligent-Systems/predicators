"""A hand-written LDL bridge policy."""

from typing import Dict, Set

from predicators import utils
from predicators.bridge_policies.ldl_bridge_policy import LDLBridgePolicy, \
    get_failure_predicate
from predicators.settings import CFG
from predicators.structs import NSRT, LDLRule, LiftedAtom, \
    LiftedDecisionList, ParameterizedOption, Predicate, Variable


class OracleBridgePolicy(LDLBridgePolicy):
    """A hand-written LDL bridge policy."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], nsrts: Set[NSRT]) -> None:
        super().__init__(predicates, options, nsrts)
        self._oracle_ldl = _create_oracle_ldl_bridge_policy(
            CFG.env, self._nsrts, self._options, self._predicates)

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_ldl(self) -> LiftedDecisionList:
        return self._oracle_ldl


def _create_oracle_ldl_bridge_policy(
        env_name: str, nsrts: Set[NSRT], options: Set[ParameterizedOption],
        predicates: Set[Predicate]) -> LiftedDecisionList:
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    option_name_to_option = {o.name: o for o in options}
    pred_name_to_pred = {p.name: p for p in predicates}

    if env_name == "painting":
        return _create_painting_oracle_ldl_bridge_policy(
            nsrt_name_to_nsrt, option_name_to_option, pred_name_to_pred)

    if env_name == "stick_button":
        return _create_stick_button_oracle_ldl_bridge_policy(
            nsrt_name_to_nsrt, option_name_to_option, pred_name_to_pred)

    raise NotImplementedError(f"No oracle bridge policy for {env_name}")


def _create_painting_oracle_ldl_bridge_policy(
        nsrt_name_to_nsrt: Dict[str, NSRT],
        option_name_to_option: Dict[str, ParameterizedOption],
        pred_name_to_pred: Dict[str, Predicate]) -> LiftedDecisionList:

    PlaceOnTable = nsrt_name_to_nsrt["PlaceOnTable"]
    OpenLid = nsrt_name_to_nsrt["OpenLid"]

    Place = option_name_to_option["Place"]

    IsOpen = pred_name_to_pred["IsOpen"]
    lid_type, = IsOpen.types

    # "Failure" predicates.
    PlaceFailed = get_failure_predicate(Place, (0, ))

    bridge_rules = []
    goal_preconds: Set[LiftedAtom] = set()

    # Place the held object on the table.
    name = "PlaceHeldObjectOnTable"
    nsrt = PlaceOnTable
    held_obj, robot = nsrt.parameters
    lid = Variable("?lid", lid_type)
    parameters = [held_obj, robot, lid]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(PlaceFailed, [robot]),
    }
    neg_preconds = {LiftedAtom(IsOpen, [lid])}
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    # Open the box lid.
    name = "OpenLid"
    nsrt = OpenLid
    lid, robot = nsrt.parameters
    parameters = [lid, robot]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(PlaceFailed, [robot]),
    }
    neg_preconds = {LiftedAtom(IsOpen, [lid])}
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    bridge_ldl = LiftedDecisionList(bridge_rules)

    return bridge_ldl


def _create_stick_button_oracle_ldl_bridge_policy(
        nsrt_name_to_nsrt: Dict[str, NSRT],
        option_name_to_option: Dict[str, ParameterizedOption],
        pred_name_to_pred: Dict[str, Predicate]) -> LiftedDecisionList:

    PickStickFromButton = nsrt_name_to_nsrt["PickStickFromButton"]
    robot_type, stick_type, button_type = [p.type for p in PickStickFromButton.parameters]
    types = {button_type, stick_type, robot_type}

    # "Failure" predicates.
    RobotPressButton = option_name_to_option["RobotPressButton"]
    StickPressButton = option_name_to_option["StickPressButton"]
    PlaceStick = option_name_to_option["PlaceStick"]
    RobotPressFailedRobot = get_failure_predicate(RobotPressButton, (0, ))
    RobotPressFailedButton = get_failure_predicate(RobotPressButton, (1, ))
    StickPressFailedRobot = get_failure_predicate(StickPressButton, (0, ))
    StickPressFailedStick = get_failure_predicate(StickPressButton, (1, ))
    StickPressFailedButton = get_failure_predicate(StickPressButton, (2, ))
    PlaceStickFailedRobot = get_failure_predicate(PlaceStick, (0, ))
    PlaceStickFailedStick = get_failure_predicate(PlaceStick, (1, ))
    failure_preds = {RobotPressFailedRobot, RobotPressFailedButton, StickPressFailedButton, StickPressFailedStick, StickPressFailedRobot, PlaceStickFailedRobot, PlaceStickFailedStick}
    predicates = set(pred_name_to_pred.values()) | failure_preds
    nsrts = set(nsrt_name_to_nsrt.values())

    ldl_str = """
    (define (policy)
  (:rule PickStickFromButton
    :parameters (?robot - robot ?stick - stick ?from_button - button)
    :preconditions (and (HandEmpty ?robot) (RobotAboveButton ?robot ?from_button) (RobotPressButtonFailed_arg0 ?robot) (not (AboveNoButton )) (not (Grasped ?robot ?stick)) (not (PlaceStickFailed_arg0 ?robot)) (not (PlaceStickFailed_arg1 ?stick)) (not (RobotPressButtonFailed_arg1 ?from_button)))
    :goals ()
    :action (PickStickFromButton ?robot ?stick ?from-button)
  )
  (:rule PickStickFromNothing
    :parameters (?robot - robot ?stick - stick)
    :preconditions (and (AboveNoButton ) (HandEmpty ?robot) (RobotPressButtonFailed_arg0 ?robot) (not (Grasped ?robot ?stick)))
    :goals ()
    :action (PickStickFromNothing ?robot ?stick)
  )
  (:rule PlaceStick2
    :parameters (?robot - robot ?stick - stick)
    :preconditions (and (Grasped ?robot ?stick) (RobotPressButtonFailed_arg0 ?robot) (StickPressButtonFailed_arg0 ?robot) (StickPressButtonFailed_arg1 ?stick) (not (AboveNoButton )) (not (HandEmpty ?robot)))
    :goals ()
    :action (PlaceStick2 ?robot ?stick)
  )
  (:rule PlaceStick
    :parameters (?robot - robot ?stick - stick)
    :preconditions (and (AboveNoButton ) (Grasped ?robot ?stick) (RobotPressButtonFailed_arg0 ?robot) (StickPressButtonFailed_arg0 ?robot) (StickPressButtonFailed_arg1 ?stick) (not (HandEmpty ?robot)))
    :goals ()
    :action (PlaceStick ?robot ?stick)
  )
)
"""

    bridge_ldl = utils.parse_ldl_from_str(ldl_str, types,
                       predicates,
                       nsrts)

    return bridge_ldl
