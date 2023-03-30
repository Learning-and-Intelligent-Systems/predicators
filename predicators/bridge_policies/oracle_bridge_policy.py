"""A hand-written LDL bridge policy."""

import logging
from typing import Callable, Dict, List, Set

import numpy as np

from predicators import utils
from predicators.bridge_policies import BridgePolicyDone
from predicators.bridge_policies.ldl_bridge_policy import LDLBridgePolicy, \
    get_failure_predicate
from predicators.settings import CFG
from predicators.structs import NSRT, BridgePolicy, GroundAtom, LDLRule, \
    LiftedAtom, LiftedDecisionList, ParameterizedOption, Predicate, State, \
    Variable, _Option


class OracleBridgePolicy(LDLBridgePolicy):
    """A hand-written LDL bridge policy."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], nsrts: Set[NSRT]) -> None:
        super().__init__(predicates, options, nsrts)
        self._oracle_ldl = _create_oracle_ldl_bridge_policy(
            CFG.env, self._nsrts, self._options, self._predicates, self._rng)

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    def _get_current_ldl(self) -> LiftedDecisionList:
        return self._oracle_ldl


def _create_oracle_ldl_bridge_policy(
        env_name: str, nsrts: Set[NSRT], options: Set[ParameterizedOption],
        predicates: Set[Predicate],
        rng: np.random.Generator) -> LiftedDecisionList:
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    option_name_to_option = {o.name: o for o in options}
    pred_name_to_pred = {p.name: p for p in predicates}

    if env_name == "painting":
        return _create_painting_oracle_ldl_bridge_policy(
            nsrt_name_to_nsrt, option_name_to_option, pred_name_to_pred, rng)

    if env_name == "stick_button":
        return _create_stick_button_oracle_ldl_bridge_policy(
            nsrt_name_to_nsrt, option_name_to_option, pred_name_to_pred, rng)

    raise NotImplementedError(f"No oracle bridge policy for {env_name}")


def _create_painting_oracle_ldl_bridge_policy(
        nsrt_name_to_nsrt: Dict[str, NSRT],
        option_name_to_option: Dict[str, ParameterizedOption],
        pred_name_to_pred: Dict[str, Predicate],
        rng: np.random.Generator) -> LiftedDecisionList:

    PlaceOnTable = nsrt_name_to_nsrt["PlaceOnTable"]
    OpenLid = nsrt_name_to_nsrt["OpenLid"]

    Place = option_name_to_option["Place"]

    Holding = pred_name_to_pred["Holding"]
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
        pred_name_to_pred: Dict[str, Predicate],
        rng: np.random.Generator) -> LiftedDecisionList:

    PickStickFromNothing = nsrt_name_to_nsrt["PickStickFromNothing"]
    PickStickFromButton = nsrt_name_to_nsrt["PickStickFromButton"]
    RobotPressButtonFromNothing = nsrt_name_to_nsrt[
        "RobotPressButtonFromNothing"]
    RobotPressButtonFromButton = nsrt_name_to_nsrt[
        "RobotPressButtonFromButton"]
    StickPressButtonFromNothing = nsrt_name_to_nsrt[
        "StickPressButtonFromNothing"]
    StickPressButtonFromButton = nsrt_name_to_nsrt[
        "StickPressButtonFromButton"]
    PlaceStick = nsrt_name_to_nsrt["PlaceStick"]

    RobotPressButton = option_name_to_option["RobotPressButton"]
    StickPressButton = option_name_to_option["StickPressButton"]

    Pressed = pred_name_to_pred["Pressed"]
    button_type = Pressed.types[0]

    # "Failure" predicates.
    RobotPressFailedButton = get_failure_predicate(RobotPressButton, (1, ))
    StickPressFailedButton = get_failure_predicate(StickPressButton, (2, ))

    bridge_rules = []
    goal_preconds: Set[LiftedAtom] = set()

    # We haven't tried to press the button yet, and we're holding a stick, so
    # we should put it down.
    name = "PlaceStickBeforeDirectPress"
    nsrt = PlaceStick
    robot, stick = nsrt.parameters
    button = Variable("?button", button_type)
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions)
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(RobotPressFailedButton, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    # We haven't tried to press the button yet, and the hand is empty, so we
    # should go to try the direct press.
    name = "DirectPressFromNothing"
    nsrt = RobotPressButtonFromNothing
    robot, button = nsrt.parameters
    parameters = [robot, button]
    pos_preconds = set(nsrt.preconditions)
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(RobotPressFailedButton, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    name = "DirectPressFromButton"
    nsrt = RobotPressButtonFromButton
    robot, button, from_button = nsrt.parameters
    parameters = [robot, button, from_button]
    pos_preconds = set(nsrt.preconditions)
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(RobotPressFailedButton, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    # We failed to press directly, but haven't yet tried to press with the
    # stick, and the hand is empty, so we should pick up the stick.
    name = "PickStickFromNothingBeforePress"
    nsrt = PickStickFromNothing
    robot, stick = nsrt.parameters
    button = Variable("?button", button_type)
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(StickPressFailedButton, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    name = "PickStickFromButtonBeforePress"
    nsrt = PickStickFromButton
    robot, stick, from_button = nsrt.parameters
    button = Variable("?to-button", button_type)
    parameters = [robot, stick, button, from_button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(StickPressFailedButton, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    # We failed to press directly, but haven't yet tried to press with the
    # stick, and we're holding the stick, so we should try to press.
    name = "PressWithStickFromNothing"
    nsrt = StickPressButtonFromNothing
    robot, stick, button = nsrt.parameters
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(StickPressFailedButton, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    name = "PressWithStickFromButton"
    nsrt = StickPressButtonFromButton
    robot, stick, button, from_button = nsrt.parameters
    parameters = [robot, stick, button, from_button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(StickPressFailedButton, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    # We've already tried to press both ways, and we're holding the stick, so
    # we should put it down in preparation for a regrasp.
    name = "PlaceToRegrasp"
    nsrt = PlaceStick
    robot, stick = nsrt.parameters
    button = Variable("?button", button_type)
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressFailedButton, [button]),
        LiftedAtom(StickPressFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    # We've already tried to press both ways, and the hand is empty, so we
    # should go to pick up the stick.
    name = "RegraspFromNothing"
    nsrt = PickStickFromNothing
    robot, stick = nsrt.parameters
    button = Variable("?button", button_type)
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressFailedButton, [button]),
        LiftedAtom(StickPressFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    name = "RegraspFromButton"
    nsrt = PickStickFromButton
    robot, stick, from_button = nsrt.parameters
    button = Variable("?to-button", button_type)
    parameters = [robot, stick, button, from_button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressFailedButton, [button]),
        LiftedAtom(StickPressFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
    }
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds,
                   nsrt)
    bridge_rules.append(rule)

    bridge_ldl = LiftedDecisionList(bridge_rules)

    return bridge_ldl
