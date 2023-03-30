"""A hand-written bridge policy."""

import logging
from typing import Callable, Dict, List, Set

import numpy as np

from predicators import utils
from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.settings import CFG
from predicators.structs import NSRT, BridgePolicy, GroundAtom, Predicate, \
    State, _Option, Variable, LiftedAtom, LDLRule, LiftedDecisionList


class OracleBridgePolicy(BaseBridgePolicy):
    """A hand-written bridge policy."""

    def __init__(self, predicates: Set[Predicate], nsrts: Set[NSRT]) -> None:
        super().__init__(predicates, nsrts)
        self._oracle_bridge_policy = _create_oracle_bridge_policy(
            CFG.env, self._nsrts, self._predicates, self._rng)

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    def get_option_policy(self) -> Callable[[State], _Option]:

        def _option_policy(state: State) -> _Option:
            atoms = utils.abstract(state, self._predicates)
            option = self._oracle_bridge_policy(state, atoms,
                                                self._failed_options)
            logging.debug(f"Using option {option.name}{option.objects} "
                          "from bridge policy.")
            return option

        return _option_policy


def _create_oracle_bridge_policy(env_name: str, nsrts: Set[NSRT],
                                 predicates: Set[Predicate],
                                 rng: np.random.Generator) -> BridgePolicy:
    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    pred_name_to_pred = {p.name: p for p in predicates}

    if env_name == "painting":
        return _create_painting_oracle_bridge_policy(nsrt_name_to_nsrt,
                                                     pred_name_to_pred, rng)

    if env_name == "stick_button":
        return _create_stick_button_oracle_bridge_policy(
            nsrt_name_to_nsrt, pred_name_to_pred, rng)

    raise NotImplementedError(f"No oracle bridge policy for {env_name}")


def _create_painting_oracle_bridge_policy(
        nsrt_name_to_nsrt: Dict[str, NSRT], pred_name_to_pred: Dict[str,
                                                                    Predicate],
        rng: np.random.Generator) -> BridgePolicy:

    PlaceOnTable = nsrt_name_to_nsrt["PlaceOnTable"]
    OpenLid = nsrt_name_to_nsrt["OpenLid"]

    Holding = pred_name_to_pred["Holding"]

    def _bridge_policy(state: State, atoms: Set[GroundAtom],
                       failed_options: List[_Option]) -> _Option:

        last_failed = failed_options[-1]
        lid = next(o for o in state if o.type.name == "lid")

        # If the box lid is already open, the bridge policy is done.
        # Second case should only happen when the shelf placements fail.
        if state.get(lid, "is_open") > 0.5 or last_failed.name != "Place":
            raise BridgePolicyDone()

        robot = last_failed.objects[0]
        held_objs = {a.objects[0] for a in atoms if a.predicate == Holding}

        if not held_objs:
            next_nsrt = OpenLid.ground([lid, robot])
        else:
            held_obj = next(iter(held_objs))
            next_nsrt = PlaceOnTable.ground([held_obj, robot])

        goal: Set[GroundAtom] = set()  # goal assumed not used by sampler
        return next_nsrt.sample_option(state, goal, rng)

    return _bridge_policy


def _create_stick_button_oracle_bridge_policy(
        nsrt_name_to_nsrt: Dict[str, NSRT], pred_name_to_pred: Dict[str,
                                                                    Predicate],
        rng: np.random.Generator) -> BridgePolicy:

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

    Grasped = pred_name_to_pred["Grasped"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    Pressed = pred_name_to_pred["Pressed"]
    button_type = Pressed.types[0]
    robot_type, stick_type = Grasped.types

    # "Failure" predicates.
    all_options = {n.option for n in nsrt_name_to_nsrt.values()}
    failure_preds: Dict[str, Predicate] = {}
    for option in all_options:
        # Just unary for now.
        for idx, t in enumerate(option.types):
            failure_pred = Predicate(f"{option.name}Failed-arg{idx}", [t],
                _classifier=lambda s, o: False)
            failure_preds[failure_pred.name] = failure_pred

    RobotPressButtonFailedButton = failure_preds["RobotPressButtonFailed-arg1"]
    StickPressButtonFailedButton = failure_preds["StickPressButtonFailed-arg2"]

    bridge_rules = []

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
        LiftedAtom(RobotPressButtonFailedButton, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
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
        LiftedAtom(RobotPressButtonFailedButton, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    name = "DirectPressFromButton"
    nsrt = RobotPressButtonFromButton
    robot, button, from_button = nsrt.parameters
    parameters = [robot, button, from_button]
    pos_preconds = set(nsrt.preconditions)
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(RobotPressButtonFailedButton, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    # We failed to press directly, but haven't yet tried to press with the
    # stick, and the hand is empty, so we should pick up the stick.
    name = "PickStickFromNothingBeforePress"
    nsrt = PickStickFromNothing
    robot, stick = nsrt.parameters
    button = Variable("?button", button_type)
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    name = "PickStickFromButtonkBeforePress"
    nsrt = PickStickFromButton
    robot, stick, from_button = nsrt.parameters
    button = Variable("?to-button", button_type)
    parameters = [robot, stick, button, from_button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    # We failed to press directly, but haven't yet tried to press with the
    # stick, and we're holding the stick, so we should try to press.
    name = "PressWithStickFromNothing"
    nsrt = StickPressButtonFromNothing
    robot, stick, button = nsrt.parameters
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    name = "PressWithStickFromButton"
    nsrt = StickPressButtonFromButton
    robot, stick, button, from_button = nsrt.parameters
    parameters = [robot, stick, button, from_button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    # We've already tried to press both ways, and we're holding the stick, so
    # we should put it down in preparation for a regrasp.
    name = "PlaceToRegrasp"
    nsrt = PlaceStick
    robot, stick = nsrt.parameters
    button = Variable("?button", button_type)
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    # We've already tried to press both ways, and the hand is empty, so we
    # should go to pick up the stick.
    name = "RegraspFromNothing"
    nsrt = PickStickFromNothing
    robot, stick = nsrt.parameters
    button = Variable("?button", button_type)
    parameters = [robot, stick, button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    name = "RegraspFromButton"
    nsrt = PickStickFromButton
    robot, stick, from_button = nsrt.parameters
    button = Variable("?to-button", button_type)
    parameters = [robot, stick, button, from_button]
    pos_preconds = set(nsrt.preconditions) | {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
    }
    neg_preconds = {
        LiftedAtom(Pressed, [button]),
    }
    goal_preconds: Set[LiftedAtom] = set()
    rule = LDLRule(name, parameters, pos_preconds, neg_preconds, goal_preconds, nsrt)
    bridge_rules.append(rule)

    bridge_ldl = LiftedDecisionList(bridge_rules)

    def _bridge_policy(state: State, atoms: Set[GroundAtom],
                       failed_options: List[_Option]) -> _Option:

        # Add failure atoms based on failed_options.
        atoms_with_failures = set(atoms)
        failed_option_specs = {(o.parent, tuple(o.objects)) for o in failed_options}
        for (param_opt, objs) in failed_option_specs:
            for i, obj in enumerate(objs):
                pred = failure_preds[f"{param_opt.name}Failed-arg{i}"]
                failure_atom = GroundAtom(pred, [obj])
                atoms_with_failures.add(failure_atom)

        objects = set(state)
        goal: Set[LiftedAtom] = set()  # task goal not used
        next_nsrt = utils.query_ldl(bridge_ldl, atoms_with_failures, objects, goal)
        if next_nsrt is None:
            raise BridgePolicyDone()

        goal: Set[GroundAtom] = set()  # goal assumed not used by sampler
        return next_nsrt.sample_option(state, goal, rng)

    return _bridge_policy
