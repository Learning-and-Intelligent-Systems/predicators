"""A hand-written bridge policy."""

import logging
from typing import Callable, Dict, List, Set

import numpy as np

from predicators import utils
from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.settings import CFG
from predicators.structs import NSRT, BridgePolicy, GroundAtom, Predicate, \
    State, _Option, Variable, LiftedAtom


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
    RobotPressButtonFromNothing = nsrt_name_to_nsrt[
        "RobotPressButtonFromNothing"]
    PlaceStick = nsrt_name_to_nsrt["PlaceStick"]
    StickPressButtonFromNothing = nsrt_name_to_nsrt[
        "StickPressButtonFromNothing"]

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
            not_failure_pred = Predicate(f"{option.name}Success-arg{idx}", [t],
                _classifier=lambda s, o: False)
            failure_preds[not_failure_pred.name] = not_failure_pred

    RobotPressButtonFailedButton = failure_preds["RobotPressButtonFailed-arg1"]
    StickPressButtonFailedButton = failure_preds["StickPressButtonFailed-arg2"]
    RobotPressButtonSuccessButton = failure_preds["RobotPressButtonSuccess-arg1"]
    NotStickPressButtonSuccessButton = failure_preds["StickPressButtonSuccess-arg2"]

    # Create NSRTs with "failure" preconditions. Note that we don't need to
    # use the NSRT effects because we're just going to invoke the NSRTs in a
    # policy, rather than planning with them, so we don't even bother to write
    # effects. Really this is an abuse of the NSRT data structure.
    add_effects: Set[LiftedAtom] = set()
    delete_effects: Set[LiftedAtom] = set()
    ignore_effects: Set[Predicate] = set()

    bridge_nsrts = set()

    robot = Variable("?robot", robot_type)
    stick = Variable("?stick", stick_type)
    button = Variable("?button", button_type)

    # We haven't tried to press the button yet, and we're holding a stick, so
    # we should put it down.
    name = "PlaceStickBeforeDirectPress"
    parameters = [robot, stick, button]
    option_vars = [robot, stick]
    preconditions = {
        LiftedAtom(RobotPressButtonSuccessButton, [button]),
        LiftedAtom(Grasped, [robot, stick]),
    }
    option = PlaceStick.option
    sampler = PlaceStick.sampler
    nsrt = NSRT(name, parameters, preconditions,
                add_effects, delete_effects, ignore_effects,
                option, option_vars, sampler)
    bridge_nsrts.add(nsrt)

    # We haven't tried to press the button yet, and the hand is empty, so we
    # should go to try the direct press.
    name = "DirectPress"
    parameters = [robot, button]
    option_vars = [robot, button]
    preconditions = {
        LiftedAtom(RobotPressButtonSuccessButton, [button]),
        LiftedAtom(HandEmpty, [robot]),
    }
    option = RobotPressButtonFromNothing.option
    sampler = RobotPressButtonFromNothing.sampler
    nsrt = NSRT(name, parameters, preconditions,
                add_effects, delete_effects, ignore_effects,
                option, option_vars, sampler)
    bridge_nsrts.add(nsrt)

    # We failed to press directly, but haven't yet tried to press with the
    # stick, and the hand is empty, so we should pick up the stick.
    name = "PickStickBeforePress"
    parameters = [robot, stick, button]
    option_vars = [robot, stick]
    preconditions = {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
        LiftedAtom(NotStickPressButtonSuccessButton, [button]),
        LiftedAtom(HandEmpty, [robot]),
    }
    option = PickStickFromNothing.option
    sampler = PickStickFromNothing.sampler
    nsrt = NSRT(name, parameters, preconditions,
                add_effects, delete_effects, ignore_effects,
                option, option_vars, sampler)
    bridge_nsrts.add(nsrt)

    # We failed to press directly, but haven't yet tried to press with the
    # stick, and we're holding the stick, so we should try to press.
    name = "PressWithStick"
    parameters = [robot, stick, button]
    option_vars = [robot, stick]
    preconditions = {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
        LiftedAtom(NotStickPressButtonSuccessButton, [button]),
        LiftedAtom(Grasped, [robot, stick]),
    }
    option = StickPressButtonFromNothing.option
    sampler = StickPressButtonFromNothing.sampler
    nsrt = NSRT(name, parameters, preconditions,
                add_effects, delete_effects, ignore_effects,
                option, option_vars, sampler)
    bridge_nsrts.add(nsrt)

    # We've already tried to press both ways, and we're holding the stick, so
    # we should put it down in preparation for a regrasp.
    name = "PlaceToRegrasp"
    parameters = [robot, stick, button]
    option_vars = [robot, stick]
    preconditions = {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
        LiftedAtom(Grasped, [robot, stick]),
    }
    option = PlaceStick.option
    sampler = PlaceStick.sampler
    nsrt = NSRT(name, parameters, preconditions,
                add_effects, delete_effects, ignore_effects,
                option, option_vars, sampler)
    bridge_nsrts.add(nsrt)

    # We've already tried to press both ways, and the hand is empty, so we
    # should go to pick up the stick.
    name = "Regrasp"
    parameters = [robot, stick, button]
    option_vars = [robot, stick]
    preconditions = {
        LiftedAtom(RobotPressButtonFailedButton, [button]),
        LiftedAtom(StickPressButtonFailedButton, [button]),
        LiftedAtom(HandEmpty, [robot]),
    }
    option = PickStickFromNothing.option
    sampler = PickStickFromNothing.sampler
    nsrt = NSRT(name, parameters, preconditions,
                add_effects, delete_effects, ignore_effects,
                option, option_vars, sampler)
    bridge_nsrts.add(nsrt)

    def _bridge_policy(state: State, atoms: Set[GroundAtom],
                       failed_options: List[_Option]) -> _Option:

        # Add failure atoms based on failed_options.
        atoms_with_failures = set(atoms)
        failed_option_specs = {(o.parent, tuple(o.objects)) for o in failed_options}
        objects = set(state)
        all_option_specs = set()
        for param_opt in all_options:
            for o in utils.get_object_combinations(objects, param_opt.types):
                all_option_specs.add((param_opt, tuple(o)))
        assert failed_option_specs.issubset(all_option_specs)
        for (param_opt, objs) in failed_option_specs:
            for i, obj in enumerate(objs):
                pred = failure_preds[f"{param_opt.name}Failed-arg{i}"]
                failure_atom = GroundAtom(pred, [obj])
                atoms_with_failures.add(failure_atom)
        for (param_opt, objs) in all_option_specs - failed_option_specs:
            for i, obj in enumerate(objs):
                pred = failure_preds[f"{param_opt.name}Success-arg{i}"]
                failure_atom = GroundAtom(pred, [obj])
                atoms_with_failures.add(failure_atom)

        ground_nsrts = sorted(n for nsrt in bridge_nsrts
            for n in utils.all_ground_nsrts(nsrt, objects))

        for ground_nsrt in ground_nsrts:
            if ground_nsrt.preconditions.issubset(atoms_with_failures):
                next_nsrt = ground_nsrt
                print(next_nsrt.name, next_nsrt.objects)
                import ipdb; ipdb.set_trace()
                break
        else:
            raise BridgePolicyDone()

        goal: Set[GroundAtom] = set()  # goal assumed not used by sampler
        return next_nsrt.sample_option(state, goal, rng)

    return _bridge_policy
