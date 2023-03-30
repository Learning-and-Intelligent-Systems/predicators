"""A hand-written bridge policy."""

import logging
from typing import Callable, Dict, List, Set

import numpy as np

from predicators import utils
from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.settings import CFG
from predicators.structs import NSRT, BridgePolicy, DummyOption, GroundAtom, \
    Predicate, State, _GroundNSRT, _Option


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
            option = self._oracle_bridge_policy(state, atoms, self._failed_options)
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
    RobotPressButtonFromNothing = nsrt_name_to_nsrt["RobotPressButtonFromNothing"]
    PlaceStick = nsrt_name_to_nsrt["PlaceStick"]

    Grasped = pred_name_to_pred["Grasped"]
    Pressed = pred_name_to_pred["Pressed"]

    def _bridge_policy(state: State, atoms: Set[GroundAtom],
                       failed_options: List[_Option]) -> _Option:
        
        robot = next(o for o in state if o.type.name == "robot")
        stick = next(o for o in state if o.type.name == "stick")
        pressed_buttons = {a.objects[0] for a in atoms if a.predicate == Pressed}
        failed_direct_press_buttons = {o.objects[1] for o in failed_options if o.name == "RobotPressButton"}
        failed_stick_press_buttons = {o.objects[2] for o in failed_options if o.name == "StickPressButton"}
        failed_buttons = failed_direct_press_buttons | failed_stick_press_buttons

        # Terminate if all of the failed buttons have now been pressed.
        if failed_buttons.issubset(pressed_buttons):
            raise BridgePolicyDone()

        # Otherwise, find the next button to pursue.
        button = sorted(failed_buttons - pressed_buttons)[0]

        # If we haven't yet tried to directly press the button, try that first.
        if button not in failed_direct_press_buttons:
            # If we're holding the stick, put it down first.
            if Grasped.holds(state, [robot, stick]):
                next_nsrt = PlaceStick.ground([robot, stick])
            else:
                next_nsrt = RobotPressButtonFromNothing.ground([robot, button])
        # If we have already tried to press both ways...
        elif button in failed_direct_press_buttons & failed_stick_press_buttons:
            # If we're already holding the stick, we need to regrasp.
            if Grasped.holds(state, [robot, stick]):
                next_nsrt = PlaceStick.ground([robot, stick])
            # Otherwise, grasp it.
            else:
                next_nsrt = PickStickFromNothing.ground([robot, stick])
        # We haven't yet tried the stick, pick it up.
        elif not Grasped.holds(state, [robot, stick]):
            next_nsrt = PickStickFromNothing.ground([robot, stick])
        # Otherwise, we should be holding the stick, give control back.
        else:
            raise BridgePolicyDone()

        goal: Set[GroundAtom] = set()  # goal assumed not used by sampler
        return next_nsrt.sample_option(state, goal, rng)

    return _bridge_policy
