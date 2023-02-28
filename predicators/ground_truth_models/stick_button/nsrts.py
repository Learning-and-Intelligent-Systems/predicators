"""Ground-truth NSRTs for the stick button environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class StickButtonGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the stick button environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"stick_button"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        button_type = types["button"]
        stick_type = types["stick"]

        # Predicates
        Pressed = predicates["Pressed"]
        RobotAboveButton = predicates["RobotAboveButton"]
        StickAboveButton = predicates["StickAboveButton"]
        Grasped = predicates["Grasped"]
        HandEmpty = predicates["HandEmpty"]
        AboveNoButton = predicates["AboveNoButton"]

        # Options
        RobotPressButton = options["RobotPressButton"]
        PickStick = options["PickStick"]
        StickPressButton = options["StickPressButton"]

        nsrts = set()

        # RobotPressButtonFromNothing
        robot = Variable("?robot", robot_type)
        button = Variable("?button", button_type)
        parameters = [robot, button]
        option_vars = [robot, button]
        option = RobotPressButton
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, []),
        }
        add_effects = {
            LiftedAtom(Pressed, [button]),
            LiftedAtom(RobotAboveButton, [robot, button])
        }
        delete_effects = {LiftedAtom(AboveNoButton, [])}
        ignore_effects: Set[Predicate] = set()
        robot_press_button_nsrt = NSRT("RobotPressButtonFromNothing",
                                       parameters, preconditions, add_effects,
                                       delete_effects, ignore_effects, option,
                                       option_vars, null_sampler)
        nsrts.add(robot_press_button_nsrt)

        # RobotPressButtonFromButton
        robot = Variable("?robot", robot_type)
        button = Variable("?button", button_type)
        from_button = Variable("?from-button", button_type)
        parameters = [robot, button, from_button]
        option_vars = [robot, button]
        option = RobotPressButton
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, from_button]),
        }
        add_effects = {
            LiftedAtom(Pressed, [button]),
            LiftedAtom(RobotAboveButton, [robot, button])
        }
        delete_effects = {LiftedAtom(RobotAboveButton, [robot, from_button])}
        ignore_effects = set()
        robot_press_button_nsrt = NSRT("RobotPressButtonFromButton",
                                       parameters, preconditions, add_effects,
                                       delete_effects, ignore_effects, option,
                                       option_vars, null_sampler)
        nsrts.add(robot_press_button_nsrt)

        # PickStickFromNothing
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        parameters = [robot, stick]
        option_vars = [robot, stick]
        option = PickStick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, []),
        }
        add_effects = {
            LiftedAtom(Grasped, [robot, stick]),
        }
        delete_effects = {LiftedAtom(HandEmpty, [robot])}
        ignore_effects = set()

        def pick_stick_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Normalized x position along the long dimension of the stick, in
            # the center of the short dimension.
            pick_pos = rng.uniform(0, 1)
            return np.array([pick_pos], dtype=np.float32)

        pick_stick_nsrt = NSRT("PickStickFromNothing", parameters,
                               preconditions, add_effects, delete_effects,
                               ignore_effects, option, option_vars,
                               pick_stick_sampler)
        nsrts.add(pick_stick_nsrt)

        # PickStickFromButton
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        button = Variable("?from-button", button_type)
        parameters = [robot, stick, button]
        option_vars = [robot, stick]
        option = PickStick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, button])
        }
        add_effects = {
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, [])
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, button]),
        }
        ignore_effects = set()
        pick_stick_nsrt = NSRT("PickStickFromButton", parameters,
                               preconditions, add_effects, delete_effects,
                               ignore_effects, option, option_vars,
                               pick_stick_sampler)
        nsrts.add(pick_stick_nsrt)

        # StickPressButtonFromNothing
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        button = Variable("?button", button_type)
        parameters = [robot, stick, button]
        option_vars = [robot, stick, button]
        option = StickPressButton
        preconditions = {
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        }
        add_effects = {
            LiftedAtom(StickAboveButton, [stick, button]),
            LiftedAtom(Pressed, [button])
        }
        delete_effects = {LiftedAtom(AboveNoButton, [])}
        ignore_effects = set()
        stick_button_nsrt = NSRT("StickPressButtonFromNothing", parameters,
                                 preconditions, add_effects, delete_effects,
                                 ignore_effects, option, option_vars,
                                 null_sampler)
        nsrts.add(stick_button_nsrt)

        # StickPressButtonFromButton
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        button = Variable("?button", button_type)
        from_button = Variable("?from-button", button_type)
        parameters = [robot, stick, button, from_button]
        option_vars = [robot, stick, button]
        option = StickPressButton
        preconditions = {
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(StickAboveButton, [stick, from_button])
        }
        add_effects = {
            LiftedAtom(StickAboveButton, [stick, button]),
            LiftedAtom(Pressed, [button])
        }
        delete_effects = {LiftedAtom(StickAboveButton, [stick, from_button])}
        ignore_effects = set()
        stick_button_nsrt = NSRT("StickPressButtonFromButton", parameters,
                                 preconditions, add_effects, delete_effects,
                                 ignore_effects, option, option_vars,
                                 null_sampler)
        nsrts.add(stick_button_nsrt)

        return nsrts
