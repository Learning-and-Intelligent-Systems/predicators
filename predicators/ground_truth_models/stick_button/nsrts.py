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
        PlaceStick = options["PlaceStick"]

        nsrts = set()

        # RobotPressButtonFromNothing
        robot = Variable("?robot", robot_type)
        button = Variable("?button", button_type)
        stick = Variable("?stick", stick_type)
        parameters = [robot, button, stick]
        option_vars = [robot, button, stick]
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
        from_button = Variable("?from_button", button_type)
        stick = Variable("?stick", stick_type)
        parameters = [robot, button, from_button, stick]
        option_vars = [robot, button, stick]
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
        button = Variable("?from_button", button_type)
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
        from_button = Variable("?from_button", button_type)
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

        # PlaceStickFromNothing
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        parameters = [robot, stick]
        option_vars = [robot, stick]
        option = PlaceStick
        preconditions = {
            LiftedAtom(Grasped, [robot, stick]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {LiftedAtom(Grasped, [robot, stick])}
        ignore_effects = set()

        def place_stick_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Normalized offset between hand and holder when placing.
            place_pos = rng.uniform(-1, 1)
            return np.array([place_pos], dtype=np.float32)

        place_stick_nsrt = NSRT("PlaceStickFromNothing", parameters,
                                preconditions, add_effects, delete_effects,
                                ignore_effects, option, option_vars,
                                place_stick_sampler)
        nsrts.add(place_stick_nsrt)

        # PlaceStickFromButton
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        parameters = [robot, stick]
        option_vars = [robot, stick]
        option = PlaceStick
        preconditions = {
            LiftedAtom(Grasped, [robot, stick]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(AboveNoButton, [])
        }
        delete_effects = {LiftedAtom(Grasped, [robot, stick])}
        ignore_effects = set()
        place_stick_nsrt = NSRT("PlaceStickFromButton", parameters,
                                preconditions, add_effects, delete_effects,
                                ignore_effects, option, option_vars,
                                place_stick_sampler)
        nsrts.add(place_stick_nsrt)

        return nsrts


class StickButtonMoveGroundTruthNSRTFactory(StickButtonGroundTruthNSRTFactory):
    """Ground-truth NSRTs for the stick button environment with movement
    options."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"stick_button_move"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        button_type = types["button"]
        stick_type = types["stick"]
        holder_type = types["holder"]

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
        PlaceStick = options["PlaceStick"]
        RobotMoveToButton = options["RobotMoveToButton"]
        StickMoveToButton = options["StickMoveToButton"]

        nsrts = set()

        # RobotMoveToButtonFromNothing
        robot = Variable("?robot", robot_type)
        button = Variable("?button", button_type)
        parameters = [robot, button]
        option_vars = [robot, button]
        option = RobotMoveToButton
        preconditions = {
            LiftedAtom(AboveNoButton, []),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(RobotAboveButton, [robot, button]),
        }
        delete_effects = {LiftedAtom(AboveNoButton, [])}
        ignore_effects: Set[Predicate] = set()
        robot_moveto_button_from_nothing_nsrt = NSRT(
            "RobotMoveToButtonFromNothing", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            null_sampler)
        nsrts.add(robot_moveto_button_from_nothing_nsrt)

        # RobotMoveToButtonFromButton
        robot = Variable("?robot", robot_type)
        from_button = Variable("?from", button_type)
        to_button = Variable("?to", button_type)
        parameters = [robot, from_button, to_button]
        option_vars = [robot, to_button]
        option = RobotMoveToButton
        preconditions = {
            LiftedAtom(RobotAboveButton, [robot, from_button]),
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(RobotAboveButton, [robot, to_button]),
        }
        delete_effects = {LiftedAtom(RobotAboveButton, [robot, from_button])}
        robot_moveto_button_from_button_nsrt = NSRT(
            "RobotMoveToButtonFromButton", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            null_sampler)
        nsrts.add(robot_moveto_button_from_button_nsrt)

        # StickMoveToButtonFromButton
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        to_button = Variable("?to", button_type)
        from_button = Variable("?from", button_type)
        parameters = [robot, stick, to_button, from_button]
        option_vars = [robot, to_button, stick]
        option = StickMoveToButton
        preconditions = {
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(StickAboveButton, [stick, from_button]),
        }
        add_effects = {
            LiftedAtom(StickAboveButton, [stick, to_button]),
        }
        delete_effects = {LiftedAtom(StickAboveButton, [stick, from_button])}
        ignore_effects = set()
        stick_moveto_button_from_button_nsrt = NSRT(
            "StickMoveToButtonFromButton", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            null_sampler)
        nsrts.add(stick_moveto_button_from_button_nsrt)

        # StickMoveToButtonFromNothing
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        button = Variable("?to", button_type)
        parameters = [robot, stick, to_button]
        option_vars = [robot, to_button, stick]
        option = StickMoveToButton
        preconditions = {
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        }
        add_effects = {
            LiftedAtom(StickAboveButton, [stick, button]),
        }
        delete_effects = set()
        stick_moveto_button_from_nothing_nsrt = NSRT(
            "StickMoveToButtonFromNothing", parameters, preconditions,
            add_effects, delete_effects, ignore_effects, option, option_vars,
            null_sampler)
        nsrts.add(stick_moveto_button_from_nothing_nsrt)

        # RobotPressButton
        robot = Variable("?robot", robot_type)
        button = Variable("?button", button_type)
        parameters = [robot, button]
        option_vars = [robot, button]
        option = RobotPressButton
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(RobotAboveButton, [robot, button]),
        }
        add_effects = {LiftedAtom(Pressed, [button])}
        delete_effects = set()
        robot_press_button_nsrt = NSRT("RobotPressButton", parameters,
                                       preconditions, add_effects,
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
        button = Variable("?from_button", button_type)
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

        # StickPressButton
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        button = Variable("?button", button_type)
        parameters = [robot, stick, button]
        option_vars = [robot, stick, button]
        option = StickPressButton
        preconditions = {
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(StickAboveButton, [stick, button])
        }
        add_effects = {LiftedAtom(Pressed, [button])}
        delete_effects = set()
        ignore_effects = set()
        stick_button_nsrt = NSRT("StickPressButton", parameters, preconditions,
                                 add_effects, delete_effects, ignore_effects,
                                 option, option_vars, null_sampler)
        nsrts.add(stick_button_nsrt)

        # PlaceStickFromNothing
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        holder = Variable("?holder", holder_type)
        parameters = [robot, stick, holder]
        option_vars = [robot, stick, holder]
        option = PlaceStick
        preconditions = {
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(AboveNoButton, []),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {LiftedAtom(Grasped, [robot, stick])}
        ignore_effects = set()

        def place_stick_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Normalized offset between hand and holder when placing.
            place_pos = rng.uniform(-1, 1)
            return np.array([place_pos], dtype=np.float32)

        place_stick_nsrt = NSRT("PlaceStickFromNothing", parameters,
                                preconditions, add_effects, delete_effects,
                                ignore_effects, option, option_vars,
                                place_stick_sampler)
        nsrts.add(place_stick_nsrt)

        # PlaceStickFromButton
        robot = Variable("?robot", robot_type)
        stick = Variable("?stick", stick_type)
        holder = Variable("?holder", holder_type)
        from_button = Variable("?from_button", button_type)
        parameters = [robot, stick, holder, from_button]
        option_vars = [robot, stick, holder]
        option = PlaceStick
        preconditions = {
            LiftedAtom(StickAboveButton, [stick, from_button]),
            LiftedAtom(Grasped, [robot, stick]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(Grasped, [robot, stick]),
            LiftedAtom(StickAboveButton, [stick, from_button])
        }
        ignore_effects = set()
        place_stick_nsrt = NSRT("PlaceStickFromButton", parameters,
                                preconditions, add_effects, delete_effects,
                                ignore_effects, option, option_vars,
                                place_stick_sampler)
        nsrts.add(place_stick_nsrt)

        return nsrts
