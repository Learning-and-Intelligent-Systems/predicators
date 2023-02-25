"""Implements ground-truth NSRTs and options."""
import abc
from typing import Dict, List, Sequence, Set

import numpy as np

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.envs.doors import DoorsEnv
from predicators.envs.satellites import SatellitesEnv
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class GroundTruthOptionFactory(abc.ABC):
    """Parent class for ground-truth option definitions."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds options for."""
        raise NotImplementedError("Override me!")

    @staticmethod
    @abc.abstractmethod
    def get_options(env_name: str) -> Set[ParameterizedOption]:
        """Create options for the given env name."""
        raise NotImplementedError("Override me!")


class GroundTruthNSRTFactory(abc.ABC):
    """Parent class for ground-truth NSRT definitions."""

    @classmethod
    @abc.abstractmethod
    def get_env_names(cls) -> Set[str]:
        """Get the env names that this factory builds NSRTs for."""
        raise NotImplementedError("Override me!")

    @staticmethod
    @abc.abstractmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        """Create NSRTs for the given env name."""
        raise NotImplementedError("Override me!")


def get_gt_options(env_name: str) -> Set[ParameterizedOption]:
    """Create ground truth options for an env."""
    # This is a work in progress. Gradually moving options out of environments
    # until we can remove them from the environment API entirely.
    for cls in utils.get_all_subclasses(GroundTruthOptionFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            options = factory.get_options(env_name)
            break
    else:
        # In the final version of this function, we will instead raise an
        # error in this case.
        env = get_or_create_env(env_name)
        options = env.options
    # Seed the options for reproducibility.
    for option in options:
        option.params_space.seed(CFG.seed)
    return options


def get_gt_nsrts(env_name: str, predicates_to_keep: Set[Predicate],
                 options_to_keep: Set[ParameterizedOption]) -> Set[NSRT]:
    """Create ground truth options for an env."""
    # This is a work in progress. Gradually moving NSRTs into env-specific
    # files; ground_truth_nsrts.py will be deleted.
    env = get_or_create_env(env_name)
    env_options = get_gt_options(env_name)
    assert predicates_to_keep.issubset(env.predicates)
    assert options_to_keep.issubset(env_options)
    for cls in utils.get_all_subclasses(GroundTruthNSRTFactory):
        if not cls.__abstractmethods__ and env_name in cls.get_env_names():
            factory = cls()
            # Give all predicates and options, then filter based on kept ones
            # at the end of this function. This is easier than filtering within
            # the factory itself.
            types = {t.name: t for t in env.types}
            predicates = {p.name: p for p in env.predicates}
            options = {o.name: o for o in env_options}
            nsrts = factory.get_nsrts(env_name, types, predicates, options)
            break
    else:
        # TODO: In the final version of this function, we will instead raise an
        # error in this case.
        nsrts = deprecated_get_gt_nsrts(env_name)
    # Filter out excluded predicates from NSRTs, and filter out NSRTs whose
    # options are excluded.
    final_nsrts = set()
    for nsrt in nsrts:
        if nsrt.option not in options_to_keep:
            continue
        nsrt = nsrt.filter_predicates(predicates_to_keep)
        final_nsrts.add(nsrt)
    return final_nsrts


def parse_config_included_options(env: BaseEnv) -> Set[ParameterizedOption]:
    """Parse the CFG.included_options string, given an environment.

    Return the set of included oracle options.

    Note that "all" is not implemented because setting the option_learner flag
    to "no_learning" is the preferred way to include all options.
    """
    if not CFG.included_options:
        return set()
    env_options = get_gt_options(env.get_name())
    included_names = set(CFG.included_options.split(","))
    assert included_names.issubset({option.name for option in env_options}), \
        "Unrecognized option in included_options!"
    included_options = {o for o in env_options if o.name in included_names}
    return included_options


# Find the factories.
utils.import_submodules(__path__, __name__)

############# TODO: EVERYTHING BELOW HERE IS SCHEDULED FOR REMOVAL ############


def deprecated_get_gt_nsrts(env_name: str) -> Set[NSRT]:
    """Create ground truth NSRTs for an env."""
    if env_name == "coffee":
        nsrts = _get_coffee_gt_nsrts(env_name)
    elif env_name in ("satellites", "satellites_simple"):
        nsrts = _get_satellites_gt_nsrts(env_name)
    else:
        assert env_name in ("sandwich", "sandwich_clear")
        nsrts = _get_sandwich_gt_nsrts(env_name)
    return nsrts


def _get_from_env_by_names(env_name: str, names: Sequence[str],
                           env_attr: str) -> List:
    """Helper for loading types, predicates, and options by name."""
    env = get_or_create_env(env_name)
    name_to_env_obj = {}
    for o in getattr(env, env_attr):
        name_to_env_obj[o.name] = o
    assert set(name_to_env_obj).issuperset(set(names))
    return [name_to_env_obj[name] for name in names]


def _get_types_by_names(env_name: str, names: Sequence[str]) -> List[Type]:
    """Load types from an env given their names."""
    return _get_from_env_by_names(env_name, names, "types")


def _get_predicates_by_names(env_name: str,
                             names: Sequence[str]) -> List[Predicate]:
    """Load predicates from an env given their names."""
    return _get_from_env_by_names(env_name, names, "predicates")


def _get_options_by_names(env_name: str,
                          names: Sequence[str]) -> List[ParameterizedOption]:
    """Load parameterized options from an env given their names."""
    options = get_gt_options(env_name)
    name_to_option = {o.name: o for o in options}
    return [name_to_option[name] for name in names]


def _get_narrow_passage_gt_nsrts(env_name: str) -> Set[NSRT]:
    """Create ground truth NSRTs for NarrowPassageEnv."""
    robot_type, door_type, target_type = _get_types_by_names(
        env_name, ["robot", "door", "target"])
    DoorIsClosed, DoorIsOpen, TouchedGoal = _get_predicates_by_names(
        env_name, ["DoorIsClosed", "DoorIsOpen", "TouchedGoal"])
    MoveToTarget, MoveAndOpenDoor = _get_options_by_names(
        env_name, ["MoveToTarget", "MoveAndOpenDoor"])

    nsrts = set()

    def random_sampler(state: State, goal: Set[GroundAtom],
                       rng: np.random.Generator,
                       objs: Sequence[Object]) -> Array:
        del state, goal, objs  # unused
        # Note: just return a random value from 0 to 1
        return np.array([rng.uniform()], dtype=np.float32)

    # MoveToTarget
    robot = Variable("?robot", robot_type)
    target = Variable("?target", target_type)
    parameters = [robot, target]
    option_vars = [robot, target]
    option = MoveToTarget
    preconditions: Set[LiftedAtom] = set()
    add_effects: Set[LiftedAtom] = {
        LiftedAtom(TouchedGoal, [robot, target]),
    }
    delete_effects: Set[LiftedAtom] = set()
    ignore_effects: Set[Predicate] = set()
    move_to_target_nsrt = NSRT("MoveToTarget", parameters, preconditions,
                               add_effects, delete_effects, ignore_effects,
                               option, option_vars, random_sampler)
    nsrts.add(move_to_target_nsrt)

    # MoveAndOpenDoor
    robot = Variable("?robot", robot_type)
    door = Variable("?door", door_type)
    parameters = [robot, door]
    option_vars = [robot, door]
    option = MoveAndOpenDoor
    preconditions = {
        LiftedAtom(DoorIsClosed, [door]),
    }
    add_effects = {
        LiftedAtom(DoorIsOpen, [door]),
    }
    delete_effects = {
        LiftedAtom(DoorIsClosed, [door]),
    }
    ignore_effects = set()
    move_and_open_door_nsrt = NSRT("MoveAndOpenDoor", parameters,
                                   preconditions, add_effects, delete_effects,
                                   ignore_effects, option, option_vars,
                                   random_sampler)
    nsrts.add(move_and_open_door_nsrt)

    return nsrts


def _get_coffee_gt_nsrts(env_name: str) -> Set[NSRT]:
    """Create ground truth NSRTs for CoffeeEnv."""
    robot_type, jug_type, cup_type, machine_type = _get_types_by_names(
        env_name, ["robot", "jug", "cup", "machine"])
    CupFilled, Holding, JugInMachine, MachineOn, OnTable, HandEmpty, \
        JugFilled, RobotAboveCup, JugAboveCup, NotAboveCup, PressingButton, \
        Twisting, NotSameCup = \
        _get_predicates_by_names(env_name, ["CupFilled",
            "Holding", "JugInMachine", "MachineOn", "OnTable", "HandEmpty",
            "JugFilled", "RobotAboveCup", "JugAboveCup", "NotAboveCup",
            "PressingButton", "Twisting", "NotSameCup"])
    MoveToTwistJug, TwistJug, PickJug, PlaceJugInMachine, TurnMachineOn, \
        Pour = _get_options_by_names(env_name, ["MoveToTwistJug", "TwistJug",
            "PickJug", "PlaceJugInMachine", "TurnMachineOn", "Pour"])

    nsrts = set()

    # MoveToTwistJug
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    parameters = [robot, jug]
    option_vars = [robot, jug]
    option = MoveToTwistJug
    preconditions = {
        LiftedAtom(OnTable, [jug]),
        LiftedAtom(HandEmpty, [robot]),
    }
    add_effects = {
        LiftedAtom(Twisting, [robot, jug]),
    }
    delete_effects = {
        LiftedAtom(HandEmpty, [robot]),
    }
    ignore_effects: Set[Predicate] = set()
    move_to_twist_jug_nsrt = NSRT("MoveToTwistJug", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, null_sampler)
    nsrts.add(move_to_twist_jug_nsrt)

    # TwistJug
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    parameters = [robot, jug]
    option_vars = [robot, jug]
    option = TwistJug
    preconditions = {
        LiftedAtom(OnTable, [jug]),
        LiftedAtom(Twisting, [robot, jug]),
    }
    add_effects = {
        LiftedAtom(HandEmpty, [robot]),
    }
    delete_effects = {
        LiftedAtom(Twisting, [robot, jug]),
    }
    ignore_effects = set()

    def twist_jug_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
        del state, goal, objs  # unused
        return np.array(rng.uniform(-1, 1, size=(1, )), dtype=np.float32)

    twist_jug_nsrt = NSRT("TwistJug", parameters, preconditions, add_effects,
                          delete_effects, ignore_effects, option, option_vars,
                          twist_jug_sampler)
    nsrts.add(twist_jug_nsrt)

    # PickJugFromTable
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    parameters = [robot, jug]
    option_vars = [robot, jug]
    option = PickJug
    preconditions = {
        LiftedAtom(OnTable, [jug]),
        LiftedAtom(HandEmpty, [robot])
    }
    add_effects = {
        LiftedAtom(Holding, [robot, jug]),
    }
    delete_effects = {
        LiftedAtom(OnTable, [jug]),
        LiftedAtom(HandEmpty, [robot])
    }
    ignore_effects = set()
    pick_jug_from_table_nsrt = NSRT("PickJugFromTable", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(pick_jug_from_table_nsrt)

    # PlaceJugInMachine
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    machine = Variable("?machine", machine_type)
    parameters = [robot, jug, machine]
    option_vars = [robot, jug, machine]
    option = PlaceJugInMachine
    preconditions = {
        LiftedAtom(Holding, [robot, jug]),
    }
    add_effects = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(JugInMachine, [jug, machine]),
    }
    delete_effects = {
        LiftedAtom(Holding, [robot, jug]),
    }
    ignore_effects = set()
    place_jug_in_machine_nsrt = NSRT("PlaceJugInMachine", parameters,
                                     preconditions, add_effects,
                                     delete_effects, ignore_effects, option,
                                     option_vars, null_sampler)
    nsrts.add(place_jug_in_machine_nsrt)

    # TurnMachineOn
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    machine = Variable("?machine", machine_type)
    parameters = [robot, jug, machine]
    option_vars = [robot, machine]
    option = TurnMachineOn
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(JugInMachine, [jug, machine]),
    }
    add_effects = {
        LiftedAtom(JugFilled, [jug]),
        LiftedAtom(MachineOn, [machine]),
        LiftedAtom(PressingButton, [robot, machine]),
    }
    delete_effects = set()
    ignore_effects = set()
    turn_machine_on_nsrt = NSRT("TurnMachineOn", parameters, preconditions,
                                add_effects, delete_effects, ignore_effects,
                                option, option_vars, null_sampler)
    nsrts.add(turn_machine_on_nsrt)

    # PickJugFromMachine
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    machine = Variable("?machine", machine_type)
    parameters = [robot, jug, machine]
    option_vars = [robot, jug]
    option = PickJug
    preconditions = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(JugInMachine, [jug, machine]),
        LiftedAtom(PressingButton, [robot, machine]),
    }
    add_effects = {
        LiftedAtom(Holding, [robot, jug]),
    }
    delete_effects = {
        LiftedAtom(HandEmpty, [robot]),
        LiftedAtom(JugInMachine, [jug, machine]),
        LiftedAtom(PressingButton, [robot, machine]),
    }
    ignore_effects = set()
    pick_jug_from_machine_nsrt = NSRT("PickJugFromMachine", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects, option,
                                      option_vars, null_sampler)
    nsrts.add(pick_jug_from_machine_nsrt)

    # PourFromNowhere
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    cup = Variable("?cup", cup_type)
    parameters = [robot, jug, cup]
    option_vars = [robot, jug, cup]
    option = Pour
    preconditions = {
        LiftedAtom(Holding, [robot, jug]),
        LiftedAtom(JugFilled, [jug]),
        LiftedAtom(NotAboveCup, [robot, jug]),
    }
    add_effects = {
        LiftedAtom(JugAboveCup, [jug, cup]),
        LiftedAtom(RobotAboveCup, [robot, cup]),
        LiftedAtom(CupFilled, [cup]),
    }
    delete_effects = {
        LiftedAtom(NotAboveCup, [robot, jug]),
    }
    ignore_effects = set()
    pour_from_nowhere_nsrt = NSRT("PourFromNowhere", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, null_sampler)
    nsrts.add(pour_from_nowhere_nsrt)

    # PourFromOtherCup
    robot = Variable("?robot", robot_type)
    jug = Variable("?jug", jug_type)
    cup = Variable("?cup", cup_type)
    other_cup = Variable("?other_cup", cup_type)
    parameters = [robot, jug, cup, other_cup]
    option_vars = [robot, jug, cup]
    option = Pour
    preconditions = {
        LiftedAtom(Holding, [robot, jug]),
        LiftedAtom(JugFilled, [jug]),
        LiftedAtom(JugAboveCup, [jug, other_cup]),
        LiftedAtom(RobotAboveCup, [robot, other_cup]),
        LiftedAtom(NotSameCup, [cup, other_cup]),
    }
    add_effects = {
        LiftedAtom(JugAboveCup, [jug, cup]),
        LiftedAtom(RobotAboveCup, [robot, cup]),
        LiftedAtom(CupFilled, [cup]),
    }
    delete_effects = {
        LiftedAtom(JugAboveCup, [jug, other_cup]),
        LiftedAtom(RobotAboveCup, [robot, other_cup]),
    }
    ignore_effects = set()
    pour_from_other_cup_nsrt = NSRT("PourFromOtherCup", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(pour_from_other_cup_nsrt)

    return nsrts


def _get_satellites_gt_nsrts(env_name: str) -> Set[NSRT]:
    """Create ground truth NSRTs for SatellitesEnv."""
    sat_type, obj_type = _get_types_by_names(env_name, ["satellite", "object"])
    Sees, CalibrationTarget, IsCalibrated, HasCamera, HasInfrared, HasGeiger, \
        ShootsChemX, ShootsChemY, HasChemX, HasChemY, CameraReadingTaken, \
        InfraredReadingTaken, GeigerReadingTaken = _get_predicates_by_names(
            env_name, ["Sees", "CalibrationTarget", "IsCalibrated",
                      "HasCamera", "HasInfrared", "HasGeiger",
                      "ShootsChemX", "ShootsChemY", "HasChemX", "HasChemY",
                      "CameraReadingTaken", "InfraredReadingTaken",
                      "GeigerReadingTaken"])
    MoveTo, Calibrate, ShootChemX, ShootChemY, UseInstrument = \
        _get_options_by_names(
            env_name, ["MoveTo", "Calibrate", "ShootChemX", "ShootChemY",
                      "UseInstrument"])

    nsrts = set()

    # MoveTo
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = MoveTo
    preconditions: Set[LiftedAtom] = set()
    add_effects = {
        LiftedAtom(Sees, [sat, obj]),
    }
    delete_effects: Set[LiftedAtom] = set()
    ignore_effects = {Sees}

    def moveto_sampler(state: State, goal: Set[GroundAtom],
                       rng: np.random.Generator,
                       objs: Sequence[Object]) -> Array:
        del goal  # unused
        _, obj = objs
        obj_x = state.get(obj, "x")
        obj_y = state.get(obj, "y")
        min_dist = SatellitesEnv.radius * 4
        max_dist = SatellitesEnv.fov_dist - SatellitesEnv.radius * 2
        dist = rng.uniform(min_dist, max_dist)
        angle = rng.uniform(-np.pi, np.pi)
        x = obj_x + dist * np.cos(angle)
        y = obj_y + dist * np.sin(angle)
        return np.array([x, y], dtype=np.float32)

    moveto_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                       delete_effects, ignore_effects, option, option_vars,
                       moveto_sampler)
    nsrts.add(moveto_nsrt)

    # Calibrate
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = Calibrate
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(CalibrationTarget, [sat, obj]),
    }
    add_effects = {
        LiftedAtom(IsCalibrated, [sat]),
    }
    delete_effects = set()
    ignore_effects = set()
    calibrate_nsrt = NSRT("Calibrate", parameters, preconditions, add_effects,
                          delete_effects, ignore_effects, option, option_vars,
                          null_sampler)
    nsrts.add(calibrate_nsrt)

    # ShootChemX
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = ShootChemX
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(ShootsChemX, [sat]),
    }
    add_effects = {
        LiftedAtom(HasChemX, [obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    shoot_chem_x_nsrt = NSRT("ShootChemX", parameters, preconditions,
                             add_effects, delete_effects, ignore_effects,
                             option, option_vars, null_sampler)
    nsrts.add(shoot_chem_x_nsrt)

    # ShootChemY
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = ShootChemY
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(ShootsChemY, [sat]),
    }
    add_effects = {
        LiftedAtom(HasChemY, [obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    shoot_chem_y_nsrt = NSRT("ShootChemY", parameters, preconditions,
                             add_effects, delete_effects, ignore_effects,
                             option, option_vars, null_sampler)
    nsrts.add(shoot_chem_y_nsrt)

    # TakeCameraReading
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = UseInstrument
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(IsCalibrated, [sat]),
        LiftedAtom(HasCamera, [sat]),
        # taking a camera reading requires Chemical X
        LiftedAtom(HasChemX, [obj]),
    }
    add_effects = {
        LiftedAtom(CameraReadingTaken, [sat, obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    take_camera_reading_nsrt = NSRT("TakeCameraReading", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(take_camera_reading_nsrt)

    # TakeInfraredReading
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = UseInstrument
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(IsCalibrated, [sat]),
        LiftedAtom(HasInfrared, [sat]),
        # taking an infrared reading requires Chemical Y
        LiftedAtom(HasChemY, [obj]),
    }
    add_effects = {
        LiftedAtom(InfraredReadingTaken, [sat, obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    take_infrared_reading_nsrt = NSRT("TakeInfraredReading", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects, option,
                                      option_vars, null_sampler)
    nsrts.add(take_infrared_reading_nsrt)

    # TakeGeigerReading
    sat = Variable("?sat", sat_type)
    obj = Variable("?obj", obj_type)
    parameters = [sat, obj]
    option_vars = [sat, obj]
    option = UseInstrument
    preconditions = {
        LiftedAtom(Sees, [sat, obj]),
        LiftedAtom(IsCalibrated, [sat]),
        LiftedAtom(HasGeiger, [sat]),
        # taking a Geiger reading doesn't require any chemical
    }
    add_effects = {
        LiftedAtom(GeigerReadingTaken, [sat, obj]),
    }
    delete_effects = set()
    ignore_effects = set()
    take_geiger_reading_nsrt = NSRT("TakeGeigerReading", parameters,
                                    preconditions, add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    null_sampler)
    nsrts.add(take_geiger_reading_nsrt)

    return nsrts


def _get_sandwich_gt_nsrts(env_name: str) -> Set[NSRT]:
    """Create ground truth NSRTs for SandwichEnv."""
    robot_type, ingredient_type, board_type, holder_type = _get_types_by_names(
        env_name, ["robot", "ingredient", "board", "holder"])

    On, OnBoard, InHolder, GripperOpen, Holding, Clear, BoardClear = \
        _get_predicates_by_names(env_name, ["On", "OnBoard", "InHolder",
                                            "GripperOpen", "Holding", "Clear",
                                            "BoardClear"])

    Pick, Stack, PutOnBoard = _get_options_by_names(
        env_name, ["Pick", "Stack", "PutOnBoard"])

    nsrts = set()

    # PickFromHolder
    ing = Variable("?ing", ingredient_type)
    robot = Variable("?robot", robot_type)
    holder = Variable("?holder", holder_type)
    parameters = [ing, robot, holder]
    option_vars = [robot, ing]
    option = Pick
    preconditions = {
        LiftedAtom(InHolder, [ing, holder]),
        LiftedAtom(GripperOpen, [robot])
    }
    add_effects = {LiftedAtom(Holding, [ing, robot])}
    delete_effects = {
        LiftedAtom(InHolder, [ing, holder]),
        LiftedAtom(GripperOpen, [robot])
    }

    pickfromholder_nsrt = NSRT("PickFromHolder", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, null_sampler)
    nsrts.add(pickfromholder_nsrt)

    # Stack
    ing = Variable("?ing", ingredient_type)
    othering = Variable("?othering", ingredient_type)
    robot = Variable("?robot", robot_type)
    parameters = [ing, othering, robot]
    option_vars = [robot, othering]
    option = Stack
    preconditions = {
        LiftedAtom(Holding, [ing, robot]),
        LiftedAtom(Clear, [othering])
    }
    add_effects = {
        LiftedAtom(On, [ing, othering]),
        LiftedAtom(Clear, [ing]),
        LiftedAtom(GripperOpen, [robot])
    }
    delete_effects = {
        LiftedAtom(Holding, [ing, robot]),
        LiftedAtom(Clear, [othering])
    }

    stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                      delete_effects, set(), option, option_vars, null_sampler)
    nsrts.add(stack_nsrt)

    # PutOnBoard
    ing = Variable("?ing", ingredient_type)
    robot = Variable("?robot", robot_type)
    board = Variable("?board", board_type)
    parameters = [ing, robot, board]
    option_vars = [robot, board]
    option = PutOnBoard
    preconditions = {
        LiftedAtom(Holding, [ing, robot]),
        LiftedAtom(BoardClear, [board]),
    }
    add_effects = {
        LiftedAtom(OnBoard, [ing, board]),
        LiftedAtom(Clear, [ing]),
        LiftedAtom(GripperOpen, [robot])
    }
    delete_effects = {
        LiftedAtom(Holding, [ing, robot]),
        LiftedAtom(BoardClear, [board]),
    }

    putonboard_nsrt = NSRT("PutOnBoard", parameters, preconditions,
                           add_effects, delete_effects, set(), option,
                           option_vars, null_sampler)
    nsrts.add(putonboard_nsrt)

    return nsrts
