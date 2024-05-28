"""Ground-truth NSRTs for the grid row environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class GridRowGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the grid row environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"grid_row"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        light_type = types["light"]
        cell_type = types["cell"]

        # Predicates
        RobotInCell = predicates["RobotInCell"]
        LightInCell = predicates["LightInCell"]
        LightOn = predicates["LightOn"]
        LightOff = predicates["LightOff"]
        Adjacent = predicates["Adjacent"]

        # Options
        MoveRobot = options["MoveRobot"]
        TurnOnLight = options["TurnOnLight"]
        TurnOffLight = options["TurnOffLight"]
        JumpToLight = options["JumpToLight"]

        nsrts = set()

        def light_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: just return a random value from -1 to 1
            return np.array([rng.uniform(-1.0, 1.0)], dtype=np.float32)

        # MoveRobot
        robot = Variable("?robot", robot_type)
        current_cell = Variable("?current_cell", cell_type)
        target_cell = Variable("?target_cell", cell_type)
        parameters = [robot, current_cell, target_cell]
        option_vars = parameters
        option = MoveRobot
        preconditions = {
            LiftedAtom(Adjacent, [current_cell, target_cell]),
            LiftedAtom(RobotInCell, [robot, current_cell]),
        }
        add_effects = {
            LiftedAtom(RobotInCell, [robot, target_cell]),
        }
        delete_effects = {
            LiftedAtom(RobotInCell, [robot, current_cell]),
        }
        ignore_effects: Set[Predicate] = set()
        move_robot_nsrt = NSRT("MoveRobot", parameters, preconditions,
                               add_effects, delete_effects, ignore_effects,
                               option, option_vars, null_sampler)
        nsrts.add(move_robot_nsrt)

        # TurnOnLight
        robot = Variable("?robot", robot_type)
        current_cell = Variable("?current_cell", cell_type)
        light = Variable("?light", light_type)
        parameters = [robot, current_cell, light]
        option_vars = parameters
        option = TurnOnLight
        preconditions = {
            LiftedAtom(LightInCell, [light, current_cell]),
            LiftedAtom(RobotInCell, [robot, current_cell]),
            LiftedAtom(LightOff, [light]),
        }
        add_effects = {
            LiftedAtom(LightOn, [light]),
        }
        delete_effects = {
            LiftedAtom(LightOff, [light]),
        }
        ignore_effects = set()
        turn_light_on_nsrt = NSRT("TurnOnLight", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, light_sampler)
        nsrts.add(turn_light_on_nsrt)

        # TurnOffLight
        robot = Variable("?robot", robot_type)
        current_cell = Variable("?current_cell", cell_type)
        light = Variable("?light", light_type)
        parameters = [robot, current_cell, light]
        option_vars = parameters
        option = TurnOffLight
        preconditions = {
            LiftedAtom(LightInCell, [light, current_cell]),
            LiftedAtom(RobotInCell, [robot, current_cell]),
            LiftedAtom(LightOn, [light]),
        }
        add_effects = {
            LiftedAtom(LightOff, [light]),
        }
        delete_effects = {
            LiftedAtom(LightOn, [light]),
        }
        ignore_effects = set()
        turn_light_off_nsrt = NSRT("TurnOffLight", parameters, preconditions,
                                   add_effects, delete_effects, ignore_effects,
                                   option, option_vars, light_sampler)
        nsrts.add(turn_light_off_nsrt)

        # JumpToLight (Impossible)
        robot = Variable("?robot", robot_type)
        cell1 = Variable("?cell1", cell_type)
        cell2 = Variable("?cell2", cell_type)
        cell3 = Variable("?cell3", cell_type)
        light = Variable("?light", light_type)
        parameters = [robot, cell1, cell2, cell3, light]
        option_vars = parameters
        option = JumpToLight
        preconditions = {
            LiftedAtom(RobotInCell, [robot, cell1]),
            LiftedAtom(Adjacent, [cell1, cell2]),
            LiftedAtom(Adjacent, [cell2, cell3]),
            LiftedAtom(LightInCell, [light, cell3]),
        }
        add_effects = {
            LiftedAtom(RobotInCell, [robot, cell3]),
        }
        delete_effects = {
            LiftedAtom(RobotInCell, [robot, cell1]),
        }
        ignore_effects = set()
        impossible_nsrt = NSRT("JumpToLight", parameters, preconditions,
                               add_effects, delete_effects, ignore_effects,
                               option, option_vars, light_sampler)
        nsrts.add(impossible_nsrt)

        return nsrts


class GridRowDoorGroundTruthNSRTFactory(GridRowGroundTruthNSRTFactory):
    """Ground-truth NSRTs for the grid row door environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"grid_row_door"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        light_type = types["light"]
        cell_type = types["cell"]
        door_type = types["door"]

        # Predicates
        RobotInCell = predicates["RobotInCell"]
        LightInCell = predicates["LightInCell"]
        LightOn = predicates["LightOn"]
        LightOff = predicates["LightOff"]
        Adjacent = predicates["Adjacent"]
        DoorInCell = predicates["DoorInCell"]

        # Options
        MoveRobot = options["MoveRobot"]
        TurnOnLight = options["TurnOnLight"]
        TurnOffLight = options["TurnOffLight"]
        OpenDoor = options["OpenDoor"]

        nsrts = set()

        def light_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: just return a random value from -1 to 1
            return np.array([rng.uniform(-1.0, 1.0)], dtype=np.float32)

        # MoveRobot
        robot = Variable("?robot", robot_type)
        current_cell = Variable("?current_cell", cell_type)
        target_cell = Variable("?target_cell", cell_type)
        parameters = [robot, current_cell, target_cell]
        option_vars = parameters
        option = MoveRobot
        preconditions = {
            LiftedAtom(Adjacent, [current_cell, target_cell]),
            LiftedAtom(RobotInCell, [robot, current_cell]),
        }
        add_effects = {
            LiftedAtom(RobotInCell, [robot, target_cell]),
        }
        delete_effects = {
            LiftedAtom(RobotInCell, [robot, current_cell]),
        }
        ignore_effects: Set[Predicate] = set()
        move_robot_nsrt = NSRT("MoveRobot", parameters, preconditions,
                               add_effects, delete_effects, ignore_effects,
                               option, option_vars, null_sampler)
        nsrts.add(move_robot_nsrt)

        # TurnOnLight
        robot = Variable("?robot", robot_type)
        current_cell = Variable("?current_cell", cell_type)
        light = Variable("?light", light_type)
        parameters = [robot, current_cell, light]
        option_vars = parameters
        option = TurnOnLight
        preconditions = {
            LiftedAtom(LightInCell, [light, current_cell]),
            LiftedAtom(RobotInCell, [robot, current_cell]),
            LiftedAtom(LightOff, [light]),
        }
        add_effects = {
            LiftedAtom(LightOn, [light]),
        }
        delete_effects = {
            LiftedAtom(LightOff, [light]),
        }
        ignore_effects = set()
        turn_light_on_nsrt = NSRT("TurnOnLight", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, light_sampler)
        nsrts.add(turn_light_on_nsrt)

        # TurnOffLight
        robot = Variable("?robot", robot_type)
        current_cell = Variable("?current_cell", cell_type)
        light = Variable("?light", light_type)
        parameters = [robot, current_cell, light]
        option_vars = parameters
        option = TurnOffLight
        preconditions = {
            LiftedAtom(LightInCell, [light, current_cell]),
            LiftedAtom(RobotInCell, [robot, current_cell]),
            LiftedAtom(LightOn, [light]),
        }
        add_effects = {
            LiftedAtom(LightOff, [light]),
        }
        delete_effects = {
            LiftedAtom(LightOn, [light]),
        }
        ignore_effects = set()
        turn_light_off_nsrt = NSRT("TurnOffLight", parameters, preconditions,
                                   add_effects, delete_effects, ignore_effects,
                                   option, option_vars, light_sampler)
        nsrts.add(turn_light_off_nsrt)

        # OpenDoor
        def door_sampler(state: State, goal: Set[GroundAtom],
                         rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            # Note: return 1.0 to show door is now open
            return np.array([rng.uniform(1.0, 1.0)], dtype=np.float32)

        robot = Variable("?robot", robot_type)
        current_cell = Variable("?current_cell", cell_type)
        door = Variable("?door", door_type)
        parameters = [robot, current_cell, door]
        option_vars = parameters
        option = OpenDoor
        preconditions = {
            LiftedAtom(DoorInCell, [door, current_cell]),
            LiftedAtom(RobotInCell, [robot, current_cell]),
        }
        add_effects = set()
        delete_effects = set()
        ignore_effects = set()
        open_door_nsrt = NSRT("OpenDoor", parameters, preconditions,
                              add_effects, delete_effects, ignore_effects,
                              option, option_vars, door_sampler)
        nsrts.add(open_door_nsrt)

        return nsrts
