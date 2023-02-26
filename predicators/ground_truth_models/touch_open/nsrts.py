"""Ground-truth NSRTs for the touch open environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs import get_or_create_env
from predicators.envs.touch_point import TouchOpenEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class TouchOpenGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the touch open environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"touch_open"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        door_type = types["door"]

        # Predicates
        TouchingDoor = predicates["TouchingDoor"]
        DoorIsOpen = predicates["DoorIsOpen"]

        # Options
        MoveToDoor = options["MoveToDoor"]
        OpenDoor = options["OpenDoor"]

        nsrts = set()

        # MoveToDoor
        robot = Variable("?robot", robot_type)
        door = Variable("?door", door_type)
        parameters = [robot, door]
        option_vars = [robot, door]
        option = MoveToDoor
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(TouchingDoor, [robot, door])}
        delete_effects: Set[LiftedAtom] = set()
        side_predicates: Set[Predicate] = set()

        def move_to_door_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            del goal, rng  # unused
            robot, door = objs
            assert robot.is_instance(robot_type)
            assert door.is_instance(door_type)
            r_x = state.get(robot, "x")
            r_y = state.get(robot, "y")
            d_x = state.get(door, "x")
            d_y = state.get(door, "y")
            delta_x = d_x - r_x
            delta_y = d_y - r_y
            return np.array([delta_x, delta_y], dtype=np.float32)

        move_to_door_nsrt = NSRT("MoveToDoor", parameters, preconditions,
                                 add_effects, delete_effects, side_predicates,
                                 option, option_vars, move_to_door_sampler)
        nsrts.add(move_to_door_nsrt)

        # OpenDoor
        parameters = [door, robot]
        option_vars = [door, robot]
        option = OpenDoor
        preconditions = {LiftedAtom(TouchingDoor, [robot, door])}
        add_effects = {LiftedAtom(DoorIsOpen, [door])}
        delete_effects = set()
        side_predicates = set()

        # Allow protected access because this is an oracle. Used in the sampler.
        env = get_or_create_env(CFG.env)
        assert isinstance(env, TouchOpenEnv)
        get_open_door_target_value = env._get_open_door_target_value  # pylint: disable=protected-access

        def open_door_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
            del goal, rng
            door, _ = objs
            assert door.is_instance(door_type)
            # Calculate the desired change in the doors "rotation" feature.
            mass = state.get(door, "mass")
            friction = state.get(door, "friction")
            flex = state.get(door, "flex")
            target_rot = get_open_door_target_value(mass=mass,
                                                    friction=friction,
                                                    flex=flex)
            current_rot = state.get(door, "rot")
            # The door always changes from closed to open.
            delta_open = 1.0
            return np.array([target_rot - current_rot, delta_open],
                            dtype=np.float32)

        open_door_nsrt = NSRT("OpenDoor", parameters, preconditions,
                              add_effects, delete_effects, side_predicates,
                              option, option_vars, open_door_sampler)
        nsrts.add(open_door_nsrt)

        return nsrts
