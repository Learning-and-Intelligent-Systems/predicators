"""Ground-truth NSRTs for the doors environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs import get_or_create_env
from predicators.envs.doors import DoorKnobsEnv, DoorsEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class DoorsGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the doors environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"doors"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        door_type = types["door"]
        room_type = types["room"]

        # Predicates
        InRoom = predicates["InRoom"]
        InDoorway = predicates["InDoorway"]
        InMainRoom = predicates["InMainRoom"]

        DoorInRoom = predicates["DoorInRoom"]
        DoorsShareRoom = predicates["DoorsShareRoom"]

        if env_name == "doors":
            TouchingDoor = predicates["TouchingDoor"]
            DoorIsOpen = predicates["DoorIsOpen"]

        # Options
        MoveToDoor = options["MoveToDoor"]
        OpenDoor = options["OpenDoor"]
        MoveThroughDoor = options["MoveThroughDoor"]

        nsrts = set()

        # MoveToDoorFromMainRoom
        # This operator should only be used on the first step of a plan.
        robot = Variable("?robot", robot_type)
        room = Variable("?room", room_type)
        door = Variable("?door", door_type)
        parameters = [robot, room, door]
        option_vars = [robot, door]
        option = MoveToDoor
        preconditions = {
            LiftedAtom(InRoom, [robot, room]),
            LiftedAtom(InMainRoom, [robot, room]),
            LiftedAtom(DoorInRoom, [door, room]),
        }
        add_effects = {LiftedAtom(InDoorway, [robot, door])}

        if env_name == "doors":
            add_effects.add(LiftedAtom(TouchingDoor, [robot, door]))

        delete_effects = {LiftedAtom(InMainRoom, [robot, room])}
        ignore_effects: Set[Predicate] = set()
        move_to_door_nsrt = NSRT("MoveToDoorFromMainRoom", parameters,
                                 preconditions, add_effects, delete_effects,
                                 ignore_effects, option, option_vars,
                                 null_sampler)
        nsrts.add(move_to_door_nsrt)

        # MoveToDoorFromDoorWay
        robot = Variable("?robot", robot_type)
        start_door = Variable("?start_door", door_type)
        end_door = Variable("?end_door", door_type)
        parameters = [robot, start_door, end_door]
        option_vars = [robot, end_door]
        option = MoveToDoor
        preconditions = {
            LiftedAtom(InDoorway, [robot, start_door]),
            LiftedAtom(DoorsShareRoom, [start_door, end_door]),
        }
        add_effects = {LiftedAtom(InDoorway, [robot, end_door])}
        if env_name == "doors":
            add_effects.add(LiftedAtom(TouchingDoor, [robot, end_door]))

        delete_effects = {LiftedAtom(InDoorway, [robot, start_door])}
        ignore_effects = set()
        move_to_door_nsrt = NSRT("MoveToDoorFromDoorWay", parameters,
                                 preconditions, add_effects, delete_effects,
                                 ignore_effects, option, option_vars,
                                 null_sampler)
        nsrts.add(move_to_door_nsrt)

        # OpenDoor
        if env_name == "doors":
            robot = Variable("?robot", robot_type)
            door = Variable("?door", door_type)
            parameters = [door, robot]
            option_vars = [door, robot]
            option = OpenDoor
            preconditions = {
                LiftedAtom(TouchingDoor, [robot, door]),
                LiftedAtom(InDoorway, [robot, door]),
            }
            add_effects = {LiftedAtom(DoorIsOpen, [door])}
            delete_effects = {
                LiftedAtom(TouchingDoor, [robot, door]),
            }
            ignore_effects = set()

            # Allow protected access because this is an oracle. \
            #  Used in the sampler.
            env = get_or_create_env(env_name)
            assert isinstance(env, (DoorKnobsEnv, DoorsEnv))
            get_open_door_target_value = env._get_open_door_target_value  # pylint: disable=protected-access

            # Even though this option does not need to be parameterized,
            # we make it so, because we want to match the parameter
            # space of the option that will get learned during option
            # learning. This is useful for when we
            # want to use sampler_learner = "oracle" too.
            def open_door_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
                del rng, goal  # unused
                door, _ = objs
                assert door.is_instance(door_type)
                # Calculate the desired change in the doors
                # "rotation" feature.
                # Allow protected access because this is an oracle.
                mass = state.get(door, "mass")
                friction = state.get(door, "friction")
                target_rot = state.get(door, "target_rot")
                target_val = get_open_door_target_value(mass=mass,
                                                        friction=friction,
                                                        target_rot=target_rot)
                current_val = state.get(door, "rot")
                delta_rot = target_val - current_val
                # The door always changes from closed to open.
                delta_open = 1.0
                return np.array([delta_rot, delta_open], dtype=np.float32)

            open_door_nsrt = NSRT("OpenDoor", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, open_door_sampler)

        elif env_name == "doorknobs":
            robot = Variable("?robot", robot_type)
            parameters = [robot]
            option_vars = [robot]
            option = OpenDoor
            preconditions = set()
            add_effects = set()
            delete_effects = set()
            ignore_effects = set()

            def open_doorknob_sampler(_: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      __: Sequence[Object]) -> Array:
                del goal  # unused  # pragma: no cover
                return np.array([rng.uniform(-1.0, 1.0)],
                                dtype=np.float32)  # pragma: no cover

            open_door_nsrt = NSRT("OpenDoor", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, open_doorknob_sampler)

        nsrts.add(open_door_nsrt)

        # MoveThroughDoor
        robot = Variable("?robot", robot_type)
        start_room = Variable("?start_room", room_type)
        end_room = Variable("?end_room", room_type)
        door = Variable("?door", door_type)
        parameters = [robot, start_room, door, end_room]
        option_vars = [robot, door]
        option = MoveThroughDoor
        preconditions = {
            LiftedAtom(InRoom, [robot, start_room]),
            LiftedAtom(InDoorway, [robot, door]),
            LiftedAtom(DoorInRoom, [door, start_room]),
            LiftedAtom(DoorInRoom, [door, end_room]),
        }
        if env_name == "doors":
            preconditions.add(LiftedAtom(DoorIsOpen, [door]))

        add_effects = {
            LiftedAtom(InRoom, [robot, end_room]),
        }
        delete_effects = {
            LiftedAtom(InRoom, [robot, start_room]),
        }
        ignore_effects = set()
        move_through_door_nsrt = NSRT("MoveThroughDoor", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects, option,
                                      option_vars, null_sampler)
        nsrts.add(move_through_door_nsrt)

        return nsrts


class DoorknobsGroundTruthNSRTFactory(DoorsGroundTruthNSRTFactory):
    """Ground-truth NSRTs for the doors environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"doorknobs"}
