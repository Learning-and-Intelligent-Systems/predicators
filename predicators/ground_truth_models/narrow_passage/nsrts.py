"""Ground-truth NSRTs for the narrow passage environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class NarrowPassageGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the narrow passage environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"narrow_passage"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        door_type = types["door"]
        target_type = types["target"]

        # Predicates
        DoorIsClosed = predicates["DoorIsClosed"]
        DoorIsOpen = predicates["DoorIsOpen"]
        TouchedGoal = predicates["TouchedGoal"]

        # Options
        MoveToTarget = options["MoveToTarget"]
        MoveAndOpenDoor = options["MoveAndOpenDoor"]

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
                                       preconditions, add_effects,
                                       delete_effects, ignore_effects, option,
                                       option_vars, random_sampler)
        nsrts.add(move_and_open_door_nsrt)

        return nsrts
