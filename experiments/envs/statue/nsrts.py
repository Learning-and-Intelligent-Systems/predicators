from typing import Callable, ClassVar, Dict, Sequence, Set
import numpy as np
from experiments.envs.donuts.env import Donuts
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Type, Variable
from shapely.affinity import translate

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("tkagg")

__all__ = ['StatueGroundTruthNSRTFactory']

class StatueGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    # Settings

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"statue"}

    @staticmethod
    def get_nsrts(
        env_name: str, types: Dict[str, Type],
        predicates: Dict[str, Predicate],
        options: Dict[str, ParameterizedOption]
    ) -> Set[NSRT]:

        # Types
        room_type = types["room"]
        door_type = types["door"]
        movable_type = types["movable"]
        statue_type = types["statue"]
        robot_type = types["robot"]

        # Prediactes
        InRoom = predicates["InRoom"]
        DoorwayFor = predicates["DoorwayFor"]
        Held = predicates["Held"]

        # Miscellaneous helper variables
        Act = options["Act"]
        nsrts = set()

        # Move
        robot = Variable("?robot", robot_type)
        room_from = Variable("?room_from", room_type)
        door = Variable("?door", door_type)
        room_to = Variable("?room_to", room_type)
        nsrts.add(NSRT(
            "Move",
            [robot, room_from, door, room_to],
            {InRoom([robot, room_from]), DoorwayFor([door, room_from]), DoorwayFor([door, room_to])},
            {InRoom([robot, room_to])},
            {InRoom([robot, room_from])},
            {},
            Act,
            [],
            None, # TODO
        ))

        # MoveHolding
        robot = Variable("?robot", robot_type)
        statue = Variable("?statue", statue_type)
        room_from = Variable("?room_from", room_type)
        door = Variable("?door", door_type)
        room_to = Variable("?room_to", room_type)
        nsrts.add(NSRT(
            "MoveHolding",
            [robot, statue, room_from, door, room_to], {
                InRoom([robot, room_from]), DoorwayFor([door, room_from]), DoorwayFor([door, room_to]),
                InRoom([statue, room_from]), Held([robot, statue]),
            },
            {InRoom([robot, room_to]), InRoom([statue, room_to])},
            {InRoom([robot, room_from]), InRoom([statue, room_from])},
            {},
            Act,
            [],
            None, # TODO
        ))

        # Grab
        robot = Variable("?robot", robot_type)
        statue = Variable("?statue", statue_type)
        room = Variable("?room", room_type)
        nsrts.add(NSRT(
            "Grab",
            [robot, statue, room],
            {InRoom([robot, room]), InRoom([statue, room])},
            {Held([robot, statue])},
            {},
            {},
            Act,
            [],
            None, # TODO
        ))

        # Place
        robot = Variable("?robot", robot_type)
        statue = Variable("?statue", statue_type)
        room = Variable("?room", room_type)
        nsrts.add(NSRT(
            "Place",
            [robot, statue, room],
            {InRoom([robot, room]), InRoom([statue, room]), Held([robot, statue])},
            {},
            {Held([robot, statue])},
            {},
            Act,
            [],
            None, # TODO
        ))

        return nsrts

