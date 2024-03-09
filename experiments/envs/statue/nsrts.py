from typing import Callable, ClassVar, Dict, Sequence, Set
import numpy as np
from experiments.envs.donuts.env import Donuts
from experiments.envs.statue.env import Statue
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Type, Variable
from shapely.affinity import translate

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
        def Move_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[NSRT] = []
        ) -> Array:
            statue = None
            if len(objects) == 5:
                robot, statue, room_from, _, room_to = objects
                statue_width = state.get(statue, "width")
                statue_depth = state.get(statue, "depth")
            else:
                robot, room_from, _, room_to = objects

            x_from, y_from = state.get(room_from, "x"), state.get(room_from, "y")
            x_to, y_to = state.get(room_to, "x"), state.get(room_to, "y")

            if statue is None:
                x = rng.uniform(x_to, x_to + Statue.room_size)
                y = rng.uniform(y_to, y_to + Statue.room_size)
            elif np.allclose(x_from, x_to, atol=Statue.equality_margin) and \
                np.abs(y_from - y_to) <= Statue.room_size + Statue.equality_margin:
                x = rng.uniform(x_to + statue_width / 2, x_to + Statue.room_size - statue_width / 2)
                y = rng.uniform(y_to + statue_depth / 2, y_to + Statue.room_size - statue_depth / 2)
            elif np.allclose(y_from, y_to, atol=Statue.equality_margin) and \
                np.abs(x_from - x_to) <= Statue.room_size + Statue.equality_margin:
                x = rng.uniform(x_to + statue_depth / 2, x_to + Statue.room_size - statue_depth / 2)
                y = rng.uniform(y_to + statue_width / 2, y_to + Statue.room_size - statue_width / 2)
            else:
                x, y = state.get(robot, "x"), state.get(robot, "y")

            arr = np.ones(Act.params_space.shape[0])
            arr[0] = x
            arr[1] = y
            arr[3] = 0.0
            arr[4] = 0.0
            return arr

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
            Move_sampler,
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
            Move_sampler,
        ))

        # Grab
        def Grab_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[NSRT] = []
        ) -> Array:
            _, statue, _ = objects

            if skeleton:
                doors = [nsrt.objects[3] for nsrt in skeleton if nsrt.name == "MoveHolding"]
                statue_height = state.get(statue, "height")
                grasp = float(all(state.get(door, "height") >= statue_height for door in doors))
            else:
                grasp = rng.choice([0.0, 1.0])

            arr = np.ones(Act.params_space.shape[0])
            arr[2] = grasp
            arr[4] = 1.0
            return arr

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
            Grab_sampler,
        ))

        # Place
        def Place_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[NSRT] = []
        ) -> Array:
            arr = np.ones(Act.params_space.shape[0])
            arr[4] = -1.0
            return arr

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
            Place_sampler,
        ))

        return nsrts