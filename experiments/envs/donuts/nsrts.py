from typing import Callable, ClassVar, Dict, Sequence, Set
import numpy as np
from experiments.envs.donuts.env import Donuts
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Type, Variable
from shapely.affinity import translate

__all__ = ['DonutsGroundTruthNSRTFactory']

class DonutsGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    # Settings
    move_search_dist: ClassVar[float] = 2.0
    place_search_dist: ClassVar[float] = 1.0
    margin: ClassVar[float] = 1

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"donuts"}

    @staticmethod
    def get_nsrts(
        env_name: str, types: Dict[str, Type],
        predicates: Dict[str, Predicate],
        options: Dict[str, ParameterizedOption]
    ) -> Set[NSRT]:

        # Types
        object_type = types["object"]
        robot_type = types["robot"]
        donut_type = types["donut"]
        position_type = types["position"]
        container_type = types["container"]
        box_type = types["box"]
        shelf_type = types["shelf"]
        topper_type = types["topper"]
        topper_types = {topping: types[Donuts.topper_format.format(topping)] for topping in Donuts.toppings}

        # Prediactes
        In = predicates["In"]
        NotHeld = predicates["NotHeld"]
        Held = predicates["Held"]
        NextTo = predicates["NextTo"]
        Fresh = predicates["Fresh"]
        CoveredInPredicates = {topping: predicates[Donuts.covered_in_format.format(topping)] for topping in Donuts.toppings}

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
            robot, to_obj = objects

            shapes = Donuts._get_shapes(state)
            to_x, to_y = state.get(to_obj, "x"), state.get(to_obj, "y")
            while True:
                dx, dy = rng.uniform([
                    to_x - DonutsGroundTruthNSRTFactory.move_search_dist,
                    to_y + Donuts.robot_size[1] / 2,
                ], [
                    to_x + DonutsGroundTruthNSRTFactory.move_search_dist,
                    to_y + Donuts.range_table_y[1] - Donuts.range_table_y[0] + Donuts.robot_size[1] / 2 + DonutsGroundTruthNSRTFactory.move_search_dist,
                ])
                robot_new_polygon = translate(shapes[robot], dx, dy)
                if robot_new_polygon.intersects(shapes[to_obj]):
                    continue
                if robot_new_polygon.distance(shapes[to_obj]) <= Donuts.nextto_dist_thresh - DonutsGroundTruthNSRTFactory.margin:
                    break

            arr = np.ones(Act.params_space.shape[0])
            arr[0] = dx
            arr[1] = dy
            arr[3] = 0.0
            arr[4] = 0.0
            return arr

        # MoveToDonut
        robot = Variable("?robot", robot_type)
        to_donut = Variable("?to", donut_type)
        nsrts.add(NSRT(
            "MoveToDonut",
            [robot, to_donut],
            {},
            {NextTo([robot, to_donut])},
            {},
            {NextTo},
            Act,
            [],
            Move_sampler,
        ))

        # MoveToPosition
        robot = Variable("?robot", robot_type)
        to_position = Variable("?to", position_type)
        nsrts.add(NSRT(
            "MoveToPosition",
            [robot, to_position],
            {},
            {NextTo([robot, to_position])},
            {},
            {NextTo},
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
            skeleton: Sequence[NSRT] = [],
        ) -> Array:
            robot, donut = objects
            to_box = next((
                atom for atom in goal
                if atom.predicate == In and atom.objects[0] == donut and atom.objects[1].is_instance(box_type)
            ), None) is not None
            grasp = rng.uniform(Donuts.top_grasp_thresh, 1.0) if to_box else rng.uniform(0.0, Donuts.side_grasp_thresh)

            arr = np.ones(Act.params_space.shape[0])
            arr[2] = grasp
            arr[4] = 1.0
            return arr
        robot = Variable("?robot", robot_type)
        donut = Variable("?donut", donut_type)
        nsrts.add(NSRT(
            "Grab",
            [robot, donut],
            {NextTo([robot, donut]), NotHeld([robot])},
            {Held([robot, donut])},
            {NotHeld([robot])},
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
            skeleton: Sequence[NSRT] = [],
        ) -> Array:
            robot, donut, container = objects

            shapes = Donuts._get_shapes(state)
            container_x, container_y = state.get(container, "x"), state.get(container, "y")
            while True:
                donut_x, donut_y = rng.uniform([
                    container_x - DonutsGroundTruthNSRTFactory.place_search_dist,
                    container_y - DonutsGroundTruthNSRTFactory.place_search_dist,
                ], [
                    container_x + DonutsGroundTruthNSRTFactory.place_search_dist,
                    container_y + DonutsGroundTruthNSRTFactory.place_search_dist
                ])
                if shapes[container].contains(translate(shapes[donut], donut_x, donut_y)):
                    break

            arr = np.ones(Act.params_space.shape[0])
            arr[0] = donut_x
            arr[1] = donut_y
            arr[4] = -1.0
            return arr

        robot = Variable("?robot", robot_type)
        donut = Variable("?donut", donut_type)
        container = Variable("?container", container_type)
        nsrts.add(NSRT(
            "Place",
            [robot, donut, container],
            {NextTo([robot, container]), Held([robot, donut])},
            {NotHeld([robot]), In([donut, container])},
            {Held([robot, donut]), Fresh([donut]), NextTo([robot, donut])},
            {},
            Act,
            [],
            Place_sampler,
        ))

        # AddTopping
        def AddTopping_sampler_helper(idx: int):
            def AddTopping_sampler(
                state: State,
                goal: Set[GroundAtom],
                rng: np.random.Generator,
                objects: Sequence[Object],
                skeleton: Sequence[NSRT] = [],
            ) -> Array:
                robot, donut, topper = objects
                arr = np.ones(Act.params_space.shape[0])
                arr[4] = 0.0
                arr[5 + idx] = 0.0
                return arr
            return AddTopping_sampler
        for idx, topping in enumerate(Donuts.toppings):
            robot = Variable("?robot", robot_type)
            donut = Variable("?donut", donut_type)
            topper = Variable("?topper", topper_types[topping])
            nsrts.add(NSRT(
                f"Add{topping}",
                [robot, donut, topper],
                {NextTo([robot, topper]), Held([robot, donut]), Fresh([donut])},
                {CoveredInPredicates[topping]([donut])},
                {},
                {},
                Act,
                [],
                AddTopping_sampler_helper(idx),
            ))

        return nsrts