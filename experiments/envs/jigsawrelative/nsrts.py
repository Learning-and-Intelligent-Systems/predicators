import itertools
import logging
from typing import Dict, Sequence, Set
import numpy as np
from experiments.envs.jigsawrelative.env import JigsawRelative
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, _GroundNSRT, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Type, Variable
from shapely import Point

__all__ = ['JigsawRelativeGroundTruthNSRTFactory']

class JigsawRelativeGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    # Settings
    placement_margin = 0.1
    grasp_margin = 0.1
    move_search_margin = 6
    next_to_margin = 0.8

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"jigsawrelative"}

    @classmethod
    def get_nsrts(
        cls, env_name: str, types: Dict[str, Type],
        predicates: Dict[str, Predicate],
        options: Dict[str, ParameterizedOption]
    ) -> Set[NSRT]:

        # Types
        robot_type = types["robot"]
        object_type = types["object"]
        container_type = types["container"]
        block_type = types["block"]

        # Prediactes
        Inside = predicates["Inside"]
        Outside = predicates["Outside"]
        NextToRobot = predicates["NextToRobot"]
        Held = predicates["Held"]
        NotHeld = predicates["NotHeld"]

        # Miscellaneous helper variables
        Act = options["Act"]
        nsrts = set()

        # Grab
        def Grab_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[_GroundNSRT] = []
        ) -> Array:
            _, _, block = objects
            block_shape = JigsawRelative._get_shape(state, block).buffer(-cls.grasp_margin, join_style="mitre")

            min_x, min_y, max_x, max_y = block_shape.bounds
            while True:
                x, y = rng.uniform((min_x, min_y), (max_x, max_y))
                if block_shape.intersects(Point(x, y)):
                    break
            return np.array([1, 0, 0, x, y, rng.uniform(0, 4)])

        robot = Variable("?robot", robot_type)
        container = Variable("?container", container_type)
        block = Variable("?block", block_type)
        nsrts.add(NSRT(
            "Grab",
            [robot, container, block],
            {Outside([container, block]), NextToRobot([robot, block]), NotHeld([robot])},
            {Held([robot, block])},
            {NotHeld([robot])},
            {},
            Act,
            [],
            Grab_sampler
        ))

        ## MoveToBlock
        def MoveToBlock_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[_GroundNSRT] = []
        ) -> Array:
            _, _, block = objects
            block_shape = JigsawRelative._get_shape(state, block)
            (block_center_x,), (block_center_y,) = block_shape.envelope.centroid.xy

            while True:
                x, y = rng.uniform(
                    (block_center_x - cls.move_search_margin, block_center_y - cls.move_search_margin),
                    (block_center_x + cls.move_search_margin, block_center_y + cls.move_search_margin),
                )
                if Point(x, y).distance(JigsawRelative._get_shape(state, block)) <= cls.next_to_margin:
                    break

            return np.array([0, 1.0, 0, x, y, rng.uniform(0, 4)])

        robot = Variable("?robot", robot_type)
        container = Variable("?container", container_type)
        block = Variable("?block", block_type)
        nsrts.add(NSRT(
            "MoveToBlock",
            [robot, container, block],
            {NextToRobot([robot, container]), NotHeld([robot])},
            {NextToRobot([robot, block])},
            {},
            {NextToRobot},
            Act,
            [],
            MoveToBlock_sampler,
        ))

        ## MoveToContainer
        def MoveToContainer_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[_GroundNSRT] = []
        ) -> Array:
            _, container, block = objects

            desired_x, desired_y, orientation = JigsawRelative._get_desired_pos(state, block)
            desired_block_shape = JigsawRelative._get_shape(state, block, desired_x, desired_y, orientation)
            (desired_block_center_x,), (desired_block_center_y,) = desired_block_shape.envelope.centroid.xy

            while True:
                x, y = rng.uniform(
                    (desired_block_center_x - cls.move_search_margin, desired_block_center_y - cls.move_search_margin),
                    (desired_block_center_x + cls.move_search_margin, desired_block_center_y + cls.move_search_margin),
                )
                if Point(x, y).distance(desired_block_shape) <= cls.next_to_margin:
                    break

            return np.array([0.0, 1.0, 0.0, x, y, rng.uniform(0, 4)])

        robot = Variable("?robot", robot_type)
        container = Variable("?container", container_type)
        block = Variable("?block", block_type)
        nsrts.add(NSRT(
            "MoveToContainer",
            [robot, container, block],
            {Held([robot, block]), NextToRobot([robot, block])},
            {NextToRobot([robot, container])},
            {},
            {NextToRobot},
            Act,
            [],
            MoveToContainer_sampler,
        ))

        # Place
        def Place_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[_GroundNSRT] = []
        ) -> Array:
            _, _, block = objects

            # Calculating the normalized desired coordinates
            desired_x, desired_y, orientation = JigsawRelative._get_desired_pos(state, block)

            # Sampling the placement coords
            while True:
                sampling_margin = JigsawRelative.sub_cell_margin - cls.placement_margin
                x, y = rng.uniform(
                    (desired_x - sampling_margin, desired_y - sampling_margin),
                    (desired_x + sampling_margin, desired_y + sampling_margin)
                )
                if Point(0, 0).distance(JigsawRelative._get_shape(state, block, x, y, orientation)) <= cls.next_to_margin:
                    break
            return np.array([0, 0, 1, x, y, orientation])

        robot = Variable("?robot", robot_type)
        container = Variable("?container", container_type)
        block = Variable("?block", block_type)
        nsrts.add(NSRT(
            "Place",
            [robot, container, block],
            {NextToRobot([robot, container]), Held([robot, block])},
            {Inside([container, block]), NotHeld([robot])},
            {Outside([container, block]), Held([robot, block])},
            {},
            Act,
            [],
            Place_sampler
        ))

        return nsrts