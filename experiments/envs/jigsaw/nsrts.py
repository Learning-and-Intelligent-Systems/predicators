import itertools
import logging
from typing import Dict, Sequence, Set
import numpy as np
from experiments.envs.jigsaw.env import Jigsaw
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, _GroundNSRT, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Type, Variable
from shapely import Point

__all__ = ['JigsawGroundTruthNSRTFactory']

class JigsawGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    # Settings
    placement_margin = 0.19
    grasp_margin = 0.2

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"jigsaw"}

    @staticmethod
    def get_nsrts(
        env_name: str, types: Dict[str, Type],
        predicates: Dict[str, Predicate],
        options: Dict[str, ParameterizedOption]
    ) -> Set[NSRT]:

        # Types
        container_type = types["container"]
        block_type = types["block"]

        # Prediactes
        Inside = predicates["Inside"]
        Outside = predicates["Outside"]

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
            jigsaw_margin = Jigsaw.sub_cell_margin
            placement_margin = JigsawGroundTruthNSRTFactory.placement_margin
            _, block = objects
            desidred_x, desidred_y, new_orientation = Jigsaw._get_desired_pos(state, block)
            block_shape = Jigsaw._get_shape(state, block).buffer(-JigsawGroundTruthNSRTFactory.grasp_margin, join_style='mitre')

            grasp_min_x, grasp_min_y, grasp_max_x, grasp_max_y = block_shape.bounds
            while True:
                grasp_x, grasp_y = rng.uniform((grasp_min_x, grasp_min_y), (grasp_max_x, grasp_max_y))
                grasp_point = Point(grasp_x, grasp_y)
                if block_shape.intersects(grasp_point):
                    break

            # logging.info((desidred_x, desidred_y, new_orientation, list(state)))

            new_x, new_y = rng.uniform(
                (desidred_x - jigsaw_margin + placement_margin, desidred_y - jigsaw_margin + placement_margin),
                (desidred_x + jigsaw_margin - placement_margin, desidred_y + jigsaw_margin - placement_margin)
            )
            return np.array([grasp_x, grasp_y, new_x, new_y, new_orientation])

        container = Variable("?container", container_type)
        block = Variable("?block", block_type)
        nsrts.add(NSRT(
            "Move",
            [container, block],
            {Outside([container, block])},
            {Inside([container, block])},
            {Outside([container, block])},
            {},
            Act,
            [],
            Move_sampler
        ))

        return nsrts