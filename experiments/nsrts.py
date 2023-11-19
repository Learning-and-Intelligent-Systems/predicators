
from typing import ClassVar, Dict, Sequence, Set, Type

import numpy as np
from experiments.shelves2d import Shelves2DEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Action, Array, GroundAtom, NSRTSampler, Object, ParameterizedOption, Predicate, State, Variable

__all__ = ["Shelves2DGroundTruthNSRTFactory"]

class Shelves2DGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the blocks environment."""

    margin: ClassVar[float] = 0.0001

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"shelves2d"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        margin = Shelves2DGroundTruthNSRTFactory.margin

        # Types
        box_type = types["box"]
        shelf_type = types["shelf"]
        bundle_type = types["bundle"]
        cover_type = types["cover"]

        # Prediactes
        GoesInto = predicates["GoesInto"]
        CoverAtStart = predicates["CoverAtStart"]
        Bundles = predicates["Bundles"]
        In = predicates["In"]
        CoverFor = predicates["CoverFor"]
        CoversTop = predicates["CoversFront"]
        CoversBottom = predicates["CoversBack"]

        # Options
        Move = options["Move"]

        nsrts = set()

        # MoveCoverToTop
        cover = Variable("?cover", cover_type)
        bundle = Variable("?bundle", bundle_type)
        nsrts.add(NSRT(
            "MoveCoverToTop",
            [cover, bundle],
            {CoverFor([cover, bundle])},
            {CoversTop([cover, bundle])},
            {CoverAtStart([cover])},
            set(),
            Move,
            set(),
            _MoveCover_sampler_helper(True, margin=margin)
        ))

        # MoveCoverToBottom
        cover = Variable("?cover", cover_type)
        bundle = Variable("?bundle", bundle_type)
        nsrts.add(NSRT(
            "MoveCoverToBottom",
            [cover, bundle],
            {CoverFor([cover, bundle])},
            {CoversBottom([cover, bundle])},
            {CoverAtStart([cover])},
            set(),
            Move,
            set(),
            _MoveCover_sampler_helper(False, margin=margin)
        ))

        # InsertBox
        box = Variable("?box", box_type)
        shelf = Variable("?shelf", shelf_type)
        bundle = Variable("?bundle", bundle_type)
        cover = Variable("?cover", cover_type)

        def InsertBox_sampler(state: State, goal: Set[GroundAtom], rng: np.random.Generator, objects: Sequence[Object]) -> Array:
            box, shelf, bundle, cover = objects

            box_x, box_y, box_w, box_h = Shelves2DEnv.get_shape_data(state, box)
            shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(state, shelf)

            grasp_x, grasp_y = rng.uniform([box_x + margin, box_y + margin], [box_x + box_w - margin, box_y + box_h - margin])
            dx = rng.uniform(shelf_x + margin - box_x, shelf_x + shelf_w - margin - box_w - box_x)

            dy_range = [shelf_y + margin - box_h - box_y, shelf_y + shelf_h - margin - box_y]
            if CoversBottom([cover, bundle]) in goal:
                dy_range[0] = shelf_y + margin - box_y
            if CoversTop([cover, bundle]) in goal:
                dy_range[1] = shelf_y + shelf_h - margin - box_h - box_y

            dy = rng.uniform(*dy_range)
            action = np.array([grasp_x, grasp_y, dx, dy])
            return action

        nsrts.add(NSRT(
            "InsertBox",
            [box, shelf, bundle, cover],
            {GoesInto([box, shelf]), Bundles([bundle, shelf]), CoverFor([cover, bundle]), CoverAtStart([cover])},
            {In([box, shelf])},
            set(),
            set(),
            Move,
            set(),
            InsertBox_sampler
        ))

        return nsrts

def _MoveCover_sampler_helper(move_to_top: bool, margin = 0.0001) -> NSRTSampler:
    def _MoveCover_sampler(state: State, goal: Set[GroundAtom], rng: np.random.Generator, objects: Sequence[Object]) -> Action:
        cover, bundle = objects

        cover_x, cover_y, cover_w, cover_h = Shelves2DEnv.get_shape_data(state, cover)
        bundle_x, bundle_y, bundle_w, bundle_h = Shelves2DEnv.get_shape_data(state, bundle)

        grasp_x, grasp_y = rng.uniform([cover_x, cover_y], [cover_x + cover_w, cover_y + cover_h])

        dy = bundle_y - cover_y + (bundle_h + margin if move_to_top else -cover_h - margin)
        dx = bundle_x - cover_x

        action = np.array([grasp_x, grasp_y, dx, dy])
        return action
    return _MoveCover_sampler