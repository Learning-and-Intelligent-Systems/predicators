
from typing import ClassVar, Dict, Sequence, Set, Type

import numpy as np
from experiments.envs.shelves2d.env import Shelves2DEnv
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
        CoversTop = predicates["CoversTop"]
        CoversBottom = predicates["CoversBottom"]

        # Options
        MoveCoverToTop = options["MoveCoverToTop"]
        MoveCoverToBottom = options["MoveCoverToBottom"]
        MoveBox = options["MoveBox"]

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
            MoveCoverToTop,
            [cover, bundle],
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
            MoveCoverToBottom,
            [cover, bundle],
            _MoveCover_sampler_helper(False, margin=margin)
        ))

        # InsertBox
        box = Variable("?box", box_type)
        shelf = Variable("?shelf", shelf_type)
        bundle = Variable("?bundle", bundle_type)
        cover = Variable("?cover", cover_type)

        def InsertBox_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[NSRT] = [],
        ) -> Array:
            global cover_top_ranges, cover_bottom_ranges
            box, shelf, bundle, cover = objects

            box_x, box_y, box_w, box_h = Shelves2DEnv.get_shape_data(state, box)
            shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(state, shelf)

            offset_x = rng.uniform(margin, shelf_w - box_w - margin)

            offset_y_range = [margin - box_h, shelf_h - margin]
            if CoversBottom([cover, bundle]) in goal:
                offset_y_range[0] += box_h
            elif CoversTop([cover, bundle]) in goal:
                offset_y_range[1] -= box_h
            else:
                raise ValueError("Expected either CoversTop or CoversBottom in the goal")
            offset_y = rng.uniform(*offset_y_range)

            return np.array([offset_x, offset_y])

        nsrts.add(NSRT(
            "InsertBox",
            [box, shelf, bundle, cover],
            {GoesInto([box, shelf]), Bundles([bundle, shelf]), CoverFor([cover, bundle]), CoverAtStart([cover])},
            {In([box, shelf])},
            set(),
            set(),
            MoveBox,
            [box, shelf],
            InsertBox_sampler
        ))

        return nsrts

def _MoveCover_sampler_helper(move_to_top: bool, margin = 0.0001) -> NSRTSampler:
    def _MoveCover_sampler(
        state: State,
        goal: Set[GroundAtom],
        rng: np.random.Generator,
        objects: Sequence[Object],
        skeleton: Sequence[NSRT] = [],
    ) -> Action:
        cover, bundle = objects

        cover_x, cover_y, cover_w, cover_h = Shelves2DEnv.get_shape_data(state, cover)
        bundle_x, bundle_y, bundle_w, bundle_h = Shelves2DEnv.get_shape_data(state, bundle)

        offset_x = rng.uniform(-Shelves2DEnv.cover_sideways_tolerance + margin, Shelves2DEnv.cover_sideways_tolerance - margin)
        if move_to_top:
            offset_y = rng.uniform(margin, Shelves2DEnv.cover_max_distance - margin)
        else:
            offset_y = rng.uniform(-Shelves2DEnv.cover_max_distance + margin, -margin)

        return np.array([offset_x, offset_y])
    return _MoveCover_sampler