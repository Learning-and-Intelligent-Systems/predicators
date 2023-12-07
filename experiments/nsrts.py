
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
            global cover_top_ranges, cover_bottom_ranges
            box, shelf, bundle, cover = objects

            box_x, box_y, box_w, box_h = Shelves2DEnv.get_shape_data(state, box)
            shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(state, shelf)

            grasp_x, grasp_y = rng.uniform([box_x + margin, box_y + margin], [box_x + box_w - margin, box_y + box_h - margin])
            new_box_x = rng.uniform(shelf_x + margin, shelf_x + shelf_w - margin - box_w)

            new_box_y_range = [shelf_y + margin - box_h, shelf_y + shelf_h - margin]
            if CoversBottom([cover, bundle]) in goal:
                new_box_y_range[0] += box_h
            elif CoversTop([cover, bundle]) in goal:
                new_box_y_range[1] -= box_h
            else:
                raise ValueError("Expected either CoversTop or CoversBottom in the goal")

            new_box_y = rng.uniform(*new_box_y_range)
            action = np.array([grasp_x, grasp_y, new_box_x - box_x, new_box_y - box_y])
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
    y_margin = (margin if move_to_top else -margin)
    max_x_distance = Shelves2DEnv.cover_sideways_tolerance
    max_y_distance = Shelves2DEnv.cover_max_distance if move_to_top else -Shelves2DEnv.cover_max_distance
    def _MoveCover_sampler(state: State, goal: Set[GroundAtom], rng: np.random.Generator, objects: Sequence[Object]) -> Action:
        cover, bundle = objects

        cover_x, cover_y, cover_w, cover_h = Shelves2DEnv.get_shape_data(state, cover)
        bundle_x, bundle_y, bundle_w, bundle_h = Shelves2DEnv.get_shape_data(state, bundle)

        grasp_x, grasp_y = rng.uniform([cover_x + margin, cover_y + margin], [cover_x + cover_w - margin, cover_y + cover_h - margin])

        desired_x, desired_y = bundle_x, bundle_y + (bundle_h if move_to_top else -cover_h)
        y_bound_1 = desired_y + y_margin
        y_bound_2 = desired_y + max_y_distance - y_margin
        new_cover_x, new_cover_y = rng.uniform([desired_x - max_x_distance + margin, min(y_bound_1, y_bound_2)], [desired_x + max_x_distance - margin, max(y_bound_1, y_bound_2)])

        dx = new_cover_x - cover_x
        dy = new_cover_y - cover_y

        action = np.array([grasp_x, grasp_y, dx, dy])
        return action
    return _MoveCover_sampler