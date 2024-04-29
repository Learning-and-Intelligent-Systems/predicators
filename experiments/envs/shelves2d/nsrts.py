
from typing import ClassVar, Dict, Sequence, Set, Type

import numpy as np
from typing import cast
from experiments.envs.shelves2d.env import Shelves2DEnv, SimulatorState
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Action, Array, GroundAtom, NSRTSampler, Object, ParameterizedOption, Predicate, State, Variable

__all__ = ["Shelves2DGroundTruthNSRTFactory"]


class Shelves2DGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the blocks environment."""

    margin: ClassVar[float] = 0.1

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
        Act = options["Act"]

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
            Act,
            [],
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
            Act,
            [],
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

            box_x, box_y, box_w, box_h = Shelves2DEnv.get_shape_data(
                state, box)
            shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(
                state, shelf)

            y_pos_range = [-box_h + margin, shelf_h - margin]
            if CoversBottom([cover, bundle]) in goal:
                y_pos_range[0] += box_h
            elif CoversTop([cover, bundle]) in goal:
                y_pos_range[1] -= box_h
            else:
                raise ValueError(
                    "Expected either CoversTop or CoversBottom in the goal")

            return np.array([
                rng.uniform(box_x + margin, box_x + box_w - margin),
                rng.uniform(box_y + margin, box_y + box_h - margin),
                rng.uniform(shelf_x + margin, shelf_x + shelf_w - margin),
                rng.uniform(shelf_y + margin, shelf_y + shelf_h - margin),
                rng.uniform(margin, shelf_w - box_w - margin),
                rng.uniform(*y_pos_range),
            ])

        nsrts.add(NSRT(
            "InsertBox",
            [box, shelf, bundle, cover],
            {GoesInto([box, shelf]), Bundles([bundle, shelf]),
             CoverFor([cover, bundle]), CoverAtStart([cover])},
            {In([box, shelf])},
            set(),
            set(),
            Act,
            [],
            InsertBox_sampler
        ))

        return nsrts


def _MoveCover_sampler_helper(move_to_top: bool, margin=0.1) -> NSRTSampler:
    def _MoveCover_sampler(
        state: State,
        goal: Set[GroundAtom],
        rng: np.random.Generator,
        objects: Sequence[Object],
        skeleton: Sequence[NSRT] = [],
    ) -> Action:
        cover, bundle = objects
        aux_data = cast(SimulatorState, state.simulator_state)

        cover_x, cover_y, cover_w, cover_h = Shelves2DEnv.get_shape_data(
            state, cover)
        shelf_x, shelf_y, shelf_w, shelf_h = Shelves2DEnv.get_shape_data(
            state, aux_data.leftmost_shelf)

        offset_x = rng.uniform(-Shelves2DEnv.cover_sideways_tolerance +
                               margin, Shelves2DEnv.cover_sideways_tolerance - margin)
        if move_to_top:
            offset_y = rng.uniform(
                shelf_h + margin, shelf_h + Shelves2DEnv.cover_max_distance - margin)
        else:
            offset_y = rng.uniform(-cover_h -
                                   Shelves2DEnv.cover_max_distance + margin, -cover_h - margin)

        return np.array([
            rng.uniform(cover_x + margin, cover_x + cover_w - margin),
            rng.uniform(cover_y + margin, cover_y + cover_h - margin),
            rng.uniform(shelf_x + margin, shelf_x + shelf_w - margin),
            rng.uniform(shelf_y + margin, shelf_y + shelf_h - margin),
            offset_x, offset_y
        ])
    return _MoveCover_sampler
