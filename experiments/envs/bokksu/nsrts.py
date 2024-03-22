import logging
from typing import Dict, Sequence, Set
import numpy as np
from experiments.envs.bokksu.env import Bokksu
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, _GroundNSRT, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Type, Variable

__all__ = ['BokksuGroundTruthNSRTFactory']

class BokksuGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    # Settings
    margin = 0.05
    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"bokksu"}

    @staticmethod
    def get_nsrts(
        env_name: str, types: Dict[str, Type],
        predicates: Dict[str, Predicate],
        options: Dict[str, ParameterizedOption]
    ) -> Set[NSRT]:

        # Types
        bokksu_type = types["bokksu"]
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
            margin = BokksuGroundTruthNSRTFactory.margin
            bokksu, block = objects
            desidred_x, desidred_y, new_orientation = Bokksu._get_desired_pos(state, block)
            x, y, w, h = Bokksu._get_block_shape(
                state.get(block, "x"),
                state.get(block, "y"),
                state.get(block, "sub_block_0"),
                state.get(block, "sub_block_1"),
                state.get(block, "orientation")
            )
            grasp_x, grasp_y = rng.uniform((x + margin, y + margin), (x + w - margin, y + h - margin))
            new_x, new_y = rng.uniform(
                (desidred_x + margin, desidred_y + margin),
                (desidred_x + Bokksu.sub_cell_margin * 2 - margin, desidred_y + Bokksu.sub_cell_margin * 2 - margin)
            )
            return np.array([grasp_x, grasp_y, new_x, new_y, rng.uniform(0, 0.5) + new_orientation * 0.5])

        bokksu = Variable("?bokksu", bokksu_type)
        block = Variable("?block", block_type)
        nsrts.add(NSRT(
            "Move",
            [bokksu, block],
            {Outside([bokksu, block])},
            {Inside([bokksu, block])},
            {Outside([bokksu, block])},
            {},
            Act,
            [],
            Move_sampler
        ))

        return nsrts