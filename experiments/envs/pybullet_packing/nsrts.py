import itertools
import logging
from typing import ClassVar, Dict, Sequence, Set, Tuple, cast

import numpy as np

from experiments.envs.pybullet_packing.env import PyBulletPackingState
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, _GroundNSRT, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Type, Variable


class PyBulletPackingGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    offset_perturbation: ClassVar[Tuple[float, float]] = (0.015, 0.015)

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_packing"}

    @classmethod
    def get_nsrts(
        cls, env_name: str, types: Dict[str, Type],
        predicates: Dict[str, Predicate],
        options: Dict[str, ParameterizedOption]
    ) -> Set[NSRT]:
        offset_perturbation = np.array(cls.offset_perturbation)

        # Types
        robot_type = types["robot"]
        block_type = types["block"]
        box_type = types["box"]

        # Prediactes
        OnTable = predicates["OnTable"]
        InBox = predicates["InBox"]

        # Options
        Move = options["Move"]

        def Move_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[_GroundNSRT] = [],
        ) -> Array:
            assert skeleton
            state = cast(PyBulletPackingState, state)

            _, _, block = objects
            block_h = state.get(block, "h")
            grasp_h = rng.uniform(-0.15, 0.35)
            offset_x, offset_y = state._placement_offsets[-len(skeleton)]
            dx, dy = rng.uniform(-offset_perturbation, offset_perturbation)

            return np.array(list(itertools.chain([grasp_h], [offset_x + dx, offset_y + dy])))

        robot = Variable("?robot", robot_type)
        box = Variable("?box", box_type)
        block = Variable("?block", block_type)
        nsrt = NSRT(
            "Move",
            [robot, box, block],
            {OnTable([block])},
            {InBox([block, box])},
            {OnTable([block])},
            {},
            Move,
            [robot, box, block],
            Move_sampler,
        )

        return {nsrt}