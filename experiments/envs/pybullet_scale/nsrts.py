import itertools
import logging
from typing import ClassVar, Dict, Sequence, Set, Tuple, cast

import numpy as np

from experiments.envs.pybullet_scale.env import PyBulletScaleEnv, PyBulletScaleState
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.pybullet_helpers.geometry import Pose
from predicators.structs import NSRT, _GroundNSRT, Array, GroundAtom, Object, ParameterizedOption, Predicate, State, Type, Variable


class PyBulletScaleGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    offset_perturbation: ClassVar[Tuple[float, float]] = (0.015, 0.045)

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_scale"}

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
        scale_type = types["scale"]

        # Prediactes
        OnScale = predicates["OnScale"]
        OnTable = predicates["OnTable"]
        ScaleChecked = predicates["ScaleChecked"]
        ScaleNotChecked = predicates["ScaleNotChecked"]

        # Options
        Move = options["Move"]
        CheckScale = options["CheckScale"]

        # NSRTs
        nsrts = set()

        ## Move
        def Move_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[_GroundNSRT] = [],
        ) -> Array:
            state = cast(PyBulletScaleState, state)
            _, _, block = objects

            bx, by, bz = state[block][:3]
            bqx, bqy, bqz, bqw = state[block][6:10]
            world_from_block: Pose = Pose((bx, by, bz), (bqx, bqy, bqz, bqw))

            relative_block_angle = (np.arctan2(world_from_block.position[1], world_from_block.position[0]) - world_from_block.rpy[2]) % (np.pi * 2)

            offset_x, offset_y, scale_id = state._placement_offsets[-len(skeleton) + 1]
            dx, dy = rng.uniform(-offset_perturbation, offset_perturbation)

            scale_id_onehot = np.zeros(len(PyBulletScaleEnv.scale_poses))
            scale_id_onehot[scale_id] = 1

            return np.hstack([[relative_block_angle, offset_x + dx, offset_y + dy], scale_id_onehot])

        robot = Variable("?robot", robot_type)
        scale = Variable("?scale", scale_type)
        block = Variable("?block", block_type)
        nsrts.add(NSRT(
            "Move",
            [robot, scale, block],
            {OnTable([block]), ScaleNotChecked([scale])},
            {OnScale([block, scale])},
            {OnTable([block])},
            set(),
            Move,
            [robot, scale, block],
            Move_sampler,
        ))

        ## CheckScale
        def CheckScale_sampler(
            state: State,
            goal: Set[GroundAtom],
            rng: np.random.Generator,
            objects: Sequence[Object],
            skeleton: Sequence[_GroundNSRT] = [],
        ) -> Array:
            return np.empty((), dtype=np.float32)

        scale = Variable("?scale", scale_type)
        nsrts.add(NSRT(
            "CheckScale",
            [scale],
            {ScaleNotChecked([scale])},
            {ScaleChecked([scale])},
            {ScaleNotChecked([scale])},
            set(),
            CheckScale,
            [scale],
            CheckScale_sampler,
        ))

        return nsrts