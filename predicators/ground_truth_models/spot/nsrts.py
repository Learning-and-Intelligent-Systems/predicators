"""Ground-truth NSRTs for the PDDLEnv."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Type
from predicators.utils import null_sampler


class SpotEnvsGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Spot Env."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"spot_grocery_env", "spot_bike_env"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        def move_sampler(state: State, goal: Set[GroundAtom],
                         rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
            del state, goal, rng
            assert len(objs) == 2
            if objs[1].type.name == "bag":
                return np.array([0.5, 0.0, 0.0])
            return np.array([-0.25, 0.0, 0.0])

        def grasp_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs
            return np.array([0])

        def place_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs
            return np.array([0.0])

        env = get_or_create_env(env_name)
        assert isinstance(env, SpotEnv)

        nsrts = set()

        for strips_op in env.strips_operators:
            option = options[strips_op.name]
            if "MoveTo" in strips_op.name:
                nsrt = strips_op.make_nsrt(
                    option=option,
                    option_vars=strips_op.parameters,
                    sampler=move_sampler,
                )
            elif "Grasp" in strips_op.name:
                nsrt = strips_op.make_nsrt(
                    option=option,
                    option_vars=strips_op.parameters,
                    sampler=grasp_sampler,
                )
            elif "Place" in strips_op.name:
                nsrt = strips_op.make_nsrt(
                    option=option,
                    option_vars=strips_op.parameters,
                    sampler=place_sampler,
                )
            else:
                nsrt = strips_op.make_nsrt(
                    option=option,
                    option_vars=strips_op.parameters,
                    sampler=null_sampler,
                )
            nsrts.add(nsrt)

        return nsrts
