"""Ground-truth NSRTs for the PDDLEnv."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.spot_utils.utils import get_spot_home_pose
from predicators.structs import NSRT, Array, GroundAtom, NSRTSampler, Object, \
    ParameterizedOption, Predicate, State, Type
from predicators.utils import null_sampler


def _move_to_tool_on_surface_sampler(state: State, goal: Set[GroundAtom],
                                     rng: np.random.Generator,
                                     objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del state, goal, objs, rng  # randomization coming soon

    # Currently assume that the robot is facing the surface in its home pose.
    # Soon, we will change this to actually sample angles of approach and do
    # collision detection.
    home_pose = get_spot_home_pose()
    approach_angle = home_pose.angle - np.pi

    return np.array([1.20, approach_angle])


def _move_to_tool_on_floor_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del state, goal, objs, rng  # randomization coming soon

    # Currently assume that the robot is facing the surface in its home pose.
    # Soon, we will change this to actually sample angles of approach and do
    # collision detection.
    home_pose = get_spot_home_pose()
    approach_angle = home_pose.angle - np.pi

    return np.array([1.20, approach_angle])


def _move_to_surface_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the surface you're moving to).
    del state, goal, objs, rng  # randomization coming soon

    # Currently assume that the robot is facing the surface in its home pose.
    # Soon, we will change this to actually sample angles of approach and do
    # collision detection.
    home_pose = get_spot_home_pose()
    approach_angle = home_pose.angle - np.pi

    return np.array([1.20, approach_angle])


def _grasp_tool_from_surface_sampler(state: State, goal: Set[GroundAtom],
                                     rng: np.random.Generator,
                                     objs: Sequence[Object]) -> Array:
    # Not parameterized; may change in the future.
    return null_sampler(state, goal, rng, objs)


def _grasp_tool_from_floor_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
    # Not parameterized; may change in the future.
    return null_sampler(state, goal, rng, objs)


def _place_tool_on_surface_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dz (to surface objects center).
    del state, goal, objs, rng  # randomization coming soon
    return np.array([0.0, 0.0, 0.25])


def _place_tool_on_floor_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
    # Not parameterized; may change in the future.
    return null_sampler(state, goal, rng, objs)


class SpotCubeEnvGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Spot Env."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"spot_cube_env"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        env = get_or_create_env(env_name)
        assert isinstance(env, SpotEnv)

        nsrts = set()

        operator_name_to_sampler: Dict[str, NSRTSampler] = {
            "MoveToToolOnSurface": _move_to_tool_on_surface_sampler,
            "MoveToToolOnFloor": _move_to_tool_on_floor_sampler,
            "MoveToSurface": _move_to_surface_sampler,
            "GraspToolFromSurface": _grasp_tool_from_surface_sampler,
            "GraspToolFromFloor": _grasp_tool_from_floor_sampler,
            "PlaceToolOnSurface": _place_tool_on_surface_sampler,
            "PlaceToolOnFloor": _place_tool_on_floor_sampler,
        }

        for strips_op in env.strips_operators:
            sampler = operator_name_to_sampler[strips_op.name]
            option = options[strips_op.name]
            nsrt = strips_op.make_nsrt(
                option=option,
                option_vars=strips_op.parameters,
                sampler=sampler,
            )
            nsrts.add(nsrt)

        return nsrts
