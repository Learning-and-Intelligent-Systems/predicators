"""Ground-truth NSRTs for the PDDLEnv."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotRearrangementEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.spot_utils.utils import get_spot_home_pose
from predicators.structs import NSRT, Array, GroundAtom, NSRTSampler, Object, \
    ParameterizedOption, Predicate, State, Type
from predicators.utils import null_sampler


def _move_to_body_view_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del state, goal, rng  # randomization coming soon

    home_pose = get_spot_home_pose()
    # Currently assume that the robot is facing the surface in its home pose.
    # Soon, we will change this to actually sample angles of approach and do
    # collision detection.
    approach_angle = home_pose.angle - np.pi
    approach_dist = 1.2

    # For the cup on the table, we need to be further back to actually
    # view it.
    if len(objs) == 2 and objs[1].name == "cup":
        approach_dist = 1.75
        approach_angle = home_pose.angle - (np.pi / 2)

    return np.array([approach_dist, approach_angle])


def _move_to_hand_view_object_sampler(state: State, goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del state, goal, objs, rng  # randomization coming soon

    home_pose = get_spot_home_pose()
    # Currently assume that the robot is facing the surface in its home pose.
    # Soon, we will change this to actually sample angles of approach and do
    # collision detection.
    approach_angle = home_pose.angle - np.pi
    approach_dist = 1.2
    return np.array([approach_dist, approach_angle])


def _move_to_reach_object_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are relative distance, dyaw (to the object you're moving to).
    del state, goal, rng  # randomization coming soon

    home_pose = get_spot_home_pose()

    # Currently assume that the robot is facing the surface in its home pose.
    # Soon, we will change this to actually sample angles of approach and do
    # collision detection.
    approach_angle = home_pose.angle - np.pi

    if len(objs) == 2 and objs[1].name == "cup":
        approach_angle = home_pose.angle - (np.pi / 2)

    # NOTE: closer than move_to_view. Important for placing.
    return np.array([0.8, approach_angle])


def _pick_object_from_top_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Not parameterized; may change in the future.
    return null_sampler(state, goal, rng, objs)


def _place_object_on_top_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dz (to surface objects center).
    del state, goal, objs, rng  # randomization coming soon
    return np.array([0.0, 0.0, 0.05])


def _drop_object_inside_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dz to the center of the top of the
    # container.
    del state, goal, rng  # randomization coming soon

    drop_height = 0.5
    dx = 0.0
    if len(objs) == 4 and objs[2].name == "cup":
        drop_height = 0.15
        dx = 0.08  # we benefit from dropping more forward in the x!

    return np.array([dx, 0.0, drop_height])


def _drag_to_unblock_object_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objs: Sequence[Object]) -> Array:
    # Parameters are relative dx, dy, dyaw to move while holding.
    del state, goal, objs, rng  # randomization coming soon
    return np.array([-1.25, 0.0, np.pi / 3])


def _sweep_into_container_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
    # Parameters are start dx, start dy.
    del state, goal, objs, rng  # randomization coming soon
    return np.array([0.0, 0.25])


def _prepare_sweeping_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
    # Parameters are dx, dy, yaw w.r.t. the target object.
    del state, goal, objs, rng  # randomization coming soon

    # Currently assume that the robot is facing the surface in its home pose.
    # Soon, we will change this to actually sample angles of approach and do
    # collision detection.
    home_pose = get_spot_home_pose()

    return np.array([-0.8, -0.4, home_pose.angle])


class SpotCubeEnvGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Spot Env."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "spot_cube_env", "spot_soda_table_env", "spot_soda_bucket_env",
            "spot_soda_chair_env", "spot_soda_sweep_env",
            "spot_ball_and_cup_sticky_table_env"
        }

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        env = get_or_create_env(env_name)
        assert isinstance(env, SpotRearrangementEnv)

        nsrts = set()

        operator_name_to_sampler: Dict[str, NSRTSampler] = {
            "MoveToHandViewObject": _move_to_hand_view_object_sampler,
            "MoveToBodyViewObject": _move_to_body_view_object_sampler,
            "MoveToReachObject": _move_to_reach_object_sampler,
            "PickObjectFromTop": _pick_object_from_top_sampler,
            "PlaceObjectOnTop": _place_object_on_top_sampler,
            "DropObjectInside": _drop_object_inside_sampler,
            "DropObjectInsideContainerOnTop": _drop_object_inside_sampler,
            "DragToUnblockObject": _drag_to_unblock_object_sampler,
            "SweepIntoContainer": _sweep_into_container_sampler,
            "PrepareContainerForSweeping": _prepare_sweeping_sampler,
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
