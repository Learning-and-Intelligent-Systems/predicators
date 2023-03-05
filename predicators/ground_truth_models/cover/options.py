"""Ground-truth options for the cover environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type


class CoverGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cover environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "cover", "cover_regrasp", "cover_handempty",
            "cover_hierarchical_types"
        }

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        PickPlace = utils.SingletonParameterizedOption("PickPlace",
                                                       _policy,
                                                       params_space=Box(
                                                           0, 1, (1, )))

        return {PickPlace}


class CoverTypedOptionsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cover_typed_options environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cover_typed_options"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        block_type = types["block"]
        target_type = types["target"]

        def _Pick_policy(s: State, m: Dict, o: Sequence[Object],
                         p: Array) -> Action:
            del m  # unused
            # The pick parameter is a RELATIVE position, so we need to
            # add the pose of the object.
            pick_pose = s.get(o[0], "pose") + p[0]
            pick_pose = min(max(pick_pose, 0.0), 1.0)
            return Action(np.array([pick_pose], dtype=np.float32))

        Pick = utils.SingletonParameterizedOption("Pick",
                                                  _Pick_policy,
                                                  types=[block_type],
                                                  params_space=Box(
                                                      -0.1, 0.1, (1, )))

        def _Place_policy(state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        Place = utils.SingletonParameterizedOption(
            "Place",
            _Place_policy,  # use the parent class's policy
            types=[target_type],
            params_space=Box(0, 1, (1, )))

        return {Pick, Place}


# class CoverMultiStepOptionsGroundTruthOptionFactory(GroundTruthOptionFactory):
#     """Ground-truth options for the cover_multistep_options environment."""

#     @classmethod
#     def get_env_names(cls) -> Set[str]:
#         return {"cover_multistep_options"}

#     @classmethod
#     def get_options(cls, env_name: str, types: Dict[str, Type],
#                     predicates: Dict[str, Predicate],
#                     action_space: Box) -> Set[ParameterizedOption]:
#         import ipdb; ipdb.set_trace()

# class PybulletCoverGroundTruthOptionFactory(GroundTruthOptionFactory):
#     """Ground-truth options for the PyBullet cover environment."""

#     @classmethod
#     def get_env_names(cls) -> Set[str]:
#         return {"pybullet_cover"}

#     @classmethod
#     def get_options(cls, env_name: str, types: Dict[str, Type],
#                     predicates: Dict[str, Predicate],
#                     action_space: Box) -> Set[ParameterizedOption]:
#         import ipdb; ipdb.set_trace()
