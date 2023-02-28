"""Ground-truth options for the cover environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State


class CoverGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the cover environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "cover", "cover_regrasp", "cover_handempty",
            "cover_hierarchical_types"
        }

    @staticmethod
    def get_options(env_name: str) -> Set[ParameterizedOption]:

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        PickPlace = utils.SingletonParameterizedOption("PickPlace",
                                                       _policy,
                                                       params_space=Box(
                                                           0, 1, (1, )))

        return {PickPlace}
