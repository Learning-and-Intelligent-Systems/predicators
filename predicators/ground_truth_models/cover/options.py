"""Ground-truth options for the cover environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State


class _CoverGroundTruthOptionFactory(GroundTruthOptionFactory):

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cover"}

    @staticmethod
    def get_options(env_name: str) -> Set[ParameterizedOption]:
        assert env_name == "cover"

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        PickPlace = utils.SingletonParameterizedOption("PickPlace",
                                                       _policy,
                                                       params_space=Box(
                                                           0, 1, (1, )))

        return {PickPlace}
