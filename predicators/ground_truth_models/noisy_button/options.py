"""Ground-truth options for the noisy button environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type


class NoisyButtonGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the noisy button environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"noisy_button"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        button_type = types["button"]

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)  # action is simply the parameter

        Click = utils.SingletonParameterizedOption("Click",
                                                   _policy,
                                                   types=[button_type],
                                                   params_space=Box(
                                                       0, 1, (2, )))

        return {Click}
