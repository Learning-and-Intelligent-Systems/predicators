"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class TeaMakingGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the tea making environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"iced_tea_making"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        object_type = types["object"]
        cup_type = types["cup"]
        hand_type = types["hand"]

        Pick = utils.SingletonParameterizedOption(
            # variables: [teabag to pick]
            # params: []
            "pick",
            cls._create_dummy_policy(action_space),
            types=[object_type, hand_type])

        PlaceInCup = utils.SingletonParameterizedOption(
            # variables: [object to place, thing to place in]
            # params: []
            "place_in",
            cls._create_dummy_policy(action_space),
            types=[object_type, cup_type])

        return {Pick, PlaceInCup}

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
