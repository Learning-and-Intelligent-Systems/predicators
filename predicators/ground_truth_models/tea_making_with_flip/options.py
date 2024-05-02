"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class TeaMakingWithFlipGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the tea making environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"tea_making_with_flip"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        object_type = types["object"]
        mug_type = types["mug"]
        hand_type = types["hand"]
        milk_carton_type = types["milk_carton"]

        Pick = utils.SingletonParameterizedOption(
            "pick",
            cls._create_dummy_policy(action_space),
            types=[object_type, hand_type])

        PlaceInCup = utils.SingletonParameterizedOption(
            "place_in",
            cls._create_dummy_policy(action_space),
            types=[object_type, mug_type])

        FlipUpright = utils.SingletonParameterizedOption(
            "flip",
            cls._create_dummy_policy(action_space),
            types=[mug_type, hand_type])

        PourInMilk = utils.SingletonParameterizedOption(
            "pour",
            cls._create_dummy_policy(action_space),
            types=[mug_type, milk_carton_type, hand_type])

        return {Pick, PlaceInCup, FlipUpright, PourInMilk}

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
