"""Ground-truth options for the burger making environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class BurgerMakingGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the burger making environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"burger_making"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        object_type = types["object"]
        robot_type = types["robot"]
        cutting_board_type = types["cutting_board"]
        grill_type = types["grill"]
        tomato_type = types["tomato"]
        cheese_type = types["cheese"]
        patty_type = types["patty"]
        bottom_bun_type = types["bottom_bun"]
        top_bun_type = types["top_bun"]

        Pick = utils.SingletonParameterizedOption(
            "pick",
            cls._create_dummy_policy(action_space),
            types = [object_type, robot_type]
        )

        Place = utils.SingletonParameterizedOption(
            "place",
            cls._create_dummy_policy(action_space),
            types = [object_type, object_type, robot_type]
        )

        Cook = utils.SingletonParameterizedOption(
            "cook",
            cls._create_dummy_policy(action_space),
            types = [object_type, grill_type, robot_type]
        )

        Slice = utils.SingletonParameterizedOption(
            "slice",
            cls._create_dummy_policy(action_space),
            types = [object_type, cutting_board_type, robot_type]
        )

        return {Pick, Place, Cook, Slice}


    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy