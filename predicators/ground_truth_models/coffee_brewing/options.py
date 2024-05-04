"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class CoffeeBrewingGroundTruthOptionFactory(GroundTruthOptionFactory):
    """TODO"""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"coffee_brewing"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        object_type = types["object"]
        mug_type = types["mug"]
        robot_gripper_type = types["robot_gripper"]
        hot_plate_type = types["hot_plate"]
        coffee_pot_type = types["coffee_pot"]

        Pick = utils.SingletonParameterizedOption(
            "pick",
            cls._create_dummy_policy(action_space),
            types=[object_type, robot_gripper_type])

        PlaceOn = utils.SingletonParameterizedOption(
            "place_on",
            cls._create_dummy_policy(action_space),
            types=[object_type, object_type, robot_gripper_type])
        
        SwitchOn = utils.SingletonParameterizedOption(
            "switch_on",
            cls._create_dummy_policy(action_space),
            types=[hot_plate_type, robot_gripper_type])
        
        Pour = utils.SingletonParameterizedOption(
            "pour",
            cls._create_dummy_policy(action_space),
            types=[coffee_pot_type, mug_type, robot_gripper_type])

        return {Pick, PlaceOn, SwitchOn, Pour}

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
