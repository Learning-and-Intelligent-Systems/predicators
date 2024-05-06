"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class KitchenTestMakingGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the tea making environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen_test"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        object_type = types["object"]
        kettle_type = types["kettle"]
        burner_type = types["stove_burner"]
        robot_gripper_type = types["robot_gripper"]

        PickPlace = utils.SingletonParameterizedOption(
            "pick_place",
            cls._create_dummy_policy(action_space),
            types=[kettle_type, burner_type, burner_type, robot_gripper_type])

        OpenDoor = utils.SingletonParameterizedOption(
            "open_door",
            cls._create_dummy_policy(action_space),
            types=[object_type, robot_gripper_type])
        
        TurnOn = utils.SingletonParameterizedOption(
            "turn_on",
            cls._create_dummy_policy(action_space),
            types=[object_type, robot_gripper_type])

        return {PickPlace, OpenDoor, TurnOn}

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
