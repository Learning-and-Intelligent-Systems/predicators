"""Ground-truth options for the (non-pybullet) blocks environment."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class BagelMakingGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the tea making environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"bagel_making"}

    @classmethod
    def get_options(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            action_space: Box) -> Set[ParameterizedOption]:  # pragma: no cover

        del env_name, predicates  # unused.

        # object_type = types["object"]
        gripper_type = types["robot_gripper"]
        oven_type = types["oven"]
        bagel_type = types["bagel"]

        OpenDoor = utils.SingletonParameterizedOption(
            "open_door",
            cls._create_dummy_policy(action_space),
            types=[oven_type, gripper_type])
        
        CloseDoor = utils.SingletonParameterizedOption(
            "close_door",
            cls._create_dummy_policy(action_space),
            types=[oven_type, gripper_type])
        
        PullTrayOutside = utils.SingletonParameterizedOption(
            "pull_tray_outside",
            cls._create_dummy_policy(action_space),
            types=[oven_type, gripper_type])
        
        PushTrayInside = utils.SingletonParameterizedOption(
            "push_tray_inside",
            cls._create_dummy_policy(action_space),
            types=[oven_type, gripper_type])

        Pick = utils.SingletonParameterizedOption(
            "pick",
            cls._create_dummy_policy(action_space),
            types=[bagel_type, gripper_type])

        PlaceInTray = utils.SingletonParameterizedOption(
            "place_in_tray",
            cls._create_dummy_policy(action_space),
            types=[bagel_type, oven_type, gripper_type])

        return {OpenDoor, CloseDoor, PullTrayOutside, PushTrayInside, Pick, PlaceInTray}

    @classmethod
    def _create_dummy_policy(
            cls, action_space: Box) -> ParameterizedPolicy:  # pragma: no cover
        del action_space  # unused

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params
            raise ValueError("Shouldn't be attempting to run this policy!")

        return policy
