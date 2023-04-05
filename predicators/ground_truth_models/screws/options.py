"""Ground-truth options for the screws environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.screws import ScrewsEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class ScrewsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the screws environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"screws"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        gripper_type = types["gripper"]
        screw_type = types["screw"]
        receptacle_type = types["receptacle"]

        MoveToScrew = utils.SingletonParameterizedOption(
            # variables: [robot, screw to pick up].
            # params: [].
            "MoveToScrew",
            cls._create_move_to_screw_policy(),
            types=[gripper_type, screw_type])

        MoveToReceptacle = utils.SingletonParameterizedOption(
            # variables: [robot, receptacle].
            # params: [].
            "MoveToReceptacle",
            cls._create_move_to_receptacle_policy(),
            types=[gripper_type, receptacle_type, screw_type])

        MagnetizeGripper = utils.SingletonParameterizedOption(
            # variables: [robot].
            # params: [].
            "MagnetizeGripper",
            cls._create_magnetize_gripper_policy(),
            types=[gripper_type])

        DemagnetizeGripper = utils.SingletonParameterizedOption(
            # variables: [robot].
            # params: [].
            "DemagnetizeGripper",
            cls._create_demagnetize_gripper_policy(),
            types=[gripper_type])

        return {
            MoveToScrew, MoveToReceptacle, MagnetizeGripper, DemagnetizeGripper
        }

    @classmethod
    def _create_move_to_screw_policy(cls) -> ParameterizedPolicy:
        """Policy to return an action that moves to a position above a
        particular screw such that the screw can be grasped."""

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused.
            robot, screw = objects
            screw_x = state.get(screw, "pose_x")
            screw_y = state.get(screw, "pose_y")

            target_x = screw_x
            target_y = screw_y + (ScrewsEnv.screw_height /
                                  2.0) + (ScrewsEnv.magnetic_field_dist / 2.0)

            current_x = state.get(robot, "pose_x")
            current_y = state.get(robot, "pose_y")

            return Action(
                np.array([target_x - current_x, target_y - current_y, 0.0],
                         dtype=np.float32))

        return policy

    @classmethod
    def _create_move_to_receptacle_policy(cls) -> ParameterizedPolicy:
        """Policy to return an action that moves to a position above the
        receptacle."""

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:

            del memory, params  # unused.
            robot, receptacle, _ = objects
            receptacle_x = state.get(receptacle, "pose_x")
            receptacle_y = state.get(receptacle, "pose_y")

            target_x = receptacle_x
            target_y = receptacle_y + (ScrewsEnv.magnetic_field_dist)

            current_x = state.get(robot, "pose_x")
            current_y = state.get(robot, "pose_y")

            return Action(
                np.array([target_x - current_x, target_y - current_y, 1.0],
                         dtype=np.float32))

        return policy

    @classmethod
    def _create_magnetize_gripper_policy(cls) -> ParameterizedPolicy:
        """Policy to return an action that magnetizes the gripper at its
        current position."""

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused.
            return Action(np.array([0.0, 0.0, 1.0], dtype=np.float32))

        return policy

    @classmethod
    def _create_demagnetize_gripper_policy(cls) -> ParameterizedPolicy:
        """Policy to return an action that demagnetizes the gripper at its
        current position."""

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects, params  # unused.
            return Action(np.array([0.0, 0.0, 0.0], dtype=np.float32))

        return policy
