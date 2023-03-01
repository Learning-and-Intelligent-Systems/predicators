"""Ground-truth options for the sandwich environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.sandwich import SandwichEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class SandwichGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the sandwich environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"sandwich", "sandwich_clear"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        ingredient_type = types["ingredient"]
        board_type = types["board"]

        Pick = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            "Pick",
            cls._create_pick_policy(action_space),
            types=[robot_type, ingredient_type])

        Stack = utils.SingletonParameterizedOption(
            # variables: [robot, object on which to stack held-object]
            "Stack",
            cls._create_stack_policy(action_space),
            types=[robot_type, ingredient_type])

        PutOnBoard = utils.SingletonParameterizedOption(
            # variables: [robot, board]
            "PutOnBoard",
            cls._create_put_on_board_policy(action_space),
            types=[robot_type, board_type])

        return {Pick, Stack, PutOnBoard}

    @classmethod
    def _create_pick_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, ing = objects
            pose = np.array([
                state.get(ing, "pose_x"),
                state.get(ing, "pose_y"),
                state.get(ing, "pose_z")
            ])
            arr = np.r_[pose, 0.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_stack_policy(cls, action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:

            del memory, params  # unused
            _, ing = objects
            pose = np.array([
                state.get(ing, "pose_x"),
                state.get(ing, "pose_y"),
                state.get(ing, "pose_z")
            ])
            relative_grasp = np.array([
                0.,
                0.,
                SandwichEnv.ingredient_thickness,
            ])
            arr = np.r_[pose + relative_grasp, 1.0].astype(np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy

    @classmethod
    def _create_put_on_board_policy(cls,
                                    action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, board = objects
            x = state.get(board, "pose_x")
            y = state.get(board, "pose_y")
            z = SandwichEnv.table_height + SandwichEnv.board_thickness
            arr = np.array([x, y, z, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy
