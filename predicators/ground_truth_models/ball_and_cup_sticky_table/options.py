"""Ground-truth options for the ball and cup sticky table environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.ball_and_cup_sticky_table import BallAndCupStickyTableEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class BallAndCupStickyTableGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the sticky table environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"ball_and_cup_sticky_table"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        cup_type = types["cup"]
        ball_type = types["ball"]
        table_type = types["table"]
        # Parameters are move_or_pickplace, obj_type_id, ball_only,
        # absolute x, y actions.
        params_space = Box(
            np.array([
                0.0, 0.0, 0.0, BallAndCupStickyTableEnv.x_lb,
                BallAndCupStickyTableEnv.y_lb
            ]),
            np.array([
                1.0, 3.0, 1.0, BallAndCupStickyTableEnv.x_ub,
                BallAndCupStickyTableEnv.y_ub
            ]))
        robot_type = types["robot"]

        PickBallFromTable = utils.SingletonParameterizedOption(
            # variables: [robot, ball, table]
            "PickBallFromTable",
            cls._create_pass_through_policy(action_space),
            params_space=params_space,
            types=[robot_type, ball_type, cup_type, table_type])

        PickBallFromFloor = utils.SingletonParameterizedOption(
            # variables: [robot, ball]
            "PickBallFromFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type])

        PlaceBallOnTable = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup, table]
            "PlaceBallOnTable",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type, table_type])

        PlaceBallOnFloor = utils.SingletonParameterizedOption(
            # variables: [robot, cup, ball]
            "PlaceBallOnFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cup_type, ball_type])

        PickCupWithoutBallFromTable = utils.SingletonParameterizedOption(
            # variables: [robot, cup, ball, table]
            "PickCupWithoutBallFromTable",
            cls._create_pass_through_policy(action_space),
            params_space=params_space,
            types=[robot_type, cup_type, ball_type, table_type])

        PickCupWithBallFromTable = utils.SingletonParameterizedOption(
            # variables: [robot, cup, ball, table]
            "PickCupWithBallFromTable",
            cls._create_pass_through_policy(action_space),
            params_space=params_space,
            types=[robot_type, cup_type, ball_type, table_type])

        PickCupWithoutBallFromFloor = utils.SingletonParameterizedOption(
            # variables: [robot, cup, ball]
            "PickCupWithoutBallFromFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cup_type, ball_type])

        PickCupWithBallFromFloor = utils.SingletonParameterizedOption(
            # variables: [robot, cup, ball]
            "PickCupWithBallFromFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cup_type, ball_type])

        PlaceCupWithoutBallOnTable = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup, table]
            "PlaceCupWithoutBallOnTable",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type, table_type])

        PlaceCupWithBallOnFloor = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup]
            "PlaceCupWithBallOnFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type])

        PlaceCupWithoutBallOnFloor = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup]
            "PlaceCupWithoutBallOnFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type])

        PlaceBallInCupOnFloor = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup]
            "PlaceBallInCupOnFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type])

        PlaceBallInCupOnTable = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup]
            "PlaceBallInCupOnTable",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type, table_type])

        NavigateToTable = utils.SingletonParameterizedOption(
            # variables: [robot, table]
            "NavigateToTable",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, table_type])

        NavigateToBall = utils.SingletonParameterizedOption(
            # variables: [robot, ball]
            "NavigateToBall",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type])

        NavigateToCup = utils.SingletonParameterizedOption(
            # variables: [robot, cup]
            "NavigateToCup",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cup_type])

        return {
            NavigateToTable,
            PickBallFromTable,
            PickBallFromFloor,
            PlaceBallOnTable,
            PlaceBallOnFloor,
            PickCupWithoutBallFromTable,
            PickCupWithBallFromTable,
            PickCupWithoutBallFromFloor,
            PickCupWithBallFromFloor,  #PlaceCupWithBallOnTable,
            PlaceCupWithoutBallOnTable,
            PlaceCupWithBallOnFloor,
            PlaceCupWithoutBallOnFloor,
            PlaceBallInCupOnFloor,
            PlaceBallInCupOnTable,
            NavigateToBall,
            NavigateToCup
        }

    @classmethod
    def _create_pass_through_policy(cls,
                                    action_space: Box) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            arr = np.array(params, dtype=np.float32)
            arr = np.clip(arr, action_space.low, action_space.high)
            return Action(arr)

        return policy
