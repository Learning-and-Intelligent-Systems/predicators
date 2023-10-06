"""Ground-truth options for the sticky table environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.sticky_table import StickyTableEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class StickyTableGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the sticky table environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"sticky_table", "sticky_table_tricky_floor"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        cube_type = types["cube"]
        cup_type = types["cup"]
        ball_type = types["ball"]
        table_type = types["table"]
        # Parameters are move_or_pickplace, obj_type_id, absolute x, y actions.
        params_space = Box(
            np.array([0.0, 0.0, StickyTableEnv.x_lb, StickyTableEnv.y_lb]),
            np.array([1.0, 2.0, StickyTableEnv.x_ub, StickyTableEnv.y_ub]))
        robot_type = types["robot"]

        PickCubeFromTable = utils.SingletonParameterizedOption(
            # variables: [robot, cube, table]
            "PickCubeFromTable",
            cls._create_pass_through_policy(action_space),
            params_space=params_space,
            types=[robot_type, cube_type, table_type])

        PickCubeFromFloor = utils.SingletonParameterizedOption(
            # variables: [robot, cube]
            "PickCubeFromFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cube_type])

        PlaceCubeOnTable = utils.SingletonParameterizedOption(
            # variables: [robot, cube, table]
            "PlaceCubeOnTable",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cube_type, table_type])

        PlaceCubeOnFloor = utils.SingletonParameterizedOption(
            # variables: [robot, cube]
            "PlaceCubeOnFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cube_type])

        PickBallFromTable = utils.SingletonParameterizedOption(
            # variables: [robot, ball, table]
            "PickBallFromTable",
            cls._create_pass_through_policy(action_space),
            params_space=params_space,
            types=[robot_type, ball_type, table_type])

        PickBallFromFloor = utils.SingletonParameterizedOption(
            # variables: [robot, ball]
            "PickBallFromFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type])

        PlaceBallOnTable = utils.SingletonParameterizedOption(
            # variables: [robot, ball, table]
            "PlaceBallOnTable",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, table_type])

        PlaceBallOnFloor = utils.SingletonParameterizedOption(
            # variables: [robot, ball]
            "PlaceBallOnFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type])

        PickCupFromTable = utils.SingletonParameterizedOption(
            # variables: [robot, cup, table]
            "PickCupFromTable",
            cls._create_pass_through_policy(action_space),
            params_space=params_space,
            types=[robot_type, cup_type, table_type])

        PickCupFromFloor = utils.SingletonParameterizedOption(
            # variables: [robot, cup]
            "PickCupFromFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cup_type])

        PlaceCupWithBallOnTable = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup, table]
            "PlaceCupOnTable",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type, table_type])

        PlaceCupWithoutBallOnTable = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup, table]
            "PlaceCupWithoutBallOnTable",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type, table_type])

        PlaceCupWithBallOnFloor = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup, floor]
            "PlaceCupOnFloor",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, ball_type, cup_type])

        PlaceCupWithoutBallOnFloor = utils.SingletonParameterizedOption(
            # variables: [robot, ball, cup, floor]
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

        NavigateToCube = utils.SingletonParameterizedOption(
            # variables: [robot, cube]
            "NavigateToCube",
            cls._create_pass_through_policy(action_space),
            # Parameters are absolute x, y actions.
            params_space=params_space,
            types=[robot_type, cube_type])

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
            PickCubeFromTable, PickCubeFromFloor, PlaceCubeOnTable,
            PlaceCubeOnFloor, NavigateToCube, NavigateToTable,
            PickBallFromTable, PickBallFromFloor, PlaceBallOnTable,
            PlaceBallOnFloor, PickCupFromTable, PickCupFromFloor,
            PlaceCupWithBallOnTable, PlaceCupWithoutBallOnTable,
            PlaceCupWithBallOnFloor, PlaceCupWithoutBallOnFloor,
            PlaceBallInCupOnFloor, PlaceBallInCupOnTable, NavigateToBall,
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
