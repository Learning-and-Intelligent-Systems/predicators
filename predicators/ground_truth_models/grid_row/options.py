"""Ground-truth options for the grid row environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type


class GridRowGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grid row environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"grid_row"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        light_type = types["light"]
        cell_type = types["cell"]

        # MoveTo
        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
            del memory, params  # unused
            robot, _, target_cell = objects
            rob_x = state.get(robot, "x")
            target_x = state.get(target_cell, "x")
            dx = target_x - rob_x
            return Action(np.array([dx, 0.0], dtype=np.float32))

        MoveRobot = utils.SingletonParameterizedOption(
            "MoveRobot",
            types=[robot_type, cell_type, cell_type],
            policy=_MoveTo_policy,
        )

        # TurnOnLight
        def _toggle_light_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
            del state, objects, memory  # unused
            dlight, = params
            return Action(np.array([0.0, dlight], dtype=np.float32))

        TurnOnLight = utils.SingletonParameterizedOption(
            "TurnOnLight",
            types=[robot_type, cell_type, light_type],
            policy=_toggle_light_policy,
            params_space=Box(-1.0, 1.0, (1, )),
        )

        # TurnOffLight
        TurnOffLight = utils.SingletonParameterizedOption(
            "TurnOffLight",
            types=[robot_type, cell_type, light_type],
            policy=_toggle_light_policy,
            params_space=Box(-1.0, 1.0, (1, )),
        )

        # Impossible
        def _null_policy(state: State, memory: Dict, objects: Sequence[Object],
                         params: Array) -> Action:
            del state, memory, objects, params  # unused
            return Action(np.array([0.0, 0.0], dtype=np.float32))

        JumpToLight = utils.SingletonParameterizedOption(
            "JumpToLight",
            types=[robot_type, cell_type, cell_type, cell_type, light_type],
            policy=_null_policy,
            params_space=Box(-1.0, 1.0, (1, )),
        )

        return {MoveRobot, TurnOnLight, TurnOffLight, JumpToLight}


class GridRowDoorGroundTruthOptionFactory(GridRowGroundTruthOptionFactory):
    """Ground-truth options for the grid row door environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"grid_row_door"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        light_type = types["light"]
        cell_type = types["cell"]

        # MoveTo
        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
            del memory, params  # unused
            robot, _, target_cell = objects
            rob_x = state.get(robot, "x")
            target_x = state.get(target_cell, "x")
            dx = target_x - rob_x
            return Action(np.array([dx, 0.0, 0.0, 0.0], dtype=np.float32))

        MoveRobot = utils.SingletonParameterizedOption(
            "MoveRobot",
            types=[robot_type, cell_type, cell_type],
            policy=_MoveTo_policy,
        )

        def _toggle_light_policy(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
            del state, objects, memory  # unused
            dlight, = params
            return Action(np.array([0.0, dlight, 0.0, 0.0], dtype=np.float32))

        # TurnOnLight
        TurnOnLight = utils.SingletonParameterizedOption(
            "TurnOnLight",
            types=[robot_type, cell_type, light_type],
            policy=_toggle_light_policy,
            params_space=Box(-1.0, 1.0, (1, )),
        )

        # TurnOffLight
        TurnOffLight = utils.SingletonParameterizedOption(
            "TurnOffLight",
            types=[robot_type, cell_type, light_type],
            policy=_toggle_light_policy,
            params_space=Box(-1.0, 1.0, (1, )),
        )

        # MoveKey
        def _move_key_policy(state: State, memory: Dict,
                             objects: Sequence[Object],
                             params: Array) -> Action:
            del state, objects, memory  # unused
            dmove, = params
            return Action(np.array([0.0, 0.0, dmove, 0.0], dtype=np.float32))

        MoveKey = utils.SingletonParameterizedOption(
            "MoveKey",
            types=[robot_type, cell_type],
            policy=_move_key_policy,
            params_space=Box(-1.0, 1.0, (1, )),
        )

        # TurnKey
        def _turn_key_policy(state: State, memory: Dict,
                             objects: Sequence[Object],
                             params: Array) -> Action:
            del state, objects, memory  # unused
            dturn, = params
            return Action(np.array([0.0, 0.0, 0.0, dturn], dtype=np.float32))

        TurnKey = utils.SingletonParameterizedOption(
            "TurnKey",
            types=[robot_type, cell_type],
            policy=_turn_key_policy,
            params_space=Box(-1.0, 1.0, (1, )),
        )

        return {MoveRobot, TurnOnLight, TurnOffLight, TurnKey, MoveKey}
    