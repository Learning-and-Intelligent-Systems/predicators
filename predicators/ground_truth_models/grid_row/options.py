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
            params_space=Box(0.0, 1.0, (1, )),
        )

        TurnOffLight = utils.SingletonParameterizedOption(
            "TurnOffLight",
            types=[robot_type, cell_type, light_type],
            policy=_toggle_light_policy,
            params_space=Box(0.0, 1.0, (1, )),
        )

        return {MoveRobot, TurnOnLight, TurnOffLight}
