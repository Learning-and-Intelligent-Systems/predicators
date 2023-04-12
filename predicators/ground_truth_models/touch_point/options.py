"""Ground-truth options for the touch point environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.envs.touch_point import TouchPointEnvParam
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type


class TouchPointGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the touch point environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"touch_point"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        target_type = types["target"]
        Touched = predicates["Touched"]

        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
            # Move in the direction of the target.
            del memory, params  # unused
            robot, target = objects
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            tx = state.get(target, "x")
            ty = state.get(target, "y")
            dx = tx - rx
            dy = ty - ry
            rot = np.arctan2(dy, dx)  # between -pi and pi
            return Action(np.array([rot], dtype=np.float32))

        def _MoveTo_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            return Touched.holds(state, objects)

        MoveTo = ParameterizedOption("MoveTo",
                                     types=[robot_type, target_type],
                                     params_space=Box(0, 1, (0, )),
                                     policy=_MoveTo_policy,
                                     initiable=lambda s, m, o, p: True,
                                     terminal=_MoveTo_terminal)

        return {MoveTo}


class TouchPointParamGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the touch point param environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"touch_point_param"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        target_type = types["target"]
        Touched = predicates["Touched"]

        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
            del state, memory, objects  # unused
            return Action(params)

        def _MoveTo_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del memory, params  # unused
            return Touched.holds(state, objects)

        MoveTo = ParameterizedOption(
            "MoveTo",
            types=[robot_type, target_type],
            params_space=Box(TouchPointEnvParam.action_limits[0],
                             TouchPointEnvParam.action_limits[1], (2, )),
            policy=_MoveTo_policy,
            initiable=lambda s, m, o, p: True,
            terminal=_MoveTo_terminal)

        return {MoveTo}
