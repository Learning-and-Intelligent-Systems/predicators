"""Ground-truth options for the repeated nextto environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.repeated_nextto import RepeatedNextToEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class RepeatedNextToGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the repeated nextto environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "repeated_nextto", "repeated_nextto_ambiguous",
            "repeated_nextto_simple"
        }

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        dot_type = types["dot"]

        Move = utils.SingletonParameterizedOption("Move",
                                                  cls.create_move_policy(),
                                                  types=[robot_type, dot_type],
                                                  params_space=Box(
                                                      -1, 1, (1, )))

        Grasp = utils.SingletonParameterizedOption(
            "Grasp",
            policy=cls.create_grasp_policy(),
            types=[robot_type, dot_type])

        return {Move, Grasp}

    @classmethod
    def create_move_policy(cls) -> ParameterizedPolicy:
        """Made public for use by RNTSingleOptionGroundTruthOptionFactory."""

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            _, dot = objects
            dot_x = state.get(dot, "x")
            delta, = params
            robot_x = max(min(RepeatedNextToEnv.env_ub, dot_x + delta),
                          RepeatedNextToEnv.env_lb)
            norm_robot_x = (robot_x - RepeatedNextToEnv.env_lb) / (
                RepeatedNextToEnv.env_ub - RepeatedNextToEnv.env_lb)
            return Action(np.array([0, norm_robot_x, 0], dtype=np.float32))

        return policy

    @classmethod
    def create_grasp_policy(cls) -> ParameterizedPolicy:
        """Made public for use by RNTSingleOptionGroundTruthOptionFactory."""

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, dot = objects
            dot_x = state.get(dot, "x")
            norm_dot_x = (dot_x - RepeatedNextToEnv.env_lb) / (
                RepeatedNextToEnv.env_ub - RepeatedNextToEnv.env_lb)
            return Action(np.array([1, 0, norm_dot_x], dtype=np.float32))

        return policy


class RNTSingleOptionGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the single option RNT environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"repeated_nextto_single_option"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        dot_type = types["dot"]

        # The first parameter dimension modulates whether the action will be a
        # move (negative value) or a grasp (nonnegative value). The second
        # dimension is the same as that for self._Move in the parent class.
        MoveGrasp = utils.SingletonParameterizedOption(
            "MoveGrasp",
            policy=cls._create_move_grasp_policy(),
            types=[robot_type, dot_type],
            params_space=Box(-1, 1, (2, )))

        return {MoveGrasp}

    @classmethod
    def _create_move_grasp_policy(cls) -> ParameterizedPolicy:

        move = RepeatedNextToGroundTruthOptionFactory.create_move_policy()
        grasp = RepeatedNextToGroundTruthOptionFactory.create_grasp_policy()

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            if params[0] < 0:
                return move(state, memory, objects, params[1:])
            return grasp(state, memory, objects, params[1:])

        return policy
