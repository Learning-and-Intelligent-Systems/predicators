"""Ground truth options for the shelves2d environment"""

import gym
from typing import Dict, Sequence, Set, Type, cast

import numpy as np
from experiments.shelves2d import Shelves2DEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, Predicate, State
from predicators.utils import SingletonParameterizedOption

class Shelves2DGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground truth options for the shelves2d environment"""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return set(["shelves2d"])

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: gym.spaces.Box) -> Set[ParameterizedOption]:
        return set([
            ParameterizedOption(
                "Move",
                [],
                action_space,
                cls.move,
                cls.initiable,
                cls.terminal
            )
        ])

    @classmethod
    def initiable(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> bool:
        return True

    @classmethod
    def terminal(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> bool:
        terminal_executed = data.get("terminal_executed", False)
        data["terminal_executed"] = True
        return terminal_executed

    @classmethod
    def move(cls, state: State, data: Dict, objects: Sequence[Object], arr: Array) -> Action:
        return Action(arr)

