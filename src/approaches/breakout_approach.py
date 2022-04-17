"""A hardcoded approach for the Breakout environment."""

from typing import Callable

from predicators.src.approaches import BaseApproach
from predicators.src.structs import Action, State, Task
from predicators.src.settings import CFG

import numpy as np


class HardcodedBreakoutApproach(BaseApproach):
    """A hardcoded approach for the Breakout environment."""

    @classmethod
    def get_name(cls) -> str:
        return "hardcoded_breakout"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        assert CFG.env == "breakout"
        ball_type, brick_type, paddle_type = sorted(self._types)
        assert ball_type.name == "ball"
        assert brick_type.name == "brick"
        assert paddle_type.name == "paddle"

        def _policy(state: State) -> Action:
            ball, = state.get_objects(ball_type)
            paddle, = state.get_objects(paddle_type)

            ball_c = state.get(ball, "c")
            paddle_c = state.get(paddle, "c") + state.get(paddle, "w")/2
            paddle_dc = state.get(paddle, "dc")

            # If the ball and paddle are this far apart, always try to move.
            rush_margin = 10

            if ball_c < paddle_c and (np.sign(paddle_dc) != -1 or paddle_c - ball_c > rush_margin):
                act_str = "left"
            elif ball_c > paddle_c and (np.sign(paddle_dc) != -1 or ball_c - paddle_c > rush_margin):
                act_str = "right"
            else:
                act_str = "noop"

            if act_str == "left":
                continuous_act = -1.0
            elif act_str == "noop":
                continuous_act = 0.0
            else:
                assert act_str == "right"
                continuous_act = 1.0
            return Action(np.array([continuous_act], dtype=np.float32))

        return _policy
