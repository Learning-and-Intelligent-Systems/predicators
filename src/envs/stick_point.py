"""An environment where a robot must touch points with its hand or a stick."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class StickPointEnv(BaseEnv):
    """An environment where a robot must touch points with its hand or a stick.
    """
    x_lb: ClassVar[float] = 0.0
    y_lb: ClassVar[float] = 0.0
    theta_lb: ClassVar[float] = -np.pi
    x_ub: ClassVar[float] = 10.0
    y_ub: ClassVar[float] = 6.0
    theta_ub: ClassVar[float] = np.pi
    robot_radius: ClassVar[float] = 0.1
    point_radius: ClassVar[float] = 0.1
    stick_width: ClassVar[float] = 0.02
    stick_length: ClassVar[float] = 1.0

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x", "y", "theta"])
        self._point_type = Type("point", ["x", "y"])
        self._stick_type = Type("stick", ["x", "y", "theta", "held"])
        # Predicates
        self._Touched = Predicate("Touched",
                                  [self._point_type],
                                  self._Touched_holds)
        self._InContactStickPoint = Predicate("InContactStickPoint",
                                    [self._stick_type, self._point_type],
                                    self._InContactStickPoint_holds)
        self._InContactRobotPoint = Predicate("InContactRobotPoint",
                                    [self._robot_type, self._point_type],
                                    self._InContactRobotPoint_holds)
        self._InContactRobotStick = Predicate("InContactRobotStick",
                                    [self._robot_type, self._stick_type],
                                    self._InContactRobotStick_holds)
        self._Grasped = Predicate("Grasped", [self._robot_type, self._stick_type], self._Grasped_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type], self._HandEmpty_holds)
        # Options
        self._MoveTo = ParameterizedOption(
            "MoveTo",
            types=[self._robot_type],
            params_space=Box(np.array([self.x_lb, self.y_lb, self.theta_lb]),
                             np.array([self.x_ub, self.y_ub, self.theta_ub]),
                             (3, ),
                             dtype=np.float32),
            policy=self._MoveTo_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._MoveTo_terminal)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._stick = Object("stick", self._stick_type)

    def _MoveTo_policy(self, state: State, memory: Dict, objects: Sequence[Object],
                       params: Array) -> Action:
        import ipdb; ipdb.set_trace()

    def _MoveTo_terminal(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> bool:
        import ipdb; ipdb.set_trace()

    def _Touched_holds(self, state: State, objects: Sequence[Object]) -> bool:
        import ipdb; ipdb.set_trace()

    def _InContactStickPoint_holds(self, state: State, objects: Sequence[Object]) -> bool:
        import ipdb; ipdb.set_trace()

    def _InContactRobotPoint_holds(self, state: State, objects: Sequence[Object]) -> bool:
        import ipdb; ipdb.set_trace()

    def _InContactRobotStick_holds(self, state: State, objects: Sequence[Object]) -> bool:
        import ipdb; ipdb.set_trace()

    def _Grasped_holds(self, state: State, objects: Sequence[Object]) -> bool:
        import ipdb; ipdb.set_trace()

    def _HandEmpty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        import ipdb; ipdb.set_trace()
