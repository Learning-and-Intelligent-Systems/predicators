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
    """An environment where a robot must touch points with its hand or a
    stick."""
    x_lb: ClassVar[float] = 0.0
    y_lb: ClassVar[float] = 0.0
    theta_lb: ClassVar[float] = -np.pi  # radians
    x_ub: ClassVar[float] = 10.0
    y_ub: ClassVar[float] = 6.0
    theta_ub: ClassVar[float] = np.pi  # radians
    max_speed: ClassVar[float] = 0.1  # shared by dx, dy, dtheta
    robot_radius: ClassVar[float] = 0.1
    point_radius: ClassVar[float] = 0.1
    stick_width: ClassVar[float] = 0.02
    stick_length: ClassVar[float] = 1.0

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x", "y", "theta"])
        self._point_type = Type("point", ["x", "y", "touched"])
        self._stick_type = Type("stick", ["x", "y", "theta", "held"])
        # Predicates
        self._Touched = Predicate("Touched", [self._point_type],
                                  self._Touched_holds)
        self._InContactStickPoint = Predicate(
            "InContactStickPoint", [self._stick_type, self._point_type],
            self._InContactStickPoint_holds)
        self._InContactRobotPoint = Predicate(
            "InContactRobotPoint", [self._robot_type, self._point_type],
            self._InContactRobotPoint_holds)
        self._InContactRobotStick = Predicate(
            "InContactRobotStick", [self._robot_type, self._stick_type],
            self._InContactRobotStick_holds)
        self._Grasped = Predicate("Grasped",
                                  [self._robot_type, self._stick_type],
                                  self._Grasped_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
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

    @classmethod
    def get_name(cls) -> str:
        return "stick_point"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, dtheta = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        theta = state.get(self._robot, "theta")
        import ipdb
        ipdb.set_trace()
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        next_state.set(self._robot, "theta", new_theta)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               num_point_lst=CFG.stick_point_num_points_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               num_point_lst=CFG.stick_point_num_points_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Touched, self._InContactRobotPoint,
            self._InContactRobotStick, self._InContactStickPoint,
            self._Grasped, self._HandEmpty
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Touched}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._stick_type, self._point_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._MoveTo}

    @property
    def action_space(self) -> Box:
        # dx, dy, dtheta (radians)
        return Box(low=-self.max_speed,
                   high=self.max_speed,
                   shape=(3, ),
                   dtype=np.float32)

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        fig, ax = plt.subplots(1,
                               1,
                               figsize=(self.x_ub - self.x_lb,
                                        self.y_ub - self.y_lb))
        assert caption is None
        import ipdb
        ipdb.set_trace()
        plt.tight_layout()
        img = utils.fig2data(fig)
        plt.close()
        return [img]

    def _get_tasks(self, num: int, num_point_lst: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        while len(tasks) < num:
            state_dict = {}
            num_points = num_point_lst[rng.choice(len(num_point_lst))]
            points = [Object(f"point{i}", self._point_type) for i in range(num_points)]
            goal = {GroundAtom(self._Touched, [p]) for p in points}
            import ipdb; ipdb.set_trace()


            init_state = utils.create_state_from_dict(state_dict)

        return tasks

    def _MoveTo_policy(self, state: State, memory: Dict,
                       objects: Sequence[Object], params: Array) -> Action:
        import ipdb
        ipdb.set_trace()

    def _MoveTo_terminal(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> bool:
        import ipdb
        ipdb.set_trace()

    def _Touched_holds(self, state: State, objects: Sequence[Object]) -> bool:
        import ipdb
        ipdb.set_trace()

    def _InContactStickPoint_holds(self, state: State,
                                   objects: Sequence[Object]) -> bool:
        import ipdb
        ipdb.set_trace()

    def _InContactRobotPoint_holds(self, state: State,
                                   objects: Sequence[Object]) -> bool:
        import ipdb
        ipdb.set_trace()

    def _InContactRobotStick_holds(self, state: State,
                                   objects: Sequence[Object]) -> bool:
        import ipdb
        ipdb.set_trace()

    def _Grasped_holds(self, state: State, objects: Sequence[Object]) -> bool:
        import ipdb
        ipdb.set_trace()

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        import ipdb
        ipdb.set_trace()
