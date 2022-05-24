"""Toy environment for testing option learning."""

import logging
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class TouchPointEnv(BaseEnv):
    """An environment where a 2D point mass robot must reach a static 2D point.

    The action space is 1D, denoting the angle of movement. The
    magnitude of the movement is constant. The point is considered
    touched if the distance between the center of the robot and the
    center of the target point is less than a certain threshold, which
    is greater than the action magnitude.
    """
    x_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 1.0
    y_lb: ClassVar[float] = 0.0
    y_ub: ClassVar[float] = 1.0
    action_magnitude: ClassVar[float] = 0.1
    # The target point is touched if the distance between the robot and target
    # is less than action_magnitude * touch_multiplier.
    touch_multiplier: ClassVar[float] = 1.5

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x", "y"])
        self._target_type = Type("target", ["x", "y"])
        # Predicates
        self._Touched = Predicate("Touched",
                                  [self._robot_type, self._target_type],
                                  self._Touched_holds)
        # Options
        self._MoveTo = ParameterizedOption(
            "MoveTo",
            types=[self._robot_type, self._target_type],
            params_space=Box(0, 1, (0, )),
            policy=self._MoveTo_policy,
            initiable=lambda s, m, o, p: True,
            terminal=self._MoveTo_terminal)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._target = Object("target", self._target_type)

    @classmethod
    def get_name(cls) -> str:
        return "touch_point"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        rot, = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        new_x = x + np.cos(rot) * self.action_magnitude
        new_y = y + np.sin(rot) * self.action_magnitude
        new_x = np.clip(new_x, self.x_lb, self.x_ub)
        new_y = np.clip(new_y, self.y_lb, self.y_ub)
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Touched}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Touched}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._target_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._MoveTo}

    @property
    def action_space(self) -> Box:
        # An angle in radians.
        return Box(-np.pi, np.pi, (1, ))

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        robot_color = "red"
        target_color = "blue"
        rad = (self.touch_multiplier * self.action_magnitude) / 2
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        target_x = state.get(self._target, "x")
        target_y = state.get(self._target, "y")
        robot_circ = plt.Circle((robot_x, robot_y), rad, color=robot_color)
        target_circ = plt.Circle((target_x, target_y), rad, color=target_color)
        ax.add_patch(robot_circ)
        ax.add_patch(target_circ)
        ax.set_xlim(self.x_lb - rad, self.x_ub + rad)
        ax.set_ylim(self.y_lb - rad, self.y_ub + rad)
        title = f"{robot_color} = robot, {target_color} = target"
        if caption is not None:
            title += f";\n{caption}"
        plt.suptitle(title, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        # There is only one goal in this environment.
        goal_atom = GroundAtom(self._Touched, [self._robot, self._target])
        goal = {goal_atom}
        # The initial positions of the robot and dot vary. The only constraint
        # is that the initial positions should be far enough away that the goal
        # is not initially satisfied.
        tasks: List[Task] = []
        while len(tasks) < num:
            state = utils.create_state_from_dict({
                self._robot: {
                    "x": rng.uniform(self.x_lb, self.x_ub),
                    "y": rng.uniform(self.y_lb, self.y_ub),
                },
                self._target: {
                    "x": rng.uniform(self.x_lb, self.x_ub),
                    "y": rng.uniform(self.y_lb, self.y_ub),
                },
            })
            # Make sure goal is not satisfied.
            if not goal_atom.holds(state):
                tasks.append(Task(state, goal))
        return tasks

    @staticmethod
    def _MoveTo_policy(state: State, memory: Dict, objects: Sequence[Object],
                       params: Array) -> Action:
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

    def _MoveTo_terminal(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        return self._Touched_holds(state, objects)

    def _Touched_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, target = objects
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        tx = state.get(target, "x")
        ty = state.get(target, "y")
        dist = np.sqrt((rx - tx)**2 + (ry - ty)**2)
        return dist < self.action_magnitude * self.touch_multiplier

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:
        logging.info("Controls: mouse click to move")

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:
            assert event.key is None, "Keyboard controls not allowed."
            rx = state.get(self._robot, "x")
            ry = state.get(self._robot, "y")
            tx = event.xdata
            ty = event.ydata
            assert tx is not None and ty is not None, "Out-of-bounds click"
            dx = tx - rx
            dy = ty - ry
            rot = np.arctan2(dy, dx)  # between -pi and pi
            return Action(np.array([rot], dtype=np.float32))

        return _event_to_action
