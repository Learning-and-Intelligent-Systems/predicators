"""Toy environment for testing option learning."""

import logging
from typing import Callable, ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


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

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", ["x", "y"])
        self._target_type = Type("target", ["x", "y"])
        # Predicates
        self._Touched = Predicate("Touched",
                                  [self._robot_type, self._target_type],
                                  self._Touched_holds)
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

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
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
    def action_space(self) -> Box:
        # An angle in radians.
        return Box(-np.pi, np.pi, (1, ))

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
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

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        # There is only one goal in this environment.
        goal_atom = GroundAtom(self._Touched, [self._robot, self._target])
        goal = {goal_atom}
        # The initial positions of the robot and dot vary. The only constraint
        # is that the initial positions should be far enough away that the goal
        # is not initially satisfied.
        tasks: List[EnvironmentTask] = []
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
                tasks.append(EnvironmentTask(state, goal))
        return tasks

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


class TouchPointEnvParam(TouchPointEnv):
    """TouchPointEnv with a parameterized option and a 2D action space."""

    action_limits: ClassVar[List[float]] = [-2.0, 2.0]

    @classmethod
    def get_name(cls) -> str:
        return "touch_point_param"

    @property
    def action_space(self) -> Box:
        # The action space is (dx, dy).
        lb, ub = self.action_limits
        return Box(np.array([lb, lb], dtype=np.float32),
                   np.array([ub, ub], dtype=np.float32))

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        new_x = x + (dx * self.action_magnitude)
        new_y = y + (dy * self.action_magnitude)
        new_x = np.clip(new_x, self.x_lb, self.x_ub)
        new_y = np.clip(new_y, self.y_lb, self.y_ub)
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        return next_state


class TouchOpenEnv(TouchPointEnvParam):
    """TouchPointEnvParam but where the target is a door from DoorsEnv that
    needs to be opened."""

    open_door_threshold: ClassVar[float] = 1e-2

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__()

        # Add door type.
        self._door_type = Type(
            "door", ["x", "y", "mass", "friction", "rot", "flex", "open"])

        # Add predicates.
        del self._Touched
        self._TouchingDoor = Predicate("TouchingDoor",
                                       [self._robot_type, self._door_type],
                                       self._TouchingDoor_holds)
        self._DoorIsOpen = Predicate("DoorIsOpen", [self._door_type],
                                     self._DoorIsOpen_holds)
        # Add static object.
        self._door = Object("door", self._door_type)

    @classmethod
    def get_name(cls) -> str:
        return "touch_open"

    @property
    def action_space(self) -> Box:
        # The action space is (dx, dy, drot).
        lb, ub = self.action_limits
        return Box(np.array([lb, lb, lb], dtype=np.float32),
                   np.array([ub, ub, ub], dtype=np.float32))

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, drot = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        rot = state.get(self._door, "rot")

        new_x = x + (dx * self.action_magnitude)
        new_y = y + (dy * self.action_magnitude)
        new_x = np.clip(new_x, self.x_lb, self.x_ub)
        new_y = np.clip(new_y, self.y_lb, self.y_ub)
        new_rot = rot + drot
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)

        # If touching the door, change its value based on the action.
        for door in state.get_objects(self._door_type):
            if self._TouchingDoor_holds(state, [self._robot, door]):
                # Rotate the door handle.
                next_state.set(door, "rot", new_rot)
                # Check if we should open the door.
                target = self._get_open_door_target_value(
                    mass=state.get(door, "mass"),
                    friction=state.get(door, "friction"),
                    flex=state.get(door, "flex"),
                )
                if abs(new_rot - target) < self.open_door_threshold:
                    next_state.set(door, "open", 1.0)
                else:
                    next_state.set(door, "open", 0.0)
        return next_state

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._TouchingDoor, self._DoorIsOpen}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._DoorIsOpen}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._door_type}

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        robot_color = "gray"
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        door_x = state.get(self._door, "x")
        door_y = state.get(self._door, "y")
        door_open = state.get(self._door, "open")
        if door_open:
            door_color = "yellow"
        else:
            door_color = "brown"
        rad = (self.touch_multiplier * self.action_magnitude) / 2
        robot_circ = plt.Circle((robot_x, robot_y), rad / 2, color=robot_color)
        door_circ = plt.Circle((door_x, door_y),
                               rad,
                               color=door_color,
                               alpha=0.5)
        ax.add_patch(robot_circ)
        ax.add_patch(door_circ)
        ax.set_xlim(self.x_lb - rad, self.x_ub + rad)
        ax.set_ylim(self.y_lb - rad, self.y_ub + rad)
        title = f"{robot_color} = robot, {door_color} = door"
        if caption is not None:
            title += f";\n{caption}"
        plt.suptitle(title, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        goal_atom1 = GroundAtom(self._TouchingDoor, [self._robot, self._door])
        goal_atom2 = GroundAtom(self._DoorIsOpen, [self._door])
        goal1 = {goal_atom1, goal_atom2}
        # The initial positions of the robot and door vary. The only constraint
        # is that the initial positions should be far enough away that the goal
        # is not initially satisfied.
        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            flex = rng.uniform(0.0, 1.0)
            rot = rng.uniform(0.0, 1.0)
            state = utils.create_state_from_dict({
                self._robot: {
                    "x": rng.uniform(self.x_lb, self.x_ub),
                    "y": rng.uniform(self.y_lb, self.y_ub),
                },
                self._door: {
                    "x": rng.uniform(self.x_lb, self.x_ub),
                    "y": rng.uniform(self.y_lb, self.y_ub),
                    "mass": rng.uniform(0.0, 1.0),
                    "friction": rng.uniform(0.0, 1.0),
                    "rot": rot,
                    "flex": flex,
                    "open": 0.0  # start out closed
                },
            })
            # Make sure the goal is not satisfied.
            if not goal_atom1.holds(state) and not goal_atom2.holds(state):
                tasks.append(EnvironmentTask(state, goal1))
        return tasks

    def _TouchingDoor_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        robot, door = objects
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        tx = state.get(door, "x")
        ty = state.get(door, "y")
        dist = np.sqrt((rx - tx)**2 + (ry - ty)**2)
        return dist < self.action_magnitude * self.touch_multiplier

    @staticmethod
    def _DoorIsOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        door, = objects
        return state.get(door, "open") > 0.5

    @staticmethod
    def _get_open_door_target_value(mass: float, friction: float,
                                    flex: float) -> float:
        # A made up complicated function.
        # Want to be able to swap this out based on CFG.
        return np.tanh(flex) * (np.sin(mass) +
                                np.cos(friction) * np.sqrt(mass))
