"""A 2D navigation environment with obstacles, rooms, and doors."""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches
from numpy.typing import NDArray

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, Object, \
    ParameterizedOption, Predicate, State, Task, Type
from predicators.src.utils import _Geom2D


class DoorsEnv(BaseEnv):
    """A 2D navigation environment with obstacles, rooms, and doors."""
    room_size: ClassVar[float] = 1.0
    hallway_width: ClassVar[float] = 0.25
    wall_depth: ClassVar[float] = 0.01
    robot_radius: ClassVar[float] = 0.05
    action_magnitude: ClassVar[float] = 0.05
    robot_initial_position_radius: ClassVar[float] = 0.05

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x", "y"])
        self._door_type = Type("door", ["x", "y", "theta", "target", "open"])
        self._room_type = Type("room", ["x", "y"])
        self._obstacle_type = Type("obstacle",
                                   ["x", "y", "width", "height", "theta"])
        # Predicates
        self._InRoom = Predicate("InRoom", [self._robot_type, self._room_type],
                                 self._InRoom_holds)
        # TODO
        # Options
        # TODO
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "doors"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, drot, push = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        new_x = x + dx
        new_y = y + dy
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        # Check for collisions.
        if self._state_has_collision(next_state):
            # Revert the change to the robot position.
            next_state.set(self._robot, "x", x)
            next_state.set(self._robot, "y", y)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InRoom}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InRoom}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._door_type, self._room_type,
            self._obstacle_type
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return set()  # TODO

    @property
    def action_space(self) -> Box:
        # dx, dy, drot, push
        lb = np.array(
            [-self.action_magnitude, -self.action_magnitude, -np.pi, 0.0],
            dtype=np.float32)
        ub = np.array(
            [self.action_magnitude, self.action_magnitude, np.pi, 1.0],
            dtype=np.float32)
        return Box(lb, ub)

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        # Draw rooms.
        default_room_color = "lightgray"
        in_room_color = "lightsteelblue"
        goal_room_color = "khaki"
        goal_room = next(iter(task.goal)).objects[1]
        for room in state.get_objects(self._room_type):
            room_geom = self._object_to_geom(room, state)
            if room == goal_room:
                color = goal_room_color
            elif self._InRoom_holds(state, [self._robot, room]):
                color = in_room_color
            else:
                color = default_room_color
            room_geom.plot(ax, color=color)

        # Draw robot.
        robot_color = "blue"
        robot_geom = self._object_to_geom(self._robot, state)
        robot_geom.plot(ax, color=robot_color)

        # TODO draw doors

        # Draw obstacles (including room walls).
        obstacle_color = "black"
        for obstacle in state.get_objects(self._obstacle_type):
            obstacle_geom = self._object_to_geom(obstacle, state)
            obstacle_geom.plot(ax, color=obstacle_color)

        x_lb, x_ub, y_lb, y_ub = self._get_world_boundaries(state)
        pad = 2 * self.wall_depth
        ax.set_xlim(x_lb - pad, x_ub + pad)
        ax.set_ylim(y_lb - pad, y_ub + pad)

        assert caption is None
        plt.axis("off")
        plt.tight_layout()
        img = utils.fig2data(fig)
        plt.close()
        return [img]

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        tasks: List[Task] = []
        # Create the static parts of the initial state.
        static_state_dict = {}
        # TODO randomize this.
        room_map = np.array([
            [1, 0, 0, 0, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1],
        ])
        num_rows, num_cols = room_map.shape
        for (r, c) in np.argwhere(room_map):
            room = Object(f"room{r}-{c}", self._room_type)
            room_x = float(c * self.room_size)
            room_y = float((num_rows - 1 - r) * self.room_size)
            static_state_dict[room] = {
                "x": room_x,
                "y": room_y,
            }
            # Create obstacles for the room walls.
            hall_top = float(r > 0 and room_map[r - 1, c])
            hall_bottom = float(r < num_rows - 1 and room_map[r + 1, c])
            hall_left = float(c > 0 and room_map[r, c - 1])
            hall_right = float(c < num_cols - 1 and room_map[r, c + 1])
            wall_rects = self._get_rectangles_for_room_walls(
                room_x, room_y, hall_top, hall_bottom, hall_left, hall_right)
            for i, rect in enumerate(wall_rects):
                wall = Object(f"wall{r}-{c}-{i}", self._obstacle_type)
                static_state_dict[wall] = {
                    "x": rect.x,
                    "y": rect.y,
                    "height": rect.height,
                    "width": rect.width,
                    "theta": rect.theta,
                }
        # TODO add other obstacles.
        init_state = utils.create_state_from_dict({
            **static_state_dict,
            # Will get overridden.
            self._robot: {
                "x": np.inf,
                "y": np.inf,
            },
        })
        rooms = sorted(init_state.get_objects(self._room_type))
        while len(tasks) < num:
            # Sample an initial and target room.
            start_idx, goal_idx = rng.choice(len(rooms), size=2, replace=False)
            start_room, goal_room = rooms[start_idx], rooms[goal_idx]
            # Sample an initial state.
            state = init_state.copy()
            # Always start out near the center of the room to avoid issues with
            # rotating in corners.
            room_x = init_state.get(start_room, "x")
            room_y = init_state.get(start_room, "y")
            room_cx = room_x + self.room_size / 2
            room_cy = room_y + self.room_size / 2
            rad = self.robot_initial_position_radius
            x = rng.uniform(room_cx - rad, room_cx + rad)
            y = rng.uniform(room_cy - rad, room_cy + rad)
            state.set(self._robot, "x", x)
            state.set(self._robot, "y", y)
            # Make sure the state is collision-free. Should be guaranteed
            # because we're initializing the robot in the middle of the room.
            assert not self._state_has_collision(state)
            # Set the goal.
            goal_atom = GroundAtom(self._InRoom, [self._robot, goal_room])
            goal = {goal_atom}
            assert not goal_atom.holds(state)
            tasks.append(Task(state, goal))
        return tasks

    def _InRoom_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # The robot is in the room if its center is in the room.
        robot, room = objects
        robot_geom = self._object_to_geom(robot, state)
        assert isinstance(robot_geom, utils.Circle)
        room_geom = self._object_to_geom(room, state)
        return room_geom.contains_point(robot_geom.x, robot_geom.y)

    def _state_has_collision(self, state: State) -> bool:
        # TODO
        return False

    def _object_to_geom(self, obj: Object, state: State) -> _Geom2D:
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        if obj.is_instance(self._robot_type):
            return utils.Circle(x, y, self.robot_radius)
        if obj.is_instance(self._room_type):
            width = self.room_size
            height = self.room_size
            theta = 0.0
        elif obj.is_instance(self._door_type):
            width = self.hallway_width
            height = self.wall_depth
            theta = state.get(obj, "theta")
        else:
            assert obj.is_instance(self._obstacle_type)
            width = state.get(obj, "width")
            height = state.get(obj, "height")
            theta = state.get(obj, "theta")
        return utils.Rectangle(x=x,
                               y=y,
                               width=width,
                               height=height,
                               theta=theta)

    def _get_world_boundaries(
            self, state: State) -> Tuple[float, float, float, float]:
        x_lb, y_lb = np.inf, np.inf
        x_ub, y_ub = -np.inf, -np.inf
        for room in state.get_objects(self._room_type):
            room_x = state.get(room, "x")
            room_y = state.get(room, "y")
            x_lb = min(x_lb, room_x)
            x_ub = max(x_ub, room_x + self.room_size)
            y_lb = min(y_lb, room_y)
            y_ub = max(y_ub, room_y + self.room_size)
        return x_lb, x_ub, y_lb, y_ub

    @lru_cache(maxsize=None)
    def _get_rectangles_for_room_walls(
            self, room_x: float, room_y: float, hall_top: bool,
            hall_bottom: bool, hall_left: bool,
            hall_right: bool) -> List[utils.Rectangle]:
        rectangles = []
        s = (self.room_size + self.wall_depth - self.hallway_width) / 2
        # Top wall.
        if hall_top:
            rect = utils.Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                theta=0,
            )
            rectangles.append(rect)
            rect = utils.Rectangle(
                x=(s + self.hallway_width + room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                theta=0,
            )
            rectangles.append(rect)
        else:
            rect = utils.Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=(self.room_size + self.wall_depth),
                theta=0,
            )
            rectangles.append(rect)

        # Bottom wall.
        if hall_bottom:
            rect = utils.Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                theta=0,
            )
            rectangles.append(rect)
            rect = utils.Rectangle(
                x=(s + self.hallway_width + room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                theta=0,
            )
            rectangles.append(rect)
        else:
            rect = utils.Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=(self.room_size + self.wall_depth),
                theta=0,
            )
            rectangles.append(rect)

        # Left wall.
        if hall_left:
            rect = utils.Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)
            rect = utils.Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + s + self.hallway_width - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)
        else:
            rect = utils.Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=(self.room_size + self.wall_depth),
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)

        # Right wall.
        if hall_right:
            rect = utils.Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)
            rect = utils.Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y + s + self.hallway_width - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)
        else:
            rect = utils.Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=(self.room_size + self.wall_depth),
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)

        return rectangles
