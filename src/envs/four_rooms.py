"""Four rooms environment based on the original options paper."""

from dataclasses import dataclass
from functools import cached_property
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


@dataclass(frozen=True)
class _Rectangle:
    """A helper class for visualizing and collision-checking rectangles.

    Following the convention in plt.Rectangle, the origin is at the
    bottom left corner, and rotation is anti-clockwise about that point.
    """
    x: float
    y: float
    height: float
    width: float
    rot: float  # in radians, between -np.pi and np.pi

    def __post_init__(self) -> None:
        assert -np.pi <= self.rot <= np.pi, "Expecting angle in [-pi, pi]."

    @cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        """Get the four vertices for the rectangle."""
        scale_matrix = np.array([
            [self.width, 0],
            [0, self.height],
        ])
        rotate_matrix = np.array([[np.cos(self.rot), -np.sin(self.rot)],
                                  [np.sin(self.rot),
                                   np.cos(self.rot)]])
        translate_vector = np.array([self.x, self.y])
        vertices = np.array([
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        ])
        vertices = vertices @ scale_matrix.T
        vertices = vertices @ rotate_matrix.T
        vertices = translate_vector + vertices
        # Convert to a list of tuples. Slightly complicated to appease both
        # type checking and linting.
        return list(map(lambda p: (p[0], p[1]), vertices))

    @cached_property
    def line_segments(
            self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Get the four line segments for the rectangle."""
        return list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))

    @cached_property
    def center(self) -> Tuple[float, float]:
        """Get the point at the center of the rectangle."""
        x, y = np.mean(self.vertices, axis=0)
        return (x, y)

    @cached_property
    def circumscribed_circle(self) -> Tuple[float, float, float]:
        """Returns x, y, radius."""
        x, y = self.center
        radius = np.sqrt(self.width**2 + self.height**2)
        return x, y, radius


class FourRoomsEnv(BaseEnv):
    """An environment where a 2D robot must navigate between rooms that are
    connected by hallways to reach a target room.

    The robot is rectangular. The size of that rectangle varies between
    tasks in a way that sometimes requires the robot to rotate before
    moving through hallways. This setup is inspired by the situation where
    a person (or dog) is carrying a long stick and is trying to move through
    a narrow passage.

    The action space is 2D. The first dimension rotates the robot and the
    second dimension indicates the direction of movement (both are angles).
    """
    room_size: ClassVar[float] = 1.0
    hallway_width: ClassVar[float] = 0.25
    wall_depth: ClassVar[float] = 0.01
    robot_min_width: ClassVar[float] = 0.1
    robot_max_width: ClassVar[float] = 0.3
    robot_height: ClassVar[float] = 0.05
    action_magnitude: ClassVar[float] = 0.05
    robot_initial_position_radius: ClassVar[float] = 0.05
    rot_max_magnitude: ClassVar[float] = np.pi / 10
    train_room_map: ClassVar[NDArray[np.uint8]] = np.array([
        [1, 1],
        [1, 1],
    ])
    test_room_map: ClassVar[NDArray[np.uint8]] = np.array([
        [1, 1],
        [1, 1],
    ])

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x", "y", "rot", "width"])
        self._room_type = Type(
            "room",
            ["x", "y", "hall_top", "hall_bottom", "hall_left", "hall_right"])
        # Predicates
        self._InRoom = Predicate("InRoom", [self._robot_type, self._room_type],
                                 self._InRoom_holds)
        self._Connected = Predicate("Connected",
                                    [self._room_type, self._room_type],
                                    self._Connected_holds)
        # Options
        self._Move = ParameterizedOption(
            "Move",
            types=[self._robot_type, self._room_type, self._room_type],
            # The parameter is the angle of rotation with which the robot
            # will pass through the hallway between the rooms.
            params_space=Box(-np.pi, np.pi, (1, )),
            policy=self._Move_policy,
            initiable=self._Move_initiable,
            terminal=self._Move_terminal)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "four_rooms"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        drot, theta = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        rot = state.get(self._robot, "rot")
        new_rot = np.clip(rot + drot, -np.pi, np.pi)
        new_x = x + np.cos(theta) * self.action_magnitude
        new_y = y + np.sin(theta) * self.action_magnitude
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        next_state.set(self._robot, "rot", new_rot)
        # Check for collisions.
        if self._state_has_collision(next_state):
            return state.copy()
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               room_map=self.train_room_map,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               room_map=self.test_room_map,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InRoom, self._Connected}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InRoom}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._room_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Move}

    @property
    def action_space(self) -> Box:
        # Angles in radians. The first dimension rotates the body, and is
        # constrained to prevent large movements. The second dimension is
        # the direction that the body moves.
        lb = np.array([-self.rot_max_magnitude, -np.pi], dtype=np.float32)
        ub = np.array([self.rot_max_magnitude, np.pi], dtype=np.float32)
        return Box(lb, ub, (2, ))

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        # Draw robot.
        robot_color = "blue"
        robot_rect = self._get_rectangle_for_robot(state, self._robot)
        self._draw_rectangle(robot_rect, ax, color=robot_color)

        # Draw rooms.
        wall_color = "black"
        for room in state.get_objects(self._room_type):
            room_rects = self._get_rectangles_for_room(state, room)
            for rect in room_rects:
                self._draw_rectangle(rect, ax, color=wall_color)

        # Label the goal room with a star.
        assert len(task.goal) == 1
        goal_room = next(iter(task.goal)).objects[1]
        room_x = state.get(goal_room, "x")
        room_y = state.get(goal_room, "y")
        cx = room_x + self.room_size / 2
        cy = room_y + self.room_size / 2
        map_size = len(list(state))
        star_size = 1200 / map_size
        ax.scatter(cx, cy, s=star_size, marker='*', color='gold')

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

    def _get_tasks(self, num: int, room_map: NDArray[np.uint8],
                   rng: np.random.Generator) -> List[Task]:
        tasks: List[Task] = []
        # Create the static parts of the initial state.
        room_state_dict = {}
        num_rows, num_cols = room_map.shape
        for (r, c) in np.argwhere(room_map):
            room = Object(f"room{r}-{c}", self._room_type)
            hall_top = float(r > 0 and room_map[r - 1, c])
            hall_bottom = float(r < num_rows - 1 and room_map[r + 1, c])
            hall_left = float(c > 0 and room_map[r, c - 1])
            hall_right = float(c < num_cols - 1 and room_map[r, c + 1])
            room_state_dict[room] = {
                "x": float(c * self.room_size),
                "y": float((num_rows - 1 - r) * self.room_size),
                "hall_top": hall_top,
                "hall_bottom": hall_bottom,
                "hall_left": hall_left,
                "hall_right": hall_right,
            }
        init_state = utils.create_state_from_dict({
            **room_state_dict,
            # Will get overriden.
            self._robot: {
                "x": np.inf,
                "y": np.inf,
                "rot": np.inf,
                "width": np.inf,
            },
        })
        rot_lb = -np.pi / 10
        rot_ub = np.pi / 2 + np.pi / 10
        width_lb = self.robot_min_width
        width_ub = self.robot_max_width
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
            rot = rng.uniform(rot_lb, rot_ub)
            width = rng.uniform(width_lb, width_ub)
            state.set(self._robot, "x", x)
            state.set(self._robot, "y", y)
            state.set(self._robot, "rot", rot)
            state.set(self._robot, "width", width)
            # Make sure the state is collision-free. Should be guaranteed
            # because we're initializing the robot in the middle of the room.
            assert not self._state_has_collision(state)
            # Sample a goal.
            goal_atom = GroundAtom(self._InRoom, [self._robot, goal_room])
            goal = {goal_atom}
            assert not goal_atom.holds(state)
            tasks.append(Task(state, goal))
        return tasks

    def _Move_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        del memory  # unused
        robot, start_room, end_room = objects
        desired_rot, = params
        # Get the center of the robot.
        rect = self._get_rectangle_for_robot(state, robot)
        x, y = rect.center
        rot = state.get(robot, "rot")
        room_x = state.get(start_room, "x")
        room_y = state.get(start_room, "y")
        room_cx = room_x + self.room_size / 2
        room_cy = room_y + self.room_size / 2
        dist_to_center = np.sqrt((x - room_cx)**2 + (y - room_cy)**2)
        near_center = (dist_to_center < 1.1 * self.action_magnitude)
        dist_to_desired_rot = abs(rot - desired_rot)
        at_desired_rot = dist_to_desired_rot < 1e-6
        drot = np.clip(desired_rot - rot, -self.rot_max_magnitude,
                       self.rot_max_magnitude)
        theta_to_center = np.arctan2((room_cy - y), (room_cx - x))
        # If already at the desired rotation, move toward the hallway.
        if at_desired_rot:
            cx, cy = self._get_hallway_position(state, start_room, end_room)
            theta = np.arctan2((cy - y), (cx - x))
        # If near enough to the center of the room, rotate, and continue
        # moving toward the center.
        elif near_center:
            theta = theta_to_center
        # If not yet near the center of the room, move without rotating.
        else:
            drot = 0
            theta = theta_to_center
        return Action(np.array([drot, theta], dtype=np.float32))

    def _Move_initiable(self, state: State, memory: Dict,
                        objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        robot, start_room, target_room = objects
        return self._InRoom_holds(state, [robot, start_room]) and \
            self._Connected_holds(state, [start_room, target_room])

    def _Move_terminal(self, state: State, memory: Dict,
                       objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        robot, _, target_room = objects
        return self._InRoom_holds(state, [robot, target_room])

    def _InRoom_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, room = objects
        # Use the center of the robot to determine if it's in the room.
        rect = self._get_rectangle_for_robot(state, robot)
        robot_x, robot_y = rect.center
        room_x = state.get(room, "x")
        room_y = state.get(room, "y")
        return room_x < robot_x < room_x + self.room_size and \
               room_y < robot_y < room_y + self.room_size

    def _Connected_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        r1, r2 = objects
        x1 = state.get(r1, "x")
        x2 = state.get(r2, "x")
        y1 = state.get(r1, "y")
        y2 = state.get(r2, "y")
        # Case: room 1 is above room 2.
        if abs(x1 - x2) < 1e-7 and abs((y1 - self.room_size) - y2) < 1e-7:
            return state.get(r1, "hall_bottom") and state.get(r2, "hall_top")
        # Case: room 1 is below room 2.
        if abs(x1 - x2) < 1e-7 and abs(y1 - (y2 - self.room_size)) < 1e-7:
            return state.get(r1, "hall_top") and state.get(r2, "hall_bottom")
        # Case: room 1 is right of room 2.
        if abs((x1 - self.room_size) - x2) < 1e-7 and abs(y1 - y2) < 1e-7:
            return state.get(r1, "hall_left") and state.get(r2, "hall_right")
        # Case: room 1 is left of room 2.
        if abs(x1 - (x2 - self.room_size)) < 1e-7 and abs(y1 - y2) < 1e-7:
            return state.get(r1, "hall_right") and state.get(r2, "hall_left")
        return False

    def _state_has_collision(self, state: State) -> bool:
        robot_rect = self._get_rectangle_for_robot(state, self._robot)
        for room in state.get_objects(self._room_type):
            for rect in self._get_rectangles_for_room(state, room):
                if self._rectangles_intersect(robot_rect, rect):
                    return True
        return False

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

    def _get_rectangle_for_robot(self, state: State,
                                 robot: Object) -> _Rectangle:
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        height = self.robot_height
        width = state.get(self._robot, "width")
        rot = state.get(robot, "rot")
        return _Rectangle(x, y, height, width, rot)

    def _get_rectangles_for_room(self, state: State,
                                 room: Object) -> List[_Rectangle]:
        rectangles = []
        room_x = state.get(room, "x")
        room_y = state.get(room, "y")
        s = (self.room_size + self.wall_depth - self.hallway_width) / 2
        # Top wall.
        if state.get(room, "hall_top"):
            rect = _Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                rot=0,
            )
            rectangles.append(rect)
            rect = _Rectangle(
                x=(s + self.hallway_width + room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                rot=0,
            )
            rectangles.append(rect)

        else:
            rect = _Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=(self.room_size + self.wall_depth),
                rot=0,
            )
            rectangles.append(rect)

        # Bottom wall.
        if state.get(room, "hall_bottom"):
            rect = _Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                rot=0,
            )
            rectangles.append(rect)
            rect = _Rectangle(
                x=(s + self.hallway_width + room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                rot=0,
            )
            rectangles.append(rect)
        else:
            rect = _Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=(self.room_size + self.wall_depth),
                rot=0,
            )
            rectangles.append(rect)

        # Left wall.
        if state.get(room, "hall_left"):
            rect = _Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)
            rect = _Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + s + self.hallway_width - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)
        else:
            rect = _Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=(self.room_size + self.wall_depth),
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)

        # Right wall.
        if state.get(room, "hall_right"):
            rect = _Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)
            rect = _Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y + s + self.hallway_width - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)
        else:
            rect = _Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=(self.room_size + self.wall_depth),
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)

        return rectangles

    @staticmethod
    def _draw_rectangle(rectangle: _Rectangle, ax: plt.Axes,
                        **kwargs: Any) -> None:
        x = rectangle.x
        y = rectangle.y
        w = rectangle.width
        h = rectangle.height
        angle = rectangle.rot * 180 / np.pi
        rect = patches.Rectangle((x, y), w, h, angle, **kwargs)
        ax.add_patch(rect)

    @staticmethod
    def _rectangles_intersect(rect1: _Rectangle, rect2: _Rectangle) -> bool:
        # Optimization: if the circumscribed circles don't intersect, then
        # the rectangles also don't intersect.
        x1, y1, r1 = rect1.circumscribed_circle
        x2, y2, r2 = rect2.circumscribed_circle
        if not abs(r1 - r2) <= np.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= r1 + r2:
            return False
        for (p1, p2) in rect1.line_segments:
            for (p3, p4) in rect2.line_segments:
                if utils.intersects(p1, p2, p3, p4):
                    return True
        return False

    def _get_hallway_position(self, state: State, room1: Object,
                              room2: Object) -> Tuple[float, float]:
        assert self._Connected_holds(state, [room1, room2])
        x1 = state.get(room1, "x")
        x2 = state.get(room2, "x")
        y1 = state.get(room1, "y")
        y2 = state.get(room2, "y")
        # Case: room 1 is above room 2.
        if abs(x1 - x2) < 1e-7 and abs((y1 - self.room_size) - y2) < 1e-7:
            return (x1 + self.room_size / 2, y1)
        # Case: room 1 is below room 2.
        if abs(x1 - x2) < 1e-7 and abs(y1 - (y2 - self.room_size)) < 1e-7:
            return (x2 + self.room_size / 2, y2)
        # Case: room 1 is right of room 2.
        if abs((x1 - self.room_size) - x2) < 1e-7 and abs(y1 - y2) < 1e-7:
            return (x1, y1 + self.room_size / 2)
        # Case: room 1 is left of room 2.
        assert abs(x1 - (x2 - self.room_size)) < 1e-7 and abs(y1 - y2) < 1e-7
        return (x2, y2 + self.room_size / 2)


class FourRoomsGeneralizeEnv(FourRoomsEnv):
    """A variation on the four rooms environment where the test tasks require
    generalizing to more than four rooms."""

    test_room_map: ClassVar[NDArray[np.uint8]] = np.array([
        [1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 0, 1],
        [0, 1, 1, 1, 1],
        [1, 1, 1, 0, 1],
    ])

    @classmethod
    def get_name(cls) -> str:
        return "four_rooms_generalize"
