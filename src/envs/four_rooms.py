"""Four rooms environment based on the original options paper."""

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, Object, \
    ParameterizedOption, Predicate, State, Task, Type


@dataclass(frozen=True)
class _Rectangle:
    """A helper class for visualizing and collision-checking rectangles.

    Following the convention in plt.Rectangle, the origin is at the bottom
    left corner, and rotation is anti-clockwise about that point.
    """
    x: float
    y: float
    height: float
    width: float
    rot: float  # in radians, between -np.pi and np.pi

    def __post_init__(self):
        assert -np.pi <= self.rot <= np.pi, "Expecting angle in [-pi, pi]."

    @cached_property
    def vertices(self) -> List[Tuple[float, float]]:
        scale_matrix = np.array([
            [self.width, 0],
            [0, self.height],
        ])
        rotate_matrix = np.array([
            [np.cos(self.rot), -np.sin(self.rot)],
            [np.sin(self.rot), np.cos(self.rot)]
        ])
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
        return [(x, y) for (x, y) in vertices]

    @cached_property
    def line_segments(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        return list(zip(self.vertices, self.vertices[1:] + [self.vertices[0]]))


class FourRoomsEnv(BaseEnv):
    """An environment where a 2D robot must navigate between rooms that are
    connected by hallways to reach a target room.

    The robot is rectangular. The size of that rectangle varies between
    tasks in a way that sometimes requires the robot to rotate before
    moving through hallways. This setup is inspired by the situation where
    a person (or dog) is carrying a long stick and is trying to move through
    a narrow passage.

    The action space is 1D, which rotates the robot. At each time step, the
    rotation is applied first, and then the robot moves forward by a constant
    displacement.

    For consistency with plt.Rectangle, all rectangles are oriented at their
    bottom left corner.
    """
    room_size: ClassVar[float] = 1.0
    hallway_width: ClassVar[float] = 0.2
    wall_depth: ClassVar[float] = 0.01
    robot_min_width: ClassVar[float] = 0.1
    robot_max_width: ClassVar[float] = 0.3
    robot_height: ClassVar[float] = 0.05
    action_magnitude: ClassVar[float] = 0.05

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x", "y", "rot", "width"])
        self._room_type = Type("room", ["x", "y", "hall_top", "hall_bottom", "hall_left", "hall_right"])
        # Predicates
        self._InRoom = Predicate("InRoom",
                                  [self._robot_type, self._room_type],
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
        drot, = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        rot = state.get(self._robot, "rot")
        new_rot = np.clip(rot + drot, -np.pi, np.pi)
        new_x = x + np.cos(new_rot) * self.action_magnitude
        new_y = y + np.sin(new_rot) * self.action_magnitude
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        next_state.set(self._robot, "rot", new_rot)
        # Check for collisions.
        if self._state_has_collision(next_state):
            return state.copy()
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

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
        # An angle in radians. Constrained to prevent very large movements.
        return Box(-np.pi / 10, np.pi / 10, (1, ))

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        background_color = "white"
        # Draw robot.
        robot_color = "blue"
        robot_rect = self._get_rectangle_for_robot(state, self._robot)
        self._draw_rectangle(robot_rect, ax, color=robot_color)
        
        # if action:
        #     arrow_width = self.robot_height / 10
        #     drot, = action.arr
        #     dx = np.cos(rot + drot) * self.action_magnitude
        #     dy = np.sin(rot + drot) * self.action_magnitude
        #     ax.arrow(x, y, dx, dy, width=arrow_width, color="black")

        # Draw rooms.
        wall_color = "black"
        for room in state.get_objects(self._room_type):
            room_rects = self._get_rectangles_for_room(state, room)
            for rect in room_rects:
                self._draw_rectangle(rect, ax, color=wall_color)

        x_lb, x_ub, y_lb, y_ub = self._get_world_boundaries()
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
        top_left_room = Object("top_left_room", self._room_type)
        top_right_room = Object("top_right_room", self._room_type)
        bottom_left_room = Object("bottom_left_room", self._room_type)
        bottom_right_room = Object("bottom_right_room", self._room_type)
        init_state = utils.create_state_from_dict({
            self._robot: {
                "x": np.inf,  # will get overriden
                "y": np.inf,  # will get overriden
                "rot": np.inf,  # will get overriden
                "width": np.inf,  # will get overriden
            },
            top_left_room: {
                "x": -self.room_size,
                "y": 0,
                "hall_top": 0,
                "hall_bottom": 1,
                "hall_left": 0,
                "hall_right": 1,
            },
            top_right_room: {
                "x": 0,
                "y": 0,
                "hall_top": 0,
                "hall_bottom": 1,
                "hall_left": 1,
                "hall_right": 0,
            },
            bottom_left_room: {
                "x": -self.room_size,
                "y": -self.room_size,
                "hall_top": 1,
                "hall_bottom": 0,
                "hall_left": 0,
                "hall_right": 1,
            },
            bottom_right_room: {
                "x": 0,
                "y": -self.room_size,
                "hall_top": 1,
                "hall_bottom": 0,
                "hall_left": 1,
                "hall_right": 0,
            },
        })
        x_lb, x_ub, y_lb, y_ub = self._get_world_boundaries()
        rot_lb = -np.pi
        rot_ub = np.pi
        width_lb = self.robot_min_width
        width_ub = self.robot_max_width
        rooms = sorted(init_state.get_objects(self._room_type))
        while len(tasks) < num:
            # Sample an initial state.
            state = init_state.copy()
            x = rng.uniform(x_lb, x_ub)
            y = rng.uniform(y_lb, y_ub)
            rot = rng.uniform(rot_lb, rot_ub)
            width = rng.uniform(width_lb, width_ub)
            state.set(self._robot, "x", x)
            state.set(self._robot, "y", y)
            state.set(self._robot, "rot", rot)
            state.set(self._robot, "width", width)
            # Make sure the state is collision-free.
            if self._state_has_collision(state):
                continue
            # Sample a goal.
            goal_room_idx = rng.choice(len(rooms))
            goal_room = rooms[goal_room_idx]
            goal_atom = GroundAtom(self._InRoom, [self._robot, goal_room])
            goal = {goal_atom}
            # Make sure goal is not satisfied.
            if not goal_atom.holds(state):
                tasks.append(Task(state, goal))
        return tasks

    @staticmethod
    def _Move_policy(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> Action:
        del memory  # unused
        import ipdb; ipdb.set_trace()
        return Action(np.array([rot], dtype=np.float32))

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
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        room_x = state.get(room, "x")
        room_y = state.get(room, "y")
        return room_x < robot_x < room_x + self.room_size and \
               room_y < robot_y < room_y + self.room_size

    def _Connected_holds(self, state: State, objects: Sequence[Object]) -> bool:
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

    def _get_world_boundaries(self) -> Tuple[float, float, float, float]:
        x_lb = -self.room_size
        x_ub = self.room_size
        y_lb = -self.room_size
        y_ub = self.room_size
        return x_lb, x_ub, y_lb, y_ub

    def _get_rectangle_for_robot(self, state: State, robot: Object) -> _Rectangle:
        x = state.get(robot, "x")
        y = state.get(robot, "y")
        height = self.robot_height
        width = state.get(self._robot, "width")
        rot = state.get(robot, "rot")
        return _Rectangle(x, y, height, width, rot)

    def _get_rectangles_for_room(self, state: State, room: Object) -> List[_Rectangle]:
        rectangles = []
        room_x = state.get(room, "x")
        room_y = state.get(room, "y")
        s = (self.room_size + self.wall_depth - self.hallway_width) / 2
        # Top wall.
        if state.get(room, "hall_top"):
            rect = _Rectangle(
                x=(room_x - self.wall_depth/2),
                y=(room_y + self.room_size - self.wall_depth/2),
                height=self.wall_depth,
                width=s,
                rot=0,
            )
            rectangles.append(rect)
            rect = _Rectangle(
                x=(s + self.hallway_width + room_x - self.wall_depth/2),
                y=(room_y + self.room_size - self.wall_depth/2),
                height=self.wall_depth,
                width=s,
                rot=0,
            )
            rectangles.append(rect)

        else:
            rect = _Rectangle(
                x=(room_x - self.wall_depth/2),
                y=(room_y + self.room_size - self.wall_depth/2),
                height=self.wall_depth,
                width=(self.room_size + self.wall_depth),
                rot=0,
            )
            rectangles.append(rect)

        # Bottom wall.
        if state.get(room, "hall_bottom"):
            rect = _Rectangle(
                x=(room_x - self.wall_depth/2),
                y=(room_y - self.wall_depth/2),
                height=self.wall_depth,
                width=s,
                rot=0,
            )
            rectangles.append(rect)
            rect = _Rectangle(
                x=(s + self.hallway_width + room_x - self.wall_depth/2),
                y=(room_y - self.wall_depth/2),
                height=self.wall_depth,
                width=s,
                rot=0,
            )
            rectangles.append(rect)
        else:
            rect = _Rectangle(
                x=(room_x - self.wall_depth/2),
                y=(room_y - self.wall_depth/2),
                height=self.wall_depth,
                width=(self.room_size + self.wall_depth),
                rot=0,
            )
            rectangles.append(rect)

        # Left wall.
        if state.get(room, "hall_left"):
            rect = _Rectangle(
                x=(room_x - self.wall_depth/2),
                y=(room_y - self.wall_depth/2),
                height=s,
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)
            rect = _Rectangle(
                x=(room_x - self.wall_depth/2),
                y=(room_y + s + self.hallway_width - self.wall_depth/2),
                height=s,
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)
        else:
            rect = _Rectangle(
                x=(room_x - self.wall_depth/2),
                y=(room_y - self.wall_depth/2),
                height=(self.room_size + self.wall_depth),
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)

        # Right wall.
        if state.get(room, "hall_right"):
            rect = _Rectangle(
                x=(room_x + self.room_size - self.wall_depth/2),
                y=(room_y - self.wall_depth/2),
                height=s,
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)
            rect = _Rectangle(
                x=(room_x + self.room_size - self.wall_depth/2),
                y=(room_y + s + self.hallway_width - self.wall_depth/2),
                height=s,
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)
        else:
            rect = _Rectangle(
                x=(room_x + self.room_size - self.wall_depth/2),
                y=(room_y - self.wall_depth/2),
                height=(self.room_size + self.wall_depth),
                width=self.wall_depth,
                rot=0,
            )
            rectangles.append(rect)

        return rectangles

    @staticmethod
    def _draw_rectangle(rectangle: _Rectangle, ax: plt.Axes, **kwargs: Any) -> None:
        x = rectangle.x
        y = rectangle.y
        w = rectangle.width
        h = rectangle.height
        angle = rectangle.rot * 180 / np.pi
        rect = patches.Rectangle((x, y), w, h, angle, **kwargs)
        ax.add_patch(rect)

        # For debugging.
        # for (vx, vy) in rectangle.vertices:
        #     circ = patches.Circle((vx, vy), 0.01, color="red", alpha=0.5)
        #     ax.add_patch(circ)
        # for ((x1, y1), (x2, y2)) in rectangle.line_segments:
        #     ax.plot([x1, x2], [y1, y2], color="red", lw=0.1)

    @staticmethod
    def _rectangles_intersect(rect1: _Rectangle, rect2: _Rectangle) -> bool:
        for (p1, p2) in rect1.line_segments:
            for (p3, p4) in rect2.line_segments:
                if utils.intersects(p1, p2, p3, p4):
                    return True
        return False
