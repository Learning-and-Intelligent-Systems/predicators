"""A 2D navigation environment with obstacles, rooms, and doors."""

import itertools
from typing import ClassVar, Dict, Iterator, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from numpy.typing import NDArray

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type
from predicators.utils import Rectangle, StateWithCache, _Geom2D


class DoorsEnv(BaseEnv):
    """A 2D navigation environment with obstacles, rooms, and doors."""
    room_size: ClassVar[float] = 1.0
    hallway_width: ClassVar[float] = 0.25
    wall_depth: ClassVar[float] = 0.01
    robot_radius: ClassVar[float] = 0.05
    action_magnitude: ClassVar[float] = 0.05
    robot_initial_position_radius: ClassVar[float] = 0.05
    obstacle_initial_position_radius: ClassVar[float] = 0.1
    obstacle_size_lb: ClassVar[float] = 0.05
    obstacle_size_ub: ClassVar[float] = 0.15
    doorway_pad: ClassVar[float] = 1e-3
    move_sq_dist_tol: ClassVar[float] = 1e-5
    open_door_thresh: ClassVar[float] = 1e-2

    # Types
    _robot_type = Type("robot", ["x", "y"])
    _door_type = Type(
        "door",
        ["x", "y", "theta", "mass", "friction", "rot", "target_rot", "open"])
    _room_type = Type("room", ["x", "y"])
    _obstacle_type = Type("obstacle", ["x", "y", "width", "height", "theta"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._InRoom = Predicate("InRoom", [self._robot_type, self._room_type],
                                 self._InRoom_holds)
        self._InDoorway = Predicate("InDoorway",
                                    [self._robot_type, self._door_type],
                                    self._InDoorway_holds)
        self._InMainRoom = Predicate("InMainRoom",
                                     [self._robot_type, self._room_type],
                                     self._InMainRoom_holds)
        self._TouchingDoor = Predicate("TouchingDoor",
                                       [self._robot_type, self._door_type],
                                       self._TouchingDoor_holds)
        self._DoorIsOpen = Predicate("DoorIsOpen", [self._door_type],
                                     self._DoorIsOpen_holds)
        self._DoorInRoom = Predicate("DoorInRoom",
                                     [self._door_type, self._room_type],
                                     self._DoorInRoom_holds)
        # This predicate is needed as a precondition for moving from one
        # door to another door in the same room.
        self._DoorsShareRoom = Predicate("DoorsShareRoom",
                                         [self._door_type, self._door_type],
                                         self._DoorsShareRoom_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        # Hyperparameters from CFG.
        self._room_map_size = CFG.doors_room_map_size
        self._min_obstacles_per_room = CFG.doors_min_obstacles_per_room
        self._max_obstacles_per_room = CFG.doors_max_obstacles_per_room
        self._min_room_exists_frac = CFG.doors_min_room_exists_frac
        self._max_room_exists_frac = CFG.doors_max_room_exists_frac
        # See note in _sample_initial_state_from_map().
        self._task_id_count = itertools.count()

    @classmethod
    def get_name(cls) -> str:
        return "doors"

    def simulate_moving(self, state: State, action: Action) -> State:
        """helper function to simulate moving."""
        assert self.action_space.contains(action.arr)
        dx, dy, _ = action.arr
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        new_x = x + dx
        new_y = y + dy
        next_state = state.copy()
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        # Check for collisions.
        if self.state_has_collision(next_state):
            # Revert the change to the robot position.
            next_state.set(self._robot, "x", x)
            next_state.set(self._robot, "y", y)
        return next_state

    def simulate(self, state: State, action: Action) -> State:
        new_door_rot = action.arr[2]
        next_state = self.simulate_moving(state, action)
        # If touching a door, change its value based on the action.
        for door in state.get_objects(self._door_type):
            if self._TouchingDoor_holds(state, [self._robot, door]):
                # Rotate the door handle.
                next_state.set(door, "rot", new_door_rot)
                # Check if we should now open the door.
                target = self._get_open_door_target_value(
                    mass=state.get(door, "mass"),
                    friction=state.get(door, "friction"),
                    target_rot=state.get(door, "target_rot"),
                )
                if abs(new_door_rot - target) < self.open_door_thresh:
                    next_state.set(door, "open", 1.0)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._InRoom,
            self._InDoorway,
            self._TouchingDoor,
            self._DoorIsOpen,
            self._DoorInRoom,
            self._InMainRoom,
            self._DoorsShareRoom,
        }

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
    def action_space(self) -> Box:
        # dx, dy, drot
        lb = np.array(
            [-self.action_magnitude, -self.action_magnitude, -np.inf],
            dtype=np.float32)
        ub = np.array([self.action_magnitude, self.action_magnitude, np.inf],
                      dtype=np.float32)
        return Box(lb, ub)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        del caption  # unused
        x_lb, x_ub, y_lb, y_ub = self._get_world_boundaries(state)
        fig, ax = plt.subplots(1, 1, figsize=(x_ub - x_lb, y_ub - y_lb))

        # Draw rooms.
        default_room_color = "lightgray"
        in_room_color = "lightsteelblue"
        goal_room_color = "khaki"
        goal_room = next(iter(task.goal)).objects[1]
        for room in state.get_objects(self._room_type):
            room_geom = self.object_to_geom(room, state)
            if room == goal_room:
                color = goal_room_color
            elif self._InRoom_holds(state, [self._robot, room]):
                color = in_room_color
            else:
                color = default_room_color
            room_geom.plot(ax, color=color)

        # Draw robot.
        robot_color = "blue"
        robot_geom = self.object_to_geom(self._robot, state)
        robot_geom.plot(ax, color=robot_color)

        # Draw obstacles (including room walls).
        obstacle_color = "black"
        for obstacle in state.get_objects(self._obstacle_type):
            obstacle_geom = self.object_to_geom(obstacle, state)
            obstacle_geom.plot(ax, color=obstacle_color)

        # Draw doors.
        closed_door_color = "orangered"
        open_door_color = "lightgreen"
        doorway_color = "darkviolet"
        for door in state.get_objects(self._door_type):
            if self._DoorIsOpen_holds(state, [door]):
                color = open_door_color
            else:
                color = closed_door_color
            door_geom = self.object_to_geom(door, state)
            door_geom.plot(ax, color=color)
            if CFG.doors_draw_debug:
                doorway_geom = self.door_to_doorway_geom(door, state)
                doorway_geom.plot(ax, color=doorway_color, alpha=0.1)

        # Visualize the motion plan.
        if CFG.doors_draw_debug:
            if action is not None and action.has_option():
                option = action.get_option()
                if "position_plan" in option.memory:
                    xs, ys = zip(*option.memory["position_plan"])
                    plt.scatter(xs, ys, s=3, alpha=0.5, color=robot_color)

        pad = 2 * self.wall_depth
        ax.set_xlim(x_lb - pad, x_ub + pad)
        ax.set_ylim(y_lb - pad, y_ub + pad)

        plt.axis("off")
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []
        for _ in range(num):
            # Sample a room map.
            room_map = self._sample_room_map(rng)
            state = self._sample_initial_state_from_map(room_map, rng)
            # Sample the goal.
            rooms = state.get_objects(self._room_type)
            candidate_goal_rooms = [
                r for r in rooms
                if not self._InRoom_holds(state, [self._robot, r])
            ]
            goal_room = candidate_goal_rooms[rng.choice(
                len(candidate_goal_rooms))]
            goal_atom = GroundAtom(self._InRoom, [self._robot, goal_room])
            goal = {goal_atom}
            assert not goal_atom.holds(state)
            tasks.append(EnvironmentTask(state, goal))
        return tasks

    def _sample_initial_state_from_map(self, room_map: NDArray,
                                       rng: np.random.Generator) -> State:
        # Sample until a collision-free state is found.
        while True:
            # For each task, we create a unique ID, which is included in the
            # names of the objects created. This is important because we then
            # perform caching based on the object names. For example, we want
            # to compute the doors for a room only once. But the same room in
            # the same location may have different doors between tasks, so we
            # need to be careful to avoid accidental name collisions.
            task_id = next(self._task_id_count)
            state_dict = {}
            num_rows, num_cols = room_map.shape
            # Create the rooms.
            rooms = []
            for (r, c) in np.argwhere(room_map):
                room = Object(f"room{task_id}-{r}-{c}", self._room_type)
                rooms.append(room)
                room_x = float(c * self.room_size)
                room_y = float((num_rows - 1 - r) * self.room_size)
                state_dict[room] = {
                    "x": room_x,
                    "y": room_y,
                }
                # Create obstacles for the room walls.
                hall_top = (r > 0 and room_map[r - 1, c])
                hall_bottom = (r < num_rows - 1 and room_map[r + 1, c])
                hall_left = (c > 0 and room_map[r, c - 1])
                hall_right = (c < num_cols - 1 and room_map[r, c + 1])
                wall_rects = self._get_rectangles_for_room_walls(
                    room_x, room_y, hall_top, hall_bottom, hall_left,
                    hall_right)
                for i, rect in enumerate(wall_rects):
                    wall = Object(f"wall{task_id}-{r}-{c}-{i}",
                                  self._obstacle_type)
                    state_dict[wall] = {
                        "x": rect.x,
                        "y": rect.y,
                        "height": rect.height,
                        "width": rect.width,
                        "theta": rect.theta,
                    }
                # Create doors for this room. Note that we only need to create
                # bottom or left, because each door is on the bottom or left of
                # some room (and top / right of another room).
                for name, exists in [("bottom", hall_bottom),
                                     ("left", hall_left)]:
                    if not exists:
                        continue
                    door = Object(f"{name}-door{task_id}-{r}-{c}",
                                  self._door_type)
                    feat_dict = self._sample_door_feats(
                        room_x, room_y, name, rng)
                    state_dict[door] = feat_dict

                    if self.get_name() == "doorknobs":
                        assert isinstance(self, DoorKnobsEnv)
                        #Create doorknobs
                        doorknob = Object(
                            f"room{task_id}-{r}-{c}-{name}-doorknob",
                            self._knob_type)
                        feat_dict = self._sample_doorknob_feats(
                            room_x, room_y, name, rng)
                        state_dict[doorknob] = feat_dict
                        self._door_to_knob[door] = doorknob
            # Sample obstacles for each room. Make them small and centered
            # enough that the robot should almost always be able to find a
            # collision-free path through the room.

            for room in rooms:
                room_x = state_dict[room]["x"]
                room_y = state_dict[room]["y"]
                room_cx = room_x + self.room_size / 2
                room_cy = room_y + self.room_size / 2
                rad = self.obstacle_initial_position_radius
                num_obstacles = rng.integers(self._min_obstacles_per_room,
                                             self._max_obstacles_per_room + 1)
                obstacle_rects_for_room: List[Rectangle] = []
                for i in range(num_obstacles):
                    name = f"{room.name}-obstacle-{i}"
                    obstacle = Object(name, self._obstacle_type)
                    while True:
                        x = rng.uniform(room_cx - rad, room_cx + rad)
                        y = rng.uniform(room_cy - rad, room_cy + rad)
                        w = rng.uniform(self.obstacle_size_lb,
                                        self.obstacle_size_ub)
                        h = rng.uniform(self.obstacle_size_lb,
                                        self.obstacle_size_ub)
                        theta = rng.uniform(-np.pi, np.pi)
                        rect = Rectangle(x=x,
                                         y=y,
                                         width=w,
                                         height=h,
                                         theta=theta)
                        # Prevent collisions just for aesthetic reasons.
                        collision_free = True
                        for existing_rect in obstacle_rects_for_room:
                            if rect.intersects(existing_rect):
                                collision_free = False
                                break
                        if collision_free:
                            break
                    obstacle_rects_for_room.append(rect)
                    state_dict[obstacle] = {
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "theta": theta
                    }
            # Always start out near the center of the room. If there are
            # collisions, we'll just resample another problem.
            start_idx = rng.choice(len(rooms))
            start_room = rooms[start_idx]
            room_x = state_dict[start_room]["x"]
            room_y = state_dict[start_room]["y"]
            room_cx = room_x + self.room_size / 2
            room_cy = room_y + self.room_size / 2
            rad = self.obstacle_initial_position_radius
            x = rng.uniform(room_cx - rad, room_cx + rad)
            y = rng.uniform(room_cy - rad, room_cy + rad)
            state_dict[self._robot] = {"x": x, "y": y}
            state = utils.create_state_from_dict(state_dict)
            # Create task-constant caches and store them in the sim state.
            # We store in the sim state, rather than the environment, because
            # the caches may be used by oracle options (which are external).
            task_cache: Dict[str, Dict] = {
                "static_geom": {},
                "door_to_rooms": {},
                "room_to_doors": {},
                "door_to_doorway_geom": {},
                "position_in_doorway": {},
            }
            state_with_cache = utils.StateWithCache(state.data, task_cache)
            if not self.state_has_collision(state_with_cache):
                return state_with_cache

    def _InRoom_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # The robot is in the room if its center is in the room.
        robot, room = objects
        robot_geom = self.object_to_geom(robot, state)
        assert isinstance(robot_geom, utils.Circle)
        room_geom = self.object_to_geom(room, state)
        return room_geom.contains_point(robot_geom.x, robot_geom.y)

    def _InDoorway_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        robot, door = objects
        doorway_geom = self.door_to_doorway_geom(door, state)
        robot_geom = self.object_to_geom(robot, state)
        return robot_geom.intersects(doorway_geom)

    def _InMainRoom_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        robot, room = objects
        if not self._InRoom_holds(state, [robot, room]):
            return False
        for door in self._room_to_doors(room, state):
            if self._InDoorway_holds(state, [robot, door]):
                return False
        return True

    def _TouchingDoor_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        _, door = objects
        # Once the door is open, the robot is no longer touching it.
        if self._DoorIsOpen_holds(state, [door]):
            return False
        # The robot is considered to be touching the door if it's in the
        # doorway for that door. Note that we don't want to check if the
        # robot is literally touching the door, because collision checking
        # will forbid that from ever happening.
        return self._InDoorway_holds(state, objects)

    @staticmethod
    def _DoorIsOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        door, = objects
        return state.get(door, "open") > 0.5

    def _DoorInRoom_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        door, room = objects
        return door in self._room_to_doors(room, state)

    def _DoorsShareRoom_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        door1, door2 = objects
        # Open to debate, but let's enforce this...
        if door1 == door2:
            return False
        rooms1 = self.door_to_rooms(door1, state)
        rooms2 = self.door_to_rooms(door2, state)
        return len(rooms1 & rooms2) > 0

    @classmethod
    def state_has_collision(cls, state: State) -> bool:
        """Public for use by oracle options."""
        robot, = state.get_objects(cls._robot_type)
        robot_geom = cls.object_to_geom(robot, state)
        # Check for collisions with obstacles.
        for obstacle in state.get_objects(cls._obstacle_type):
            obstacle_geom = cls.object_to_geom(obstacle, state)
            if robot_geom.intersects(obstacle_geom):
                return True
        # Check for collisions with closed doors.
        for door in state.get_objects(cls._door_type):
            if cls._DoorIsOpen_holds(state, [door]):
                continue
            door_geom = cls.object_to_geom(door, state)
            if robot_geom.intersects(door_geom):
                return True
        return False

    @classmethod
    def object_to_geom(cls, obj: Object, state: State) -> _Geom2D:
        """Public for use by oracle options."""
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        if obj.is_instance(cls._robot_type):
            return utils.Circle(x, y, cls.robot_radius)
        # Only the robot shape is dynamic. All other shapes are cached.
        assert isinstance(state, StateWithCache)
        static_geom_cache = state.cache["static_geom"]
        if obj not in static_geom_cache:
            if obj.is_instance(cls._room_type):
                width = cls.room_size
                height = cls.room_size
                theta = 0.0
            elif obj.is_instance(cls._door_type):
                width = cls.hallway_width
                height = cls.wall_depth
                theta = state.get(obj, "theta")
            else:
                assert obj.is_instance(cls._obstacle_type)
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "theta")
            geom = Rectangle(x=x, y=y, width=width, height=height, theta=theta)
            static_geom_cache[obj] = geom
        return static_geom_cache[obj]

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

    def _get_rectangles_for_room_walls(self, room_x: float, room_y: float,
                                       hall_top: bool, hall_bottom: bool,
                                       hall_left: bool,
                                       hall_right: bool) -> List[Rectangle]:
        rectangles = []
        s = (self.room_size + self.wall_depth - self.hallway_width) / 2
        # Top wall.
        if hall_top:
            rect = Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                theta=0,
            )
            rectangles.append(rect)
            rect = Rectangle(
                x=(s + self.hallway_width + room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                theta=0,
            )
            rectangles.append(rect)
        else:
            rect = Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + self.room_size - self.wall_depth / 2),
                height=self.wall_depth,
                width=(self.room_size + self.wall_depth),
                theta=0,
            )
            rectangles.append(rect)

        # Bottom wall.
        if hall_bottom:
            rect = Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                theta=0,
            )
            rectangles.append(rect)
            rect = Rectangle(
                x=(s + self.hallway_width + room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=s,
                theta=0,
            )
            rectangles.append(rect)
        else:
            rect = Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=self.wall_depth,
                width=(self.room_size + self.wall_depth),
                theta=0,
            )
            rectangles.append(rect)

        # Left wall.
        if hall_left:
            rect = Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)
            rect = Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y + s + self.hallway_width - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)
        else:
            rect = Rectangle(
                x=(room_x - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=(self.room_size + self.wall_depth),
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)

        # Right wall.
        if hall_right:
            rect = Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)
            rect = Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y + s + self.hallway_width - self.wall_depth / 2),
                height=s,
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)
        else:
            rect = Rectangle(
                x=(room_x + self.room_size - self.wall_depth / 2),
                y=(room_y - self.wall_depth / 2),
                height=(self.room_size + self.wall_depth),
                width=self.wall_depth,
                theta=0,
            )
            rectangles.append(rect)

        return rectangles

    def _sample_door_feats(self, room_x: float, room_y: float, loc: str,
                           rng: np.random.Generator) -> Dict[str, float]:
        # This is the length of the wall on one side of the door.
        offset = (self.room_size + self.wall_depth - self.hallway_width) / 2

        if loc == "bottom":
            x = room_x + offset
            y = room_y - self.wall_depth / 2
            theta = 0.0
        else:
            assert loc == "left"
            x = room_x + self.wall_depth / 2
            y = room_y + offset
            theta = np.pi / 2

        mass, friction, target_rot = rng.uniform(0.0, 1.0, size=(3, ))
        # Sample the initial rotation so that the door is not yet opened.
        while True:
            rot = rng.uniform(0.0, 1.0)
            if abs(rot - target_rot) > self.open_door_thresh:
                break
        return {
            "x": x,
            "y": y,
            "theta": theta,
            "mass": mass,
            "friction": friction,
            "rot": rot,
            "target_rot": target_rot,
            "open": 0.0,  # always start out closed
        }

    @classmethod
    def door_to_rooms(cls, door: Object, state: State) -> Set[Object]:
        """Public for use by oracle options."""
        assert isinstance(state, StateWithCache)
        door_to_rooms_cache = state.cache["door_to_rooms"]
        if door not in door_to_rooms_cache:
            rooms = set()
            door_geom = cls.object_to_geom(door, state)
            for room in state.get_objects(cls._room_type):
                room_geom = cls.object_to_geom(room, state)
                if door_geom.intersects(room_geom):
                    rooms.add(room)
            assert len(rooms) == 2
            door_to_rooms_cache[door] = rooms
        return door_to_rooms_cache[door]

    def _room_to_doors(self, room: Object, state: State) -> Set[Object]:
        assert isinstance(state, StateWithCache)
        room_to_doors_cache = state.cache["room_to_doors"]
        if room not in room_to_doors_cache:
            doors = set()
            room_geom = self.object_to_geom(room, state)
            for door in state.get_objects(self._door_type):
                door_geom = self.object_to_geom(door, state)
                if room_geom.intersects(door_geom):
                    doors.add(door)
            assert 1 <= len(doors) <= 4
            room_to_doors_cache[room] = doors
        return room_to_doors_cache[room]

    @classmethod
    def door_to_doorway_geom(cls, door: Object, state: State) -> Rectangle:
        """Public for use by oracle options."""
        assert isinstance(state, StateWithCache)
        doorway_geom_cache = state.cache["door_to_doorway_geom"]
        if door not in doorway_geom_cache:
            x = state.get(door, "x")
            y = state.get(door, "y")
            theta = state.get(door, "theta")
            doorway_size = cls.robot_radius + cls.doorway_pad
            # Top or bottom door.
            if abs(theta) < 1e-6:
                return Rectangle(x=x,
                                 y=(y - doorway_size),
                                 width=cls.hallway_width,
                                 height=(cls.wall_depth + 2 * doorway_size),
                                 theta=0)
            # Left or right door.
            assert abs(theta - np.pi / 2) < 1e-6
            geom = Rectangle(x=(x - cls.wall_depth - doorway_size),
                             y=y,
                             width=(cls.wall_depth + 2 * doorway_size),
                             height=cls.hallway_width,
                             theta=0)
            doorway_geom_cache[door] = geom
        return doorway_geom_cache[door]

    def _sample_room_map(self, rng: np.random.Generator) -> NDArray:
        # Sample a grid where any room can be reached from any other room.
        # To do this, perform a random tree search in the grid for a certain
        # number of steps, starting from a random location.
        assert self._room_map_size > 1
        room_map = np.zeros((self._room_map_size, self._room_map_size),
                            dtype=bool)
        min_num_rooms = max(2, int(self._min_room_exists_frac * room_map.size))
        max_num_rooms = int(self._max_room_exists_frac * room_map.size)
        num_rooms = rng.integers(min_num_rooms, max_num_rooms + 1)

        def _get_neighbors(room: Tuple[int, int]) -> Iterator[Tuple[int, int]]:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            r, c = room
            for dr, dc in deltas:
                nr, nc = r + dr, c + dc
                if 0 <= nr < room_map.shape[0] and 0 <= nc < room_map.shape[1]:
                    yield (nr, nc)

        start_r, start_c = rng.integers(self._room_map_size, size=2)
        start_room = (start_r, start_c)
        queue = [start_room]
        visited = {start_room}
        room_map[start_room] = 1
        while room_map.sum() < num_rooms:
            queue_idx = rng.integers(len(queue))
            room = queue.pop(queue_idx)
            for neighbor in _get_neighbors(room):
                if neighbor not in visited:
                    room_map[neighbor] = 1
                    visited.add(neighbor)
                    queue.append(neighbor)
        return room_map

    @staticmethod
    def _get_open_door_target_value(mass: float, friction: float,
                                    target_rot: float) -> float:
        # A made up complicated function.
        return np.tanh(target_rot) * (np.sin(mass) +
                                      np.cos(friction) * np.sqrt(mass))


class DoorKnobsEnv(DoorsEnv):
    """A 2D navigation environment with obstacles, rooms, and doors."""
    # Types
    _door_type = Type("door", ["x", "y", "theta", "mass", "friction", "open"])
    _knob_type = Type("knob", ["x", "y", "theta", "rot", "target_rot", "open"])
    open_door_thresh: ClassVar[float] = 0.1

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._door_to_knob: Dict[Object, Object] = {}
        self._open_door_target_value = CFG.doorknobs_target_value

    @classmethod
    def get_name(cls) -> str:
        return "doorknobs"

    def simulate(self, state: State, action: Action) -> State:
        new_door_rot = action.arr[2]
        next_state = self.simulate_moving(state, action)

        # If touching a door, change its value based on the action.
        for door in state.get_objects(self._door_type):

            if self._TouchingDoor_holds(state, [self._robot, door]):
                # Get corresponding doorknob and rotate it accordingly
                doorknob = self._door_to_knob[door]
                new_door_level = np.clip(
                    state.get(doorknob, "rot") + new_door_rot, 0.0, 1.0)
                next_state.set(doorknob, "rot", new_door_level)
                # Check if we should now open the door.
                target = self._open_door_target_value
                if abs(new_door_level - target) < self.open_door_thresh:
                    next_state.set(door, "open", 1.0)
                    next_state.set(doorknob, "open", 1.0)
                else:
                    next_state.set(door, "open", 0.0)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        self._room_map_size = CFG.doors_room_map_size
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        self._room_map_size = CFG.test_doors_room_map_size
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._InRoom,
            self._InDoorway,
            self._DoorInRoom,
            self._InMainRoom,
            self._DoorsShareRoom,
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InRoom}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._door_type, self._knob_type,
            self._room_type, self._obstacle_type
        }

    @classmethod
    def state_has_collision(cls, state: State, robot_pos: \
                             Optional[Array] = None) -> bool:
        """Public for use by oracle options."""
        robot, = state.get_objects(cls._robot_type)
        robot_geom = cls.object_to_geom(robot, state, robot_pos)
        # Check for collisions with obstacles.
        for obstacle in state.get_objects(cls._obstacle_type):
            obstacle_geom = cls.object_to_geom(obstacle, state)
            if robot_geom.intersects(obstacle_geom):
                return True
        # Check for collisions with closed doors.
        for door in state.get_objects(cls._door_type):
            if cls._DoorIsOpen_holds(state, [door]):
                continue
            door_geom = cls.object_to_geom(door, state)
            if robot_geom.intersects(door_geom):
                return True
        return False

    @property
    def action_space(self) -> Box:
        # dx, dy, drot
        lb = np.array(
            [-self.action_magnitude, -self.action_magnitude, -np.inf],
            dtype=np.float32)
        ub = np.array([self.action_magnitude, self.action_magnitude, np.inf],
                      dtype=np.float32)
        return Box(lb, ub)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        del caption  # unused
        x_lb, x_ub, y_lb, y_ub = self._get_world_boundaries(state)
        fig, ax = plt.subplots(1, 1, figsize=(x_ub - x_lb, y_ub - y_lb))

        # Draw rooms.
        default_room_color = "lightgray"
        in_room_color = "lightsteelblue"
        goal_room_color = "khaki"
        goal_room = next(iter(task.goal)).objects[1]
        for room in state.get_objects(self._room_type):
            room_geom = self.object_to_geom(room, state)
            if room == goal_room:
                color = goal_room_color
            elif self._InRoom_holds(state, [self._robot, room]):
                color = in_room_color
            else:
                color = default_room_color
            room_geom.plot(ax, color=color)

        # Draw robot.
        robot_color = "blue"
        robot_geom = self.object_to_geom(self._robot, state)
        robot_geom.plot(ax, color=robot_color)

        # Draw obstacles (including room walls).
        obstacle_color = "black"
        for obstacle in state.get_objects(self._obstacle_type):
            obstacle_geom = self.object_to_geom(obstacle, state)
            obstacle_geom.plot(ax, color=obstacle_color)

        # Draw doors.
        closed_door_color = "orangered"
        open_door_color = "lightgreen"
        doorway_color = "darkviolet"
        for door in state.get_objects(self._door_type):
            if self._DoorIsOpen_holds(state, [door]):
                color = open_door_color
            else:
                color = closed_door_color
            door_geom = self.object_to_geom(door, state)
            door_geom.plot(ax, color=color)
            if CFG.doors_draw_debug:
                doorway_geom = self.door_to_doorway_geom(door, state)
                doorway_geom.plot(ax, color=doorway_color, alpha=0.1)

        # Visualize the motion plan.
        if CFG.doors_draw_debug:
            if action is not None and action.has_option():
                option = action.get_option()
                if "position_plan" in option.memory:
                    xs, ys = zip(*option.memory["position_plan"])
                    plt.scatter(xs, ys, s=3, alpha=0.5, color=robot_color)

        pad = 2 * self.wall_depth
        ax.set_xlim(x_lb - pad, x_ub + pad)
        ax.set_ylim(y_lb - pad, y_ub + pad)

        plt.axis("off")
        plt.tight_layout()
        return fig

    @classmethod
    def object_to_geom(cls,
                       obj: Object,
                       state: State,
                       robot_pos: Optional[Array] = None) -> _Geom2D:
        """Public for use by oracle options."""
        if robot_pos is None:
            x = state.get(obj, "x")
            y = state.get(obj, "y")
        else:
            x, y = robot_pos
        if obj.is_instance(cls._robot_type):
            return utils.Circle(x, y, cls.robot_radius)
        # Only the robot shape is dynamic. All other shapes are cached.
        assert isinstance(state, StateWithCache)
        static_geom_cache = state.cache["static_geom"]
        if obj not in static_geom_cache:
            if obj.is_instance(cls._room_type):
                width = cls.room_size
                height = cls.room_size
                theta = 0.0
            elif obj.is_instance(cls._door_type):
                width = cls.hallway_width
                height = cls.wall_depth
                theta = state.get(obj, "theta")
            elif obj.is_instance(cls._knob_type):
                width = cls.hallway_width  # pragma: no cover
                height = cls.wall_depth  # pragma: no cover
                theta = state.get(obj, "theta")  # pragma: no cover
            else:
                assert obj.is_instance(cls._obstacle_type)
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "theta")
            geom = Rectangle(x=x, y=y, width=width, height=height, theta=theta)
            static_geom_cache[obj] = geom
        return static_geom_cache[obj]

    def _sample_door_feats(self, room_x: float, room_y: float, loc: str,
                           rng: np.random.Generator) -> Dict[str, float]:
        # This is the length of the wall on one side of the door.
        offset = (self.room_size + self.wall_depth - self.hallway_width) / 2

        if loc == "bottom":
            x = room_x + offset
            y = room_y - self.wall_depth / 2
            theta = 0.0
        else:
            assert loc == "left"
            x = room_x + self.wall_depth / 2
            y = room_y + offset
            theta = np.pi / 2

        mass, friction, _ = rng.uniform(0.0, 1.0, size=(3, ))
        return {
            "x": x,
            "y": y,
            "theta": theta,
            "mass": mass,
            "friction": friction,
            "open": 0.0,  # always start out closed
        }

    def _sample_doorknob_feats(self, room_x: float, room_y: float, loc: str,
                               _: np.random.Generator) -> Dict[str, float]:
        # This is the length of the wall on one side of the door.
        offset = (self.room_size + self.wall_depth - self.hallway_width) / 2

        if loc == "bottom":
            x = room_x + offset
            y = room_y - self.wall_depth / 2
            theta = 0.0
        else:
            assert loc == "left"
            x = room_x + self.wall_depth / 2
            y = room_y + offset
            theta = np.pi / 2

        target_rot = 0.75
        # Sample the initial rotation so that the door is not yet opened.
        return {
            "x": x,
            "y": y,
            "theta": theta,
            "rot": 0,
            "target_rot": target_rot,
            "open": 0.0,  # always start out closed
        }
