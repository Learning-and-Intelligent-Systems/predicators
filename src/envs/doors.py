"""A 2D navigation environment with obstacles, rooms, and doors."""

from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Sequence, \
    Set, Tuple

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
    obstacle_initial_position_radius: ClassVar[float] = 0.1
    obstacle_size_lb: ClassVar[float] = 0.05
    obstacle_size_ub: ClassVar[float] = 0.2
    doorway_size: ClassVar[float] = 0.025
    # This can be very small because we are not learning the move option.
    move_sq_dist_tol: ClassVar[float] = 1e-5

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
        self._TouchingDoor = Predicate("TouchingDoor",
                                       [self._robot_type, self._door_type],
                                       self._TouchingDoor_holds)
        self._DoorIsOpen = Predicate("DoorIsOpen", [self._door_type],
                                     self._DoorIsOpen_holds)
        # TODO others
        # Options
        self._Move = ParameterizedOption(
            "Move",
            # The first door is the one that should already be open, and the
            # second door is the next one that the robot will move to. After
            # the robot moves through the first door, it should be in the
            # room for the second door; this also means that the two doors
            # should share a room.
            types=[self._robot_type, self._door_type, self._door_type],
            # The parameter represents a relative position on the second door.
            params_space=Box(0.0, 1.0, (1, ), dtype=np.float32),
            # The policy is a motion planner.
            policy=self._Move_policy,
            initiable=self._Move_initiable,
            terminal=self._Move_terminal)
        # TODO others
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
        return {self._Move}  # TODO add other option

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

        # Draw obstacles (including room walls).
        obstacle_color = "black"
        for obstacle in state.get_objects(self._obstacle_type):
            obstacle_geom = self._object_to_geom(obstacle, state)
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
            door_geom = self._object_to_geom(door, state)
            door_geom.plot(ax, color=color)
            # Uncomment to also draw the doorway for the door.
            doorway_geom = self._door_to_doorway_geom(door, state)
            doorway_geom.plot(ax, color=doorway_color, alpha=0.1)

        # Uncomment for debugging motion planning.
        if action is not None and action.has_option():
            option = action.get_option()
            if "plan" in option.memory:
                xs, ys = zip(*option.memory["plan"])
                plt.scatter(xs, ys, s=3, alpha=0.5, color=robot_color)

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
        # Create the common parts of the initial state.
        common_state_dict = {}
        # TODO randomize this.
        # room_map = np.array([
        #     [1, 0, 0, 0, 1],
        #     [1, 0, 1, 1, 1],
        #     [1, 1, 1, 0, 1],
        #     [0, 1, 1, 1, 1],
        #     [1, 1, 1, 0, 1],
        # ])
        room_map = np.array([
            [1, 1],
            [1, 1],
        ])
        num_rows, num_cols = room_map.shape
        rooms = []
        for (r, c) in np.argwhere(room_map):
            room = Object(f"room{r}-{c}", self._room_type)
            rooms.append(room)
            room_x = float(c * self.room_size)
            room_y = float((num_rows - 1 - r) * self.room_size)
            common_state_dict[room] = {
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
                common_state_dict[wall] = {
                    "x": rect.x,
                    "y": rect.y,
                    "height": rect.height,
                    "width": rect.width,
                    "theta": rect.theta,
                }
            # Create doors for this room. Note that we only need to create
            # bottom or left, because each door is on the bottom or left of
            # some room (and top / right of another room).
            for name, exists in [("bottom", hall_bottom), ("left", hall_left)]:
                if not exists:
                    continue
                door = Object(f"{name}-door{r}-{c}", self._door_type)
                feat_dict = self._get_door_feats(room_x, room_y, name)
                common_state_dict[door] = feat_dict
        while len(tasks) < num:
            state_dict = {k: v.copy() for k, v in common_state_dict.items()}
            # Sample obstacles for each room. Choose between 0 and 3, and
            # make them small and centered enough that the robot should always
            # be able to find a collision-free path through the room.
            for room in rooms:
                room_x = state_dict[room]["x"]
                room_y = state_dict[room]["y"]
                room_cx = room_x + self.room_size / 2
                room_cy = room_y + self.room_size / 2
                rad = self.obstacle_initial_position_radius
                num_obstacles = rng.choice(4)
                obstacle_rects_for_room: List[utils.Rectangle] = []
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
                        rect = utils.Rectangle(x=x,
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
            # Sample an initial and target room.
            start_idx, goal_idx = rng.choice(len(rooms), size=2, replace=False)
            start_room, goal_room = rooms[start_idx], rooms[goal_idx]
            # Always start out near the center of the room to avoid issues with
            # rotating in corners.
            room_x = state_dict[start_room]["x"]
            room_y = state_dict[start_room]["y"]
            room_cx = room_x + self.room_size / 2
            room_cy = room_y + self.room_size / 2
            rad = self.robot_initial_position_radius
            x = rng.uniform(room_cx - rad, room_cx + rad)
            y = rng.uniform(room_cy - rad, room_cy + rad)
            state_dict[self._robot] = {"x": x, "y": y}
            # Create the state.
            state = utils.create_state_from_dict(state_dict)
            # Make sure the state is collision-free.
            if self._state_has_collision(state):
                continue
            # Set the goal.
            goal_atom = GroundAtom(self._InRoom, [self._robot, goal_room])
            goal = {goal_atom}
            assert not goal_atom.holds(state)
            tasks.append(Task(state, goal))
        return tasks

    def _Move_initiable(self, state: State, memory: Dict,
                        objects: Sequence[Object], params: Array) -> bool:
        robot, start_door, end_door = objects
        if start_door == end_door:
            return False
        # TODO: add check that the start door is open.
        # The doors should share a room, but the robot should not already
        # be in that room.
        start_rooms = self._door_to_rooms(start_door, state)
        end_rooms = self._door_to_rooms(end_door, state)
        common_rooms = start_rooms & end_rooms
        if not common_rooms:
            return False
        assert len(common_rooms) == 1
        common_room = next(iter(common_rooms))
        if self._InRoom_holds(state, [robot, common_room]):
            return False
        # The robot should be in the other room.
        assert len(start_rooms) == 2
        noncommon_room = next(iter(start_rooms - common_rooms))
        if not self._InRoom_holds(state, [robot, noncommon_room]):
            return False
        # The option is initiable, so we're going to make a plan now and store
        # it in memory for use in the policy. Note that policies are assumed
        # to be deterministic, but RRT is stochastic. We enforce determinism
        # by using a constant seed in RRT.
        rng = np.random.default_rng(CFG.seed)

        start_room_rect = self._object_to_geom(noncommon_room, state)
        end_room_rect = self._object_to_geom(common_room, state)
        room_rects = [start_room_rect, end_room_rect]

        def _sample_fn(_: Array) -> Array:
            # Only sample positions that are inside the two rooms that the
            # robot should stay in for this option.
            room_rect = room_rects[rng.choice(len(room_rects))]
            assert isinstance(room_rect, utils.Rectangle)
            # Sample a point in this room that is far enough away from the
            # wall (to save on collision checking).
            x_lb = room_rect.x + self.robot_radius
            x_ub = room_rect.x + self.room_size - self.robot_radius
            y_lb = room_rect.y + self.robot_radius
            y_ub = room_rect.y + self.room_size - self.robot_radius
            x = rng.uniform(x_lb, x_ub)
            y = rng.uniform(y_lb, y_ub)
            return np.array([x, y], dtype=np.float32)

        def _extend_fn(pt1: Array, pt2: Array) -> Iterator[Array]:
            # Make sure that we obey the bounds on actions.
            distance = np.linalg.norm(pt2 - pt1)
            num = int(distance / self.action_magnitude) + 1
            if num == 0:
                yield pt2
            for i in range(1, num + 1):
                yield pt1 * (1 - i / num) + pt2 * i / num

        def _collision_fn(pt: Array) -> bool:
            # Make a hypothetical state for the robot at this point and check
            # if there would be collisions.
            x, y = pt
            s = state.copy()
            s.set(robot, "x", x)
            s.set(robot, "y", y)
            return self._state_has_collision(s)

        def _distance_fn(from_pt: Array, to_pt: Array) -> float:
            return np.sum(np.subtract(from_pt, to_pt)**2)

        birrt = utils.BiRRT(_sample_fn,
                            _extend_fn,
                            _collision_fn,
                            _distance_fn,
                            rng,
                            num_attempts=CFG.doors_birrt_num_attempts,
                            num_iters=CFG.doors_birrt_num_iters,
                            smooth_amt=CFG.doors_birrt_smooth_amt)

        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        initial_state = np.array([robot_x, robot_y])
        target_state = self._move_param_to_target_position(
            params, end_door, state)
        position_plan = birrt.query(initial_state, target_state)
        memory["plan"] = position_plan  # for debugging only
        assert position_plan is not None
        # Convert the plan from position space to action space.
        deltas = np.subtract(position_plan[1:], position_plan[:-1])
        action_plan = [
            Action(np.array([dx, dy, 0.0, 0.0], dtype=np.float32))
            for (dx, dy) in deltas
        ]
        memory["action_plan"] = action_plan
        return True

    def _Move_terminal(self, state: State, memory: Dict,
                       objects: Sequence[Object], params: Array) -> bool:
        del memory  # unused
        robot, _, end_door = objects
        desired_x, desired_y = self._move_param_to_target_position(
            params, end_door, state)
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        sq_dist = (robot_x - desired_x)**2 + (robot_y - desired_y)**2
        return sq_dist < self.move_sq_dist_tol

    def _Move_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        assert memory["action_plan"], "Motion plan did not reach its goal"
        return memory["action_plan"].pop(0)

    def _InRoom_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # The robot is in the room if its center is in the room.
        robot, room = objects
        robot_geom = self._object_to_geom(robot, state)
        assert isinstance(robot_geom, utils.Circle)
        room_geom = self._object_to_geom(room, state)
        return room_geom.contains_point(robot_geom.x, robot_geom.y)

    def _TouchingDoor_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        robot, door = objects
        # Once the door is open, the robot is no longer touching it.
        if self._DoorIsOpen_holds(state, [door]):
            return False
        # The robot is considered to be touching the door if it's in the
        # doorway for that door. Note that we don't want to check if the
        # robot is literally touching the door, because collision checking
        # will forbid that from ever happening.
        doorway_geom = self._door_to_doorway_geom(door, state)
        robot_geom = self._object_to_geom(robot, state)
        return robot_geom.intersects(doorway_geom)

    def _DoorIsOpen_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        door, = objects
        return state.get(door, "open") > 0.5

    def _state_has_collision(self, state: State) -> bool:
        robot, = state.get_objects(self._robot_type)
        robot_geom = self._object_to_geom(robot, state)
        # Check for collisions with obstacles.
        for obstacle in state.get_objects(self._obstacle_type):
            obstacle_geom = self._object_to_geom(obstacle, state)
            if robot_geom.intersects(obstacle_geom):
                return True
        # Check for collisions with closed doors.
        for door in state.get_objects(self._door_type):
            if self._DoorIsOpen_holds(state, [door]):
                continue
            door_geom = self._object_to_geom(door, state)
            if robot_geom.intersects(door_geom):
                return True
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

    def _get_door_feats(self, room_x: float, room_y: float,
                        loc: str) -> Dict[str, float]:
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

        # TODO randomize
        target = 0.0

        # TODO close the doors
        return {"x": x, "y": y, "theta": theta, "target": target, "open": 1.0}

    def _door_to_rooms(self, door: Object, state: State) -> Set[Object]:
        rooms = set()
        door_geom = self._object_to_geom(door, state)
        for room in state.get_objects(self._room_type):
            room_geom = self._object_to_geom(room, state)
            if door_geom.intersects(room_geom):
                rooms.add(room)
        assert len(rooms) == 2
        return rooms

    def _door_to_doorway_geom(self, door: Object, state: State) -> _Geom2D:
        x = state.get(door, "x")
        y = state.get(door, "y")
        theta = state.get(door, "theta")
        # Top or bottom door.
        if abs(theta) < 1e-6:
            return utils.Rectangle(x=x,
                                   y=(y - self.doorway_size),
                                   width=self.hallway_width,
                                   height=(self.wall_depth +
                                           2 * self.doorway_size),
                                   theta=0)
        # Left or right door.
        assert abs(theta - np.pi / 2) < 1e-6
        return utils.Rectangle(x=(x - self.wall_depth - self.doorway_size),
                               y=y,
                               width=(self.wall_depth + 2 * self.doorway_size),
                               height=self.hallway_width,
                               theta=0)
