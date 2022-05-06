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
from predicators.src.utils import Rectangle, _Geom2D


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

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._robot_type = Type("robot", ["x", "y"])
        self._door_type = Type(
            "door", ["x", "y", "theta", "current", "target", "open"])
        self._room_type = Type("room", ["x", "y"])
        self._obstacle_type = Type("obstacle",
                                   ["x", "y", "width", "height", "theta"])
        # Predicates
        self._InRoom = Predicate("InRoom", [self._robot_type, self._room_type],
                                 self._InRoom_holds)
        self._InDoorway = Predicate("InDoorway",
                                    [self._robot_type, self._door_type],
                                    self._InDoorway_holds)
        self._TouchingDoor = Predicate("TouchingDoor",
                                       [self._robot_type, self._door_type],
                                       self._TouchingDoor_holds)
        self._DoorIsOpen = Predicate("DoorIsOpen", [self._door_type],
                                     self._DoorIsOpen_holds)
        self._DoorInRoom = Predicate("DoorInRoom",
                                     [self._door_type, self._room_type],
                                     self._DoorInRoom_holds)
        # Options
        self._Move = ParameterizedOption(
            "Move",
            # The first door is the one that should already be open, and the
            # second door is the next one that the robot will move to. After
            # the robot moves through the first door, it should be in the
            # room for the second door; this also means that the two doors
            # should share a room.
            types=[self._robot_type, self._door_type, self._door_type],
            # No parameters; the option always moves to the center of the
            # doorway for the second door.
            params_space=Box(0, 1, (0, )),
            # The policy is a motion planner.
            policy=self._Move_policy,
            initiable=self._Move_initiable,
            terminal=self._Move_terminal)
        self._OpenDoor = ParameterizedOption(
            "OpenDoor",
            types=[self._robot_type, self._door_type],
            # No parameters, since the right rotation is a deterministic
            # function of the door state.
            params_space=Box(0, 1, (0, )),
            policy=self._OpenDoor_policy,
            initiable=self._OpenDoor_initiable,
            terminal=self._OpenDoor_terminal)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "doors"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, new_door_val = action.arr
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
        # If touching a door, change its value based on the action
        for door in state.get_objects(self._door_type):
            if self._TouchingDoor_holds(state, [self._robot, door]):
                next_state.set(door, "current", new_door_val)
                # Check if we should now open the door.
                target = state.get(door, "target")
                if abs(new_door_val - target) < self.open_door_thresh:
                    next_state.set(door, "open", 1.0)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._InRoom, self._InDoorway, self._TouchingDoor,
            self._DoorIsOpen, self._DoorInRoom
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
    def options(self) -> Set[ParameterizedOption]:
        return {self._Move, self._OpenDoor}

    @property
    def action_space(self) -> Box:
        # dx, dy, drot
        lb = np.array([-self.action_magnitude, -self.action_magnitude, -np.pi],
                      dtype=np.float32)
        ub = np.array([self.action_magnitude, self.action_magnitude, np.pi],
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
            if CFG.doors_draw_debug:
                doorway_geom = self._door_to_doorway_geom(door, state)
                doorway_geom.plot(ax, color=doorway_color, alpha=0.1)

        # Visualize the motion plan.
        if CFG.doors_draw_debug:
            if action is not None and action.has_option():
                option = action.get_option()
                if "position_plan" in option.memory:
                    xs, ys = zip(*option.memory["position_plan"])
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
                feat_dict = self._sample_door_feats(room_x, room_y, name, rng)
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
            # Create the state with a temporary robot x and y that we will
            # immediately override below. This is done so we can use the
            # _get_position_in_doorway() helper function.
            state_dict[self._robot] = {"x": x, "y": y}
            state = utils.create_state_from_dict(state_dict)
            # Sample an initial and target room.
            start_idx, goal_idx = rng.choice(len(rooms), size=2, replace=False)
            start_room, goal_room = rooms[start_idx], rooms[goal_idx]
            # Sample an initial door in the start room.
            # TODO: sample a non-stupid initial door.
            door_candidates = sorted(self._room_to_doors(start_room, state))
            assert len(door_candidates) > 0
            start_door = door_candidates[rng.choice(len(door_candidates))]
            # Always start out in a doorway, so that all problems just require
            # moving between doorways.
            x, y = self._get_position_in_doorway(start_room, start_door, state)
            state.set(self._robot, "x", x)
            state.set(self._robot, "y", y)
            assert self._TouchingDoor_holds(state, [self._robot, start_door])
            # By construction, the state should be collision free.
            assert not self._state_has_collision(state)
            # Set the goal.
            goal_atom = GroundAtom(self._InRoom, [self._robot, goal_room])
            goal = {goal_atom}
            assert not goal_atom.holds(state)
            tasks.append(Task(state, goal))
        return tasks

    def _Move_initiable(self, state: State, memory: Dict,
                        objects: Sequence[Object], params: Array) -> bool:
        del params  # unused
        robot, start_door, end_door = objects
        if start_door == end_door:
            return False
        # The start door must be open.
        if not self._DoorIsOpen_holds(state, [start_door]):
            return False
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
        uncommon_room = next(iter(start_rooms - common_rooms))
        if not self._InRoom_holds(state, [robot, uncommon_room]):
            return False
        # The option is initiable, so we're going to make a plan now and store
        # it in memory for use in the policy. Note that policies are assumed
        # to be deterministic, but RRT is stochastic. We enforce determinism
        # by using a constant seed in RRT.
        rng = np.random.default_rng(CFG.seed)

        start_room_rect = self._object_to_geom(uncommon_room, state)
        end_room_rect = self._object_to_geom(common_room, state)
        room_rects = [start_room_rect, end_room_rect]

        def _sample_fn(_: Array) -> Array:
            # Only sample positions that are inside the two rooms that the
            # robot should stay in for this option.
            room_rect = room_rects[rng.choice(len(room_rects))]
            assert isinstance(room_rect, Rectangle)
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
        target_x, target_y = self._get_position_in_doorway(
            common_room, end_door, state)
        initial_state = np.array([robot_x, robot_y])
        target_state = np.array([target_x, target_y])
        position_plan = birrt.query(initial_state, target_state)
        memory["position_plan"] = position_plan
        assert position_plan is not None
        # Convert the plan from position space to action space.
        deltas = np.subtract(position_plan[1:], position_plan[:-1])
        action_plan = [
            Action(np.array([dx, dy, 0.0], dtype=np.float32))
            for (dx, dy) in deltas
        ]
        memory["action_plan"] = action_plan
        return True

    def _Move_terminal(self, state: State, memory: Dict,
                       objects: Sequence[Object], params: Array) -> bool:
        del params  # unused
        robot, _, end_door = objects
        desired_x, desired_y = memory["position_plan"][-1]
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        sq_dist = (robot_x - desired_x)**2 + (robot_y - desired_y)**2
        return sq_dist < self.move_sq_dist_tol

    def _Move_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        del state, objects, params  # unused
        assert memory["action_plan"], "Motion plan did not reach its goal"
        return memory["action_plan"].pop(0)

    def _OpenDoor_initiable(self, state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        # Can only open the door if touching it.
        return self._TouchingDoor_holds(state, objects)

    def _OpenDoor_terminal(self, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        # Terminate when the door is open.
        _, door = objects
        return self._DoorIsOpen_holds(state, [door])

    def _OpenDoor_policy(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> Action:
        del memory, params  # unused
        # TODO make this more complicated.
        _, door = objects
        target = state.get(door, "target")
        assert -np.pi <= target <= np.pi
        return Action(np.array([0.0, 0.0, target], dtype=np.float32))

    def _InRoom_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # The robot is in the room if its center is in the room.
        robot, room = objects
        robot_geom = self._object_to_geom(robot, state)
        assert isinstance(robot_geom, utils.Circle)
        room_geom = self._object_to_geom(room, state)
        return room_geom.contains_point(robot_geom.x, robot_geom.y)

    def _InDoorway_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        robot, door = objects
        doorway_geom = self._door_to_doorway_geom(door, state)
        robot_geom = self._object_to_geom(robot, state)
        return robot_geom.intersects(doorway_geom)

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
        return self._InDoorway_holds(state, objects)

    def _DoorIsOpen_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        door, = objects
        return state.get(door, "open") > 0.5

    def _DoorInRoom_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        door, room = objects
        return door in self._room_to_doors(room, state)

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
        return Rectangle(x=x, y=y, width=width, height=height, theta=theta)

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

        current = rng.uniform(-np.pi, np.pi)
        while True:
            target = rng.uniform(-np.pi, np.pi)
            if abs(current - target) > self.open_door_thresh:
                break

        return {
            "x": x,
            "y": y,
            "theta": theta,
            "current": current,
            "target": target,
            "open": 0.0
        }

    def _door_to_rooms(self, door: Object, state: State) -> Set[Object]:
        rooms = set()
        door_geom = self._object_to_geom(door, state)
        for room in state.get_objects(self._room_type):
            room_geom = self._object_to_geom(room, state)
            if door_geom.intersects(room_geom):
                rooms.add(room)
        assert len(rooms) == 2
        return rooms

    def _room_to_doors(self, room: Object, state: State) -> Set[Object]:
        doors = set()
        room_geom = self._object_to_geom(room, state)
        for door in state.get_objects(self._door_type):
            door_geom = self._object_to_geom(door, state)
            if room_geom.intersects(door_geom):
                doors.add(door)
        assert 1 <= len(doors) <= 4
        return doors

    def _door_to_doorway_geom(self, door: Object, state: State) -> Rectangle:
        x = state.get(door, "x")
        y = state.get(door, "y")
        theta = state.get(door, "theta")
        doorway_size = self.robot_radius + self.doorway_pad
        # Top or bottom door.
        if abs(theta) < 1e-6:
            return Rectangle(x=x,
                             y=(y - doorway_size),
                             width=self.hallway_width,
                             height=(self.wall_depth + 2 * doorway_size),
                             theta=0)
        # Left or right door.
        assert abs(theta - np.pi / 2) < 1e-6
        return Rectangle(x=(x - self.wall_depth - doorway_size),
                         y=y,
                         width=(self.wall_depth + 2 * doorway_size),
                         height=self.hallway_width,
                         theta=0)

    def _get_position_in_doorway(self, room: Object, door: Object,
                                 state: State) -> Tuple[float, float]:
        # Find the two vertices of the doorway that are in the room.
        doorway_geom = self._door_to_doorway_geom(door, state)
        room_geom = self._object_to_geom(room, state)
        vertices_in_room = []
        for (x, y) in doorway_geom.vertices:
            if room_geom.contains_point(x, y):
                vertices_in_room.append((x, y))
        assert len(vertices_in_room) == 2
        (x0, y0), (x1, y1) = vertices_in_room
        target_x = (x0 + x1) / 2
        target_y = (y0 + y1) / 2
        return (target_x, target_y)
