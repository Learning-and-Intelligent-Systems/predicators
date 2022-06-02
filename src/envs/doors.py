"""A 2D navigation environment with obstacles, rooms, and doors."""

import itertools
from typing import ClassVar, Dict, Iterator, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from numpy.typing import NDArray

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Object, \
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
        self._door_type = Type("door", [
            "x", "y", "theta", "mass", "friction", "rot", "target_rot", "open"
        ])
        self._room_type = Type("room", ["x", "y"])
        self._obstacle_type = Type("obstacle",
                                   ["x", "y", "width", "height", "theta"])
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
        # Options
        self._MoveToDoor = ParameterizedOption(
            "MoveToDoor",
            types=[self._robot_type, self._door_type],
            # No parameters; the option always moves to the doorway center.
            params_space=Box(0, 1, (0, )),
            # The policy is a motion planner.
            policy=self._MoveToDoor_policy,
            # Only initiable when the robot is in a room for the doory.
            initiable=self._MoveToDoor_initiable,
            terminal=self._MoveToDoor_terminal)
        self._OpenDoor = ParameterizedOption(
            "OpenDoor",
            types=[self._door_type, self._robot_type],
            # Even though this option does not need to be parameterized, we
            # make it so, because we want to match the parameter space of the
            # option that will get learned during option learning. This is
            # useful for when we want to use sampler_learner = "oracle" too.
            params_space=Box(-np.inf, np.inf, (2, )),
            policy=self._OpenDoor_policy,
            # Only initiable when the robot is in the doorway.
            initiable=self._OpenDoor_initiable,
            terminal=self._OpenDoor_terminal)
        self._MoveThroughDoor = ParameterizedOption(
            "MoveThroughDoor",
            types=[self._robot_type, self._door_type],
            # No parameters; the option always moves straight through.
            params_space=Box(0, 1, (0, )),
            # The policy just moves in a straight line. No motion planning
            # required, because there are no obstacles in the doorway.
            policy=self._MoveThroughDoor_policy,
            # Only initiable when the robot is in the doorway.
            initiable=self._MoveThroughDoor_initiable,
            terminal=self._MoveThroughDoor_terminal)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        # Hyperparameters from CFG.
        self._room_map_size = CFG.doors_room_map_size
        self._min_obstacles_per_room = CFG.doors_min_obstacles_per_room
        self._max_obstacles_per_room = CFG.doors_max_obstacles_per_room
        self._min_room_exists_frac = CFG.doors_min_room_exists_frac
        self._max_room_exists_frac = CFG.doors_max_room_exists_frac
        # Caches for values that do not ever change.
        self._static_geom_cache: Dict[Object, _Geom2D] = {}
        self._door_to_rooms_cache: Dict[Object, Set[Object]] = {}
        self._room_to_doors_cache: Dict[Object, Set[Object]] = {}
        self._door_to_doorway_geom_cache: Dict[Object, Rectangle] = {}
        self._position_in_doorway_cache: Dict[Tuple[Object, Object],
                                              Tuple[float, float]] = {}
        # See note in _sample_initial_state_from_map().
        self._task_id_count = itertools.count()

    @classmethod
    def get_name(cls) -> str:
        return "doors"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, new_door_rot = action.arr
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

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
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
    def options(self) -> Set[ParameterizedOption]:
        return {self._MoveToDoor, self._OpenDoor, self._MoveThroughDoor}

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
            task: Task,
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

        pad = 2 * self.wall_depth
        ax.set_xlim(x_lb - pad, x_ub + pad)
        ax.set_ylim(y_lb - pad, y_ub + pad)

        plt.axis("off")
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        tasks: List[Task] = []
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
            tasks.append(Task(state, goal))
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
            if not self._state_has_collision(state):
                return state

    def _MoveToDoor_initiable(self, state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
        del params  # unused
        robot, door = objects
        # The robot must be in one of the rooms for the door.
        for r in self._door_to_rooms(door, state):
            if self._InRoom_holds(state, [robot, r]):
                room = r
                break
        else:
            return False
        # Make a plan and store it in memory for use in the policy. Note that
        # policies are assumed to be deterministic, but RRT is stochastic. We
        # enforce determinism by using a constant seed in RRT.
        rng = np.random.default_rng(CFG.seed)

        room_rect = self._object_to_geom(room, state)

        def _sample_fn(_: Array) -> Array:
            # Sample a point in the room that is far enough away from the
            # wall (to save on collision checking).
            assert isinstance(room_rect, Rectangle)
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

        # Set up the initial and target inputs for the motion planner.
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        target_x, target_y = self._get_position_in_doorway(room, door, state)
        initial_state = np.array([robot_x, robot_y])
        target_state = np.array([target_x, target_y])
        # Run planning.
        position_plan = birrt.query(initial_state, target_state)
        # In very rare cases, motion planning fails (it is stochastic after
        # all). In this case, determine the option to be not initiable.
        if position_plan is None:  # pragma: no cover
            return False
        # The position plan is used for the termination check, and for debug
        # drawing in the rendering.
        memory["position_plan"] = position_plan
        # Convert the plan from position space to action space.
        deltas = np.subtract(position_plan[1:], position_plan[:-1])
        action_plan = [
            Action(np.array([dx, dy, 0.0], dtype=np.float32))
            for (dx, dy) in deltas
        ]
        memory["action_plan"] = action_plan
        return True

    def _MoveToDoor_terminal(self, state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        # Terminate as soon as we are in the doorway.
        robot, door = objects
        return self._InDoorway_holds(state, [robot, door])

    @staticmethod
    def _MoveToDoor_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
        del state, objects, params  # unused
        assert memory["action_plan"], "Motion plan did not reach its goal"
        return memory["action_plan"].pop(0)

    def _MoveThroughDoor_initiable(self, state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> bool:
        del params  # unused
        robot, door = objects
        # The robot must be in the doorway.
        if not self._InDoorway_holds(state, [robot, door]):
            return False
        # The door must be open.
        if not self._DoorIsOpen_holds(state, [door]):
            return False
        # The option is initiable. Memorize the target -- otherwise, we would
        # not know which side of the door to move toward during execution.
        room1, room2 = self._door_to_rooms(door, state)
        if self._InRoom_holds(state, [robot, room1]):
            end_room = room2
        else:
            assert self._InRoom_holds(state, [robot, room2])
            end_room = room1
        memory["target"] = self._get_position_in_doorway(end_room, door, state)
        memory["target_room"] = end_room
        return True

    def _MoveThroughDoor_terminal(self, state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> bool:
        del params  # unused
        robot, door = objects
        target_room = memory["target_room"]
        # Sanity check: we should never leave the doorway.
        assert self._InDoorway_holds(state, [robot, door])
        # Terminate as soon as we enter the other room.
        return self._InRoom_holds(state, [robot, target_room])

    def _MoveThroughDoor_policy(self, state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
        del params  # unused
        robot, _ = objects
        desired_x, desired_y = memory["target"]
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        delta = np.subtract([desired_x, desired_y], [robot_x, robot_y])
        delta_norm = np.linalg.norm(delta)
        if delta_norm > self.action_magnitude:
            delta = self.action_magnitude * delta / delta_norm
        dx, dy = delta
        action = Action(np.array([dx, dy, 0.0], dtype=np.float32))
        assert self.action_space.contains(action.arr)
        return action

    def _OpenDoor_initiable(self, state: State, memory: Dict,
                            objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        # Can only open the door if touching it.
        door, robot = objects
        return self._TouchingDoor_holds(state, [robot, door])

    def _OpenDoor_terminal(self, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> bool:
        del memory, params  # unused
        # Terminate when the door is open.
        door, _ = objects
        return self._DoorIsOpen_holds(state, [door])

    @staticmethod
    def _OpenDoor_policy(state: State, memory: Dict, objects: Sequence[Object],
                         params: Array) -> Action:
        del memory  # unused
        door, _ = objects
        delta_rot, _ = params
        current_rot = state.get(door, "rot")
        target = current_rot + delta_rot
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
        rooms1 = self._door_to_rooms(door1, state)
        rooms2 = self._door_to_rooms(door2, state)
        return len(rooms1 & rooms2) > 0

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
        # Only the robot shape is dynamic. All other shapes are cached.
        if obj not in self._static_geom_cache:
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
            geom = Rectangle(x=x, y=y, width=width, height=height, theta=theta)
            self._static_geom_cache[obj] = geom
        return self._static_geom_cache[obj]

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

    def _door_to_rooms(self, door: Object, state: State) -> Set[Object]:
        if door not in self._door_to_rooms_cache:
            rooms = set()
            door_geom = self._object_to_geom(door, state)
            for room in state.get_objects(self._room_type):
                room_geom = self._object_to_geom(room, state)
                if door_geom.intersects(room_geom):
                    rooms.add(room)
            assert len(rooms) == 2
            self._door_to_rooms_cache[door] = rooms
        return self._door_to_rooms_cache[door]

    def _room_to_doors(self, room: Object, state: State) -> Set[Object]:
        if room not in self._room_to_doors_cache:
            doors = set()
            room_geom = self._object_to_geom(room, state)
            for door in state.get_objects(self._door_type):
                door_geom = self._object_to_geom(door, state)
                if room_geom.intersects(door_geom):
                    doors.add(door)
            assert 1 <= len(doors) <= 4
            self._room_to_doors_cache[room] = doors
        return self._room_to_doors_cache[room]

    def _door_to_doorway_geom(self, door: Object, state: State) -> Rectangle:
        if door not in self._door_to_doorway_geom_cache:
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
            geom = Rectangle(x=(x - self.wall_depth - doorway_size),
                             y=y,
                             width=(self.wall_depth + 2 * doorway_size),
                             height=self.hallway_width,
                             theta=0)
            self._door_to_doorway_geom_cache[door] = geom
        return self._door_to_doorway_geom_cache[door]

    def _get_position_in_doorway(self, room: Object, door: Object,
                                 state: State) -> Tuple[float, float]:
        if (room, door) not in self._position_in_doorway_cache:
            # Find the two vertices of the doorway that are in the room.
            doorway_geom = self._door_to_doorway_geom(door, state)
            room_geom = self._object_to_geom(room, state)
            vertices_in_room = []
            for (x, y) in doorway_geom.vertices:
                if room_geom.contains_point(x, y):
                    vertices_in_room.append((x, y))
            assert len(vertices_in_room) == 2
            (x0, y0), (x1, y1) = vertices_in_room
            tx = (x0 + x1) / 2
            ty = (y0 + y1) / 2
            self._position_in_doorway_cache[(room, door)] = (tx, ty)
        return self._position_in_doorway_cache[(room, door)]

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
