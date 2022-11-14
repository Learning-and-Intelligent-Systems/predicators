"""Toy environment for testing refinement cost heuristic learning."""

from typing import ClassVar, Dict, Iterator, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type
from predicators.utils import _Geom2D


class NarrowPassageEnv(BaseEnv):
    """An environment where a 2D point mass robot must reach a static 2D point
    by passing through a narrow passage, or by opening a door and passing
    through a wider passageway.

    The action space is 3D, specifying (dx, dy, door).
    (dx, dy) defines a robot movement, where the magnitude of the movement
    in each direction is constrained by action_magnitude.
    door indicates a door-opening action. If door > 0, any attempted
    movement is ignored (i.e. treat dx and dy as 0) and the robot will open
    a closed door if nearby it.

    Based on the TouchPoint and Doors environments.
    """
    x_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 1.0
    y_lb: ClassVar[float] = 0.0
    y_ub: ClassVar[float] = 1.0

    robot_radius: ClassVar[float] = 0.1 / 2
    target_radius: ClassVar[float] = 0.1 / 2
    door_x_pos: ClassVar[float] = 0.2
    door_width_padding: ClassVar[float] = 0.075
    door_sensor_radius: ClassVar[float] = 0.2
    passage_x_pos: ClassVar[float] = 0.7
    passage_width_padding: ClassVar[float] = 0.02
    wall_thickness_half: ClassVar[float] = 0.15 / 2
    doorway_depth: ClassVar[float] = 1e-2
    init_pos_margin: ClassVar[float] = 1e-3

    action_magnitude: ClassVar[float] = 0.1

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Types
        self._robot_type = Type("robot", ["x", "y"])
        self._target_type = Type("target", ["x", "y"])
        self._wall_type = Type("wall", ["x", "width"])
        self._door_type = Type("door", ["x", "open"])
        # Type for the region within which the robot must be located
        # in order for the door-opening action to work.
        self._door_sensor_type = Type("door_sensor", ["x"])
        # Predicates
        self._TouchedGoal = Predicate("TouchedGoal",
                                      [self._robot_type, self._target_type],
                                      self._TouchedGoal_holds)
        self._DoorIsOpen = Predicate("DoorIsOpen", [self._door_type],
                                     self._DoorIsOpen_holds)
        self._DoorIsClosed = Predicate("DoorIsClosed", [self._door_type],
                                       self._DoorIsClosed_holds)
        # Options
        self._MoveToTarget = ParameterizedOption(
            "MoveToTarget",
            types=[self._robot_type, self._target_type],
            params_space=Box(0, 1, (1, )),
            policy=self._MoveToTarget_policy,
            initiable=self._MoveToTarget_initiable,
            terminal=self._MoveToTarget_terminal,
        )
        self._MoveAndOpenDoor = ParameterizedOption(
            "MoveAndOpenDoor",
            types=[self._robot_type, self._door_type],
            params_space=Box(0, 1, (1, )),
            policy=self._MoveAndOpenDoor_policy,
            initiable=self._MoveAndOpenDoor_initiable,
            terminal=self._MoveAndOpenDoor_terminal,
        )
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._target = Object("target", self._target_type)
        self._walls = [Object(f"wall{i}", self._wall_type) for i in range(3)]
        self._door = Object("door", self._door_type)
        self._door_sensor = Object("door_sensor", self._door_sensor_type)

        # Cache for _Geom2D objects
        self._static_geom_cache: Dict[Object, _Geom2D] = {}

    @classmethod
    def get_name(cls) -> str:
        return "narrow_passage"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, open_door = action.arr
        next_state = state.copy()
        if open_door > 0:
            # Open door if within the door sensor range
            if self._robot_near_door(state):
                next_state.set(self._door, "open", 1)
            return next_state
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        new_x = x + dx
        new_y = y + dy
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        # Check for collisions.
        if self._state_has_collision(next_state) or self._coords_out_of_bounds(
                new_x, new_y):
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
        return {self._TouchedGoal, self._DoorIsOpen, self._DoorIsClosed}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._TouchedGoal}

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._target_type, self._wall_type,
            self._door_type, self._door_sensor_type
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._MoveToTarget, self._MoveAndOpenDoor}

    @property
    def action_space(self) -> Box:
        # (dx, dy, door), where dx and dy are offsets
        # and door > 0 is an attempt to open a door (ignoring movement)
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
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        robot_color = "red"
        target_color = "green"
        wall_color = "blue"
        door_color = "purple"
        sensor_color = "pink"

        # Draw robot and target circles
        robot_geom = self._object_to_geom(self._robot, state)
        robot_geom.plot(ax, color=robot_color)
        target_geom = self._object_to_geom(self._target, state)
        target_geom.plot(ax, color=target_color)

        # Draw door
        door_open = state.get(self._door, "open")
        if not door_open:
            sensor_geom = self._object_to_geom(self._door_sensor, state)
            sensor_geom.plot(ax, color=sensor_color, fill=False)
            door_geom = self._object_to_geom(self._door, state)
            door_geom.plot(ax, color=door_color)
        # Draw walls
        for wall in self._walls:
            wall_geom = self._object_to_geom(wall, state)
            wall_geom.plot(ax, color=wall_color)

        ax.set_xlim(self.x_lb - self.robot_radius,
                    self.x_ub + self.robot_radius)
        ax.set_ylim(self.y_lb - self.robot_radius,
                    self.y_ub + self.robot_radius)
        title = f"{robot_color} = robot, {target_color} = target"
        if caption is not None:
            title += f";\n{caption}"
        plt.suptitle(title, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        # There is only one goal in this environment.
        goal_atom = GroundAtom(self._TouchedGoal, [self._robot, self._target])
        goal = {goal_atom}

        # The initial positions of the robot and target vary, while wall and
        # door positions are fixed. The robot should be above the walls, while
        # the dot should be below the walls (y coordinate)
        y_mid = (self.y_ub - self.y_lb) / 2 + self.y_lb
        margin = self.wall_thickness_half + self.init_pos_margin
        door_width = (self.robot_radius + self.door_width_padding) * 2
        passage_width = (self.robot_radius + self.passage_width_padding) * 2

        tasks: List[Task] = []
        while len(tasks) < num:
            state = utils.create_state_from_dict({
                self._robot: {
                    "x":
                    rng.uniform(self.x_lb, self.x_ub),
                    "y":
                    rng.uniform(y_mid + margin + self.robot_radius, self.y_ub),
                },
                self._target: {
                    "x":
                    rng.uniform(self.x_lb, self.x_ub),
                    "y":
                    rng.uniform(self.y_lb,
                                y_mid - margin - self.target_radius),
                },
                # Wall and door positions are fixed, defined by class variables
                self._walls[0]: {
                    "x": self.x_lb - self.robot_radius,
                    "width": self.door_x_pos + self.robot_radius,
                },
                self._walls[1]: {
                    "x": self.door_x_pos + door_width,
                    "width": self.passage_x_pos - self.door_x_pos - door_width,
                },
                self._walls[2]: {
                    "x":
                    self.passage_x_pos + passage_width,
                    "width":
                    self.x_ub - self.passage_x_pos - passage_width +
                    self.robot_radius,
                },
                self._door: {
                    "x": self.door_x_pos,
                    "open": 0,  # door starts closed
                },
                self._door_sensor: {
                    "x": self.door_x_pos + door_width / 2,
                },
            })
            # Make sure goal is not satisfied.
            assert not goal_atom.holds(
                state
            ), "Error: goal is already satisfied in this state initialization"
            tasks.append(Task(state, goal))
        return tasks

    @staticmethod
    def _MoveToTarget_policy(state: State, memory: Dict,
                             objects: Sequence[Object],
                             params: Array) -> Action:
        del state, objects, params  # unused
        assert memory["action_plan"], "Motion plan did not reach its goal"
        return memory["action_plan"].pop(0)

    def _MoveToTarget_initiable(self, state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> bool:
        robot, target = objects
        # Set up the target input for the motion planner.
        target_x = state.get(target, "x")
        target_y = state.get(target, "y")
        success = self._run_birrt(state, memory, params, robot,
                                  np.array([target_x, target_y]))
        return success

    def _MoveToTarget_terminal(self, state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> bool:
        del memory, params  # unused
        return self._TouchedGoal_holds(state, objects)

    @staticmethod
    def _MoveAndOpenDoor_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
        del state, objects, params  # unused
        assert memory["action_plan"], "Motion plan did not reach its goal"
        return memory["action_plan"].pop(0)

    def _MoveAndOpenDoor_initiable(self, state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> bool:
        robot, door = objects
        # If door is already open, this is not initiable
        if self._DoorIsOpen_holds(state, [door]):
            return False
        # If robot is already within range of the door, just open the door
        if self._robot_near_door(state):
            memory["action_plan"] = [
                Action(np.array([0.0, 0.0, 1.0], dtype=np.float32))
            ]
            return True
        # Select target point slightly above door
        door_center_x = state.get(door, "x")
        door_target_y = (
            self.y_ub - self.y_lb
        ) / 2 + self.y_lb + self.door_sensor_radius - self.robot_radius
        success = self._run_birrt(state, memory, params, robot,
                                  np.array([door_center_x, door_target_y]))
        if not success:
            # Failed to find motion plan, so option is not initiable
            return False
        # Append open door action to memory action plan
        memory["action_plan"].append(
            Action(np.array([0.0, 0.0, 1.0], dtype=np.float32)))
        return True

    def _MoveAndOpenDoor_terminal(self, state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> bool:
        del memory, params  # unused
        return self._DoorIsOpen_holds(state, objects[1:])

    def _run_birrt(self, state: State, memory: Dict, params: Array,
                   robot: Object, target_position: Array) -> bool:
        """Runs BiRRT to motion plan from start to target positions, and store
        the position and action plans in memory if successful.

        Returns true if successful, else false
        """
        # The seed is determined by the parameter passed into the option.
        # This is a hack for bilevel planning from giving up if motion planning
        # fails on the first attempt. We make the params array non-empty so it
        # is resampled, and this sets the BiRRT rng.
        rng = np.random.default_rng(int(params[0] * 1e4))

        def _sample_fn(_: Array) -> Array:
            # Sample a point in the environment
            x = rng.uniform(self.x_lb, self.x_ub)
            y = rng.uniform(self.y_lb, self.y_ub)
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
                            num_attempts=CFG.narrow_passage_birrt_num_attempts,
                            num_iters=CFG.narrow_passage_birrt_num_iters,
                            smooth_amt=CFG.narrow_passage_birrt_smooth_amt)
        # Run planning.
        robot_x = state.get(robot, "x")
        robot_y = state.get(robot, "y")
        start_position = np.array([robot_x, robot_y])
        position_plan = birrt.query(start_position, target_position)
        # If motion planning fails, determine the option to be not initiable.
        if position_plan is None:
            return False
        # The position plan is used for the termination check, and possibly
        # can be used for debug drawing in the rendering in the future.
        memory["position_plan"] = position_plan
        # Convert the plan from position space to action space.
        deltas = np.subtract(position_plan[1:], position_plan[:-1])
        action_plan = [
            Action(np.array([dx, dy, 0.0], dtype=np.float32))
            for (dx, dy) in deltas
        ]
        memory["action_plan"] = action_plan
        return True

    def _TouchedGoal_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robot, target = objects
        robot_geom = self._object_to_geom(robot, state)
        target_geom = self._object_to_geom(target, state)
        return robot_geom.intersects(target_geom)

    def _DoorIsOpen_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        door = objects[0]
        return state.get(door, "open") == 1

    def _DoorIsClosed_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        door = objects[0]
        return state.get(door, "open") == 0

    def _coords_out_of_bounds(self, new_x: float, new_y: float) -> bool:
        if (self.x_lb <= new_x <= self.x_ub) and (self.y_lb <= new_y <=
                                                  self.y_ub):
            return False
        return True

    def _state_has_collision(self, state: State) -> bool:
        robot, = state.get_objects(self._robot_type)
        robot_geom = self._object_to_geom(robot, state)
        # Check for collisions with obstacles.
        for obstacle in state.get_objects(self._wall_type):
            obstacle_geom = self._object_to_geom(obstacle, state)
            if robot_geom.intersects(obstacle_geom):
                return True
        # Check for collisions with closed doors.
        door, = state.get_objects(self._door_type)
        if self._DoorIsClosed_holds(state, [door]):
            door_geom = self._object_to_geom(door, state)
            if robot_geom.intersects(door_geom):
                return True
        return False

    def _robot_near_door(self, state: State) -> bool:
        """Returns true if the robot is within range of the door sensor."""
        robot, = state.get_objects(self._robot_type)
        robot_geom = self._object_to_geom(robot, state)
        # Check for "collision" with door sensor
        door_sensor, = state.get_objects(self._door_sensor_type)
        door_sensor_geom = self._object_to_geom(door_sensor, state)
        return robot_geom.intersects(door_sensor_geom)

    def _object_to_geom(self, obj: Object, state: State) -> _Geom2D:
        """Adapted from doors.py."""
        x = state.get(obj, "x")
        if (obj.is_instance(self._robot_type)
                or obj.is_instance(self._target_type)):
            y = state.get(obj, "y")
            return utils.Circle(x, y, self.robot_radius)
        # Cache static objects such as door and walls
        if obj not in self._static_geom_cache:
            if obj.is_instance(self._door_sensor_type):
                y = self.y_lb + (self.y_ub - self.y_lb) / 2
                self._static_geom_cache[obj] = utils.Circle(
                    x, y, self.door_sensor_radius)
            else:
                if obj.is_instance(self._wall_type):
                    y = self.y_lb + (self.y_ub -
                                     self.y_lb) / 2 - self.wall_thickness_half
                    width = state.get(obj, "width")
                    height = self.wall_thickness_half * 2
                else:
                    assert obj.is_instance(self._door_type)
                    y = self.y_lb + (
                        self.y_ub - self.y_lb
                    ) / 2 - self.wall_thickness_half + self.doorway_depth
                    width = (self.robot_radius + self.door_width_padding) * 2
                    height = (self.wall_thickness_half -
                              self.doorway_depth) * 2
                self._static_geom_cache[obj] = utils.Rectangle(x=x,
                                                               y=y,
                                                               width=width,
                                                               height=height,
                                                               theta=0)
        return self._static_geom_cache[obj]
