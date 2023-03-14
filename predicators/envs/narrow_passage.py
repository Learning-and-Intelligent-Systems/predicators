"""Toy environment for testing refinement cost heuristic learning."""

from typing import ClassVar, List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type
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
    door_sensor_radius: ClassVar[float] = 0.2
    passage_x_pos: ClassVar[float] = 0.7
    wall_thickness_half: ClassVar[float] = 0.15 / 2
    doorway_depth: ClassVar[float] = 1e-2
    init_pos_margin: ClassVar[float] = 1e-3

    action_magnitude: ClassVar[float] = 0.1

    # Types
    _robot_type = Type("robot", ["x", "y"])
    _target_type = Type("target", ["x", "y"])
    _wall_type = Type("wall", ["x", "width"])
    _door_type = Type("door", ["x", "width", "open"])
    # Type for the region within which the robot must be located
    # in order for the door-opening action to work.
    _door_sensor_type = Type("door_sensor", ["x"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Predicates
        self._TouchedGoal = Predicate("TouchedGoal",
                                      [self._robot_type, self._target_type],
                                      self._TouchedGoal_holds)
        self._DoorIsOpen = Predicate("DoorIsOpen", [self._door_type],
                                     self._DoorIsOpen_holds)
        self._DoorIsClosed = Predicate("DoorIsClosed", [self._door_type],
                                       self._DoorIsClosed_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._target = Object("target", self._target_type)
        self._walls = [Object(f"wall{i}", self._wall_type) for i in range(3)]
        self._door = Object("door", self._door_type)
        self._door_sensor = Object("door_sensor", self._door_sensor_type)

    @classmethod
    def get_name(cls) -> str:
        return "narrow_passage"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        dx, dy, open_door = action.arr
        next_state = state.copy()
        if open_door > 0:
            # Open door if within the door sensor range
            if self.robot_near_door(state):
                next_state.set(self._door, "open", 1)
            return next_state
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        new_x = x + dx
        new_y = y + dy
        next_state.set(self._robot, "x", new_x)
        next_state.set(self._robot, "y", new_y)
        # Check for collisions.
        if self.state_has_collision(next_state) or self._coords_out_of_bounds(
                new_x, new_y):
            # Revert the change to the robot position.
            next_state.set(self._robot, "x", x)
            next_state.set(self._robot, "y", y)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
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
            task: EnvironmentTask,
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

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        # There is only one goal in this environment.
        goal_atom = GroundAtom(self._TouchedGoal, [self._robot, self._target])
        goal = {goal_atom}

        # The initial positions of the robot and target vary, while wall and
        # door y positions are fixed. The robot should be above the walls, while
        # the dot should be below the walls (y coordinate)
        y_mid = (self.y_ub - self.y_lb) / 2 + self.y_lb
        margin = self.wall_thickness_half + self.init_pos_margin

        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            # Door width is generated randomly per task
            door_width_padding = rng.uniform(
                CFG.narrow_passage_door_width_padding_lb,
                CFG.narrow_passage_door_width_padding_ub,
            )
            door_width = (self.robot_radius + door_width_padding) * 2
            # Passage width is generated randomly per task
            passage_width_padding = rng.uniform(
                CFG.narrow_passage_passage_width_padding_lb,
                CFG.narrow_passage_passage_width_padding_ub,
            )
            passage_width = (self.robot_radius + passage_width_padding) * 2

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
                    "width": door_width,
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
            tasks.append(EnvironmentTask(state, goal))
        return tasks

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

    @staticmethod
    def _DoorIsClosed_holds(state: State, objects: Sequence[Object]) -> bool:
        door = objects[0]
        return state.get(door, "open") == 0

    def _coords_out_of_bounds(self, new_x: float, new_y: float) -> bool:
        if (self.x_lb <= new_x <= self.x_ub) and (self.y_lb <= new_y <=
                                                  self.y_ub):
            return False
        return True

    @classmethod
    def state_has_collision(cls, state: State) -> bool:
        """This is made public because it is used both in simulate and in the
        externally-defined ground-truth options."""
        robot, = state.get_objects(cls._robot_type)
        robot_geom = cls._object_to_geom(robot, state)
        # Check for collisions with obstacles.
        for obstacle in state.get_objects(cls._wall_type):
            obstacle_geom = cls._object_to_geom(obstacle, state)
            if robot_geom.intersects(obstacle_geom):
                return True
        # Check for collisions with closed doors.
        door, = state.get_objects(cls._door_type)
        if cls._DoorIsClosed_holds(state, [door]):
            door_geom = cls._object_to_geom(door, state)
            if robot_geom.intersects(door_geom):
                return True
        return False

    @classmethod
    def robot_near_door(cls, state: State) -> bool:
        """Returns true if the robot is within range of the door sensor."""
        robot, = state.get_objects(cls._robot_type)
        robot_geom = cls._object_to_geom(robot, state)
        # Check for "collision" with door sensor
        door_sensor, = state.get_objects(cls._door_sensor_type)
        door_sensor_geom = cls._object_to_geom(door_sensor, state)
        return robot_geom.intersects(door_sensor_geom)

    @classmethod
    def _object_to_geom(cls, obj: Object, state: State) -> _Geom2D:
        """Adapted from doors.py."""
        x = state.get(obj, "x")
        if (obj.is_instance(cls._robot_type)
                or obj.is_instance(cls._target_type)):
            y = state.get(obj, "y")
            return utils.Circle(x, y, cls.robot_radius)
        if obj.is_instance(cls._door_sensor_type):
            y = cls.y_lb + (cls.y_ub - cls.y_lb) / 2
            return utils.Circle(x, y, cls.door_sensor_radius)
        if obj.is_instance(cls._wall_type):
            y = cls.y_lb + (cls.y_ub - cls.y_lb) / 2 - cls.wall_thickness_half
            width = state.get(obj, "width")
            height = cls.wall_thickness_half * 2
        else:
            assert obj.is_instance(cls._door_type)
            y = cls.y_lb + (cls.y_ub - cls.y_lb
                            ) / 2 - cls.wall_thickness_half + cls.doorway_depth
            width = state.get(obj, "width")
            height = (cls.wall_thickness_half - cls.doorway_depth) * 2
        return utils.Rectangle(x=x, y=y, width=width, height=height, theta=0)
