"""Environment for refinement cost learning with varying object counts."""

import logging
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Set, \
    Tuple

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


class ExitGarageEnv(BaseEnv):
    """An environment where a non-holonomic car/vehicle needs to be 'driven'
    out of a cluttered garage with some number of randomly positioned
    obstacles. A ceiling-mounted robot can be used to first 'clear' individual
    obstacles by moving them into a storage area.

    The action space is (car_vel, car_omega, robot_dx, robot_dy, robot_action):
        car_vel: linear velocity of the car (can be positive or negative)
        car_omega: steering angle of the car
        robot_dx, robot_dy: movement offsets for ceiling robot
        robot_action: if negative, do nothing; else, either place an object
            currently being carried or attempt to pick up an object at the
            robot's current location (and attempted movement is ignored)
    Note: only one of the car and robot can be operated at once, so if car_vel
        or car_omega are non-zero, then any robot-related actions are ignored
    """
    x_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 1.0
    y_lb: ClassVar[float] = 0.0
    y_ub: ClassVar[float] = 1.0

    robot_radius: ClassVar[float] = 0.05
    obstacle_radius: ClassVar[float] = 0.075
    car_width: ClassVar[float] = 0.12
    car_length: ClassVar[float] = 0.2

    exit_height: ClassVar[float] = 0.2
    exit_width: ClassVar[float] = 0.05
    exit_top: ClassVar[float] = 0.4
    storage_area_height: ClassVar[float] = 0.2
    robot_starting_x: ClassVar[float] = 0.1
    robot_starting_y: ClassVar[float] = 0.8
    obstacle_area_left_padding: ClassVar[float] = 0.4
    obstacle_area_right_padding: ClassVar[float] = 0.1
    obstacle_area_vertical_padding: ClassVar[float] = 0.05
    car_starting_x: ClassVar[float] = 0.15
    car_starting_y: ClassVar[float] = 0.3

    robot_action_magnitude: ClassVar[float] = 0.1
    car_max_absolute_vel: ClassVar[float] = 0.1
    car_steering_omega_limit: ClassVar[float] = 0.5

    # Types
    _car_type = Type("car", ["x", "y", "theta"])  # x, y, heading angle
    _robot_type = Type("robot", ["x", "y", "carrying"])  # carrying: bool
    _obstacle_type = Type("obstacle", ["x", "y", "carried"])  # carried: bool
    # Convenience type for storage area, storing number of obstacles in it
    # This is used in the ClearObstacle option to calculate where to place the
    # a new obstacle in the storage area.
    _storage_type = Type("storage", ["num_stored"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Predicates
        self._CarHasExited = Predicate("CarHasExited", [self._car_type],
                                       self._CarHasExited_holds)
        self._ObstacleCleared = Predicate("ObstacleCleared",
                                          [self._obstacle_type],
                                          self._ObstacleCleared_holds)
        self._ObstacleNotCleared = Predicate("ObstacleNotCleared",
                                             [self._obstacle_type],
                                             self._ObstacleNotCleared_holds)

        # Static objects (always exist no matter the settings)
        self._car = Object("big_car", self._car_type)
        self._robot = Object("robby", self._robot_type)
        self._storage = Object("store", self._storage_type)

        self._obstacles: List[Object] = []

        # Static _Geom2D for collision checking/rendering for exit
        self._exit_geom = utils.Rectangle(x=self.x_ub - self.exit_width,
                                          y=self.exit_top - self.exit_height,
                                          width=self.exit_width,
                                          height=self.exit_height,
                                          theta=0)

    @classmethod
    def get_name(cls) -> str:
        return "exit_garage"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        car_vel, car_omega, robot_dx, robot_dy, robot_action = action.arr
        next_state = state.copy()

        # Handle car movement if specified, ignoring robot if so
        if car_vel != 0 or car_omega != 0:
            cx = state.get(self._car, "x")
            cy = state.get(self._car, "y")
            theta = state.get(self._car, "theta")
            dx = np.cos(theta) * car_vel
            dy = np.sin(theta) * car_vel
            new_theta = theta + car_omega
            new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi  # wrap
            next_state.set(self._car, "x", cx + dx)
            next_state.set(self._car, "y", cy + dy)
            next_state.set(self._car, "theta", new_theta)
            # If this causes a collision, optionally raise failure
            collision_obj = self.get_car_collision_object(next_state)
            collision = collision_obj is not None
            if collision and CFG.exit_garage_raise_environment_failure:
                raise utils.EnvironmentFailure(
                    "Collision", info={"offending_objects": {collision_obj}})
            # If there is a collision or out-of-bounds, revert the move
            if collision or self.coords_out_of_bounds(cx + dx, cy + dy):
                next_state.set(self._car, "x", cx)
                next_state.set(self._car, "y", cy)
                next_state.set(self._car, "theta", theta)
            return next_state

        # Handle robot actions if car isn't moving
        rx = state.get(self._robot, "x")
        ry = state.get(self._robot, "y")

        # Pick or place if robot_action > 0
        if robot_action > 0:
            carried_obstacle = self._robot_carrying_obstacle(state)
            if carried_obstacle is None:
                # Pick up an obstacle if robot is over one
                object_to_pick = self._robot_picked_obstacle(state)
                if object_to_pick is not None:
                    next_state.set(object_to_pick, "carried", 1)
                    next_state.set(self._robot, "carrying", 1)
            else:
                # Place the current obstacle if in storage area and there is
                # no collision caused by doing so
                if ry > 1.0 - self.storage_area_height:
                    next_state.set(carried_obstacle, "x", rx)
                    next_state.set(carried_obstacle, "y", ry)
                    next_state.set(carried_obstacle, "carried", 0)
                    next_state.set(self._robot, "carrying", 0)
                    current_num_stored = state.get(self._storage, "num_stored")
                    next_state.set(self._storage, "num_stored",
                                   current_num_stored + 1)
            return next_state

        # No car movement or robot action, so just move robot unless it is out
        # of bounds
        new_rx = rx + robot_dx
        new_ry = ry + robot_dy
        if not self.coords_out_of_bounds(new_rx, new_ry):
            next_state.set(self._robot, "x", new_rx)
            next_state.set(self._robot, "y", new_ry)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._CarHasExited, self._ObstacleCleared, self._ObstacleNotCleared
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._CarHasExited}

    @property
    def types(self) -> Set[Type]:
        return {
            self._car_type, self._robot_type, self._obstacle_type,
            self._storage_type
        }

    @property
    def action_space(self) -> Box:
        # (car_vel, car_omega, robot_dx, robot_dy, robot_action)
        lb = np.array([
            -self.car_max_absolute_vel, -self.car_steering_omega_limit,
            -self.robot_action_magnitude, -self.robot_action_magnitude, -np.inf
        ],
                      dtype=np.float32)
        ub = np.array([
            self.car_max_absolute_vel, self.car_steering_omega_limit,
            self.robot_action_magnitude, self.robot_action_magnitude, np.inf
        ],
                      dtype=np.float32)
        return Box(lb, ub)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        car_color = "olive"
        robot_color = "green"
        obstacle_color = "red"
        carried_color = "darkred"
        storage_color = "blue"
        exit_color = "purple"

        # Draw car, storage area, and exit
        car_geom = self._object_to_geom(self._car, state)
        car_geom.plot(ax, color=car_color)
        storage_geom = self._object_to_geom(self._storage, state)
        storage_geom.plot(ax, color=storage_color)
        self._exit_geom.plot(ax, color=exit_color)

        # Draw obstacles
        carried_obstacle_geom: Optional[utils.Circle] = None
        for obstacle in state.get_objects(self._obstacle_type):
            if state.get(obstacle, "carried") == 1:
                # Obstacle is being carried, so draw it under the robot instead
                # of its stated position, and with a different color
                robot_x = state.get(self._robot, "x")
                robot_y = state.get(self._robot, "y")
                carried_obstacle_geom = utils.Circle(robot_x, robot_y,
                                                     self.obstacle_radius)
            else:
                # Obstacle is not being carried, just draw normally
                obstacle_geom = self._object_to_geom(obstacle, state)
                obstacle_geom.plot(ax, color=obstacle_color)
        if carried_obstacle_geom:
            carried_obstacle_geom.plot(ax, color=carried_color)

        # Draw robot
        robot_geom = self._object_to_geom(self._robot, state)
        robot_geom.plot(ax, color=robot_color)

        ax.set_xlim(self.x_lb - self.car_length, self.x_ub + self.car_length)
        ax.set_ylim(self.y_lb - self.car_length, self.y_ub + self.car_length)
        title = f"{car_color} = car, {robot_color} = robot, {exit_color} = exit"
        if caption is not None:
            title += f";\n{caption}"
        plt.suptitle(title, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        # There is only one goal in this environment.
        goal_atom = GroundAtom(self._CarHasExited, [self._car])
        goal = {goal_atom}

        def _sample_obstacle_position() -> Tuple[float, float]:
            return (rng.uniform(self.x_lb + self.obstacle_area_left_padding,
                                self.x_ub - self.obstacle_area_right_padding),
                    rng.uniform(
                        self.obstacle_area_vertical_padding,
                        self.y_ub - self.storage_area_height -
                        self.obstacle_area_vertical_padding))

        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            state_dict: Dict[Object, Dict[str, float]] = {
                self._car: {
                    "x": self.car_starting_x,
                    "y": self.car_starting_y,
                    "theta": 0,
                },
                self._robot: {
                    "x": self.robot_starting_x,
                    "y": self.robot_starting_y,
                    "carrying": 0,
                },
                self._storage: {
                    "num_stored": 0,  # nothing stored here from start
                },
            }

            # Generate a random number of obstacles
            num_obstacles = rng.integers(
                CFG.exit_garage_min_num_obstacles,
                CFG.exit_garage_max_num_obstacles + 1,
            )
            # Randomly generate obstacle positions to avoid collisions
            obstacle_geoms: List[utils.Circle] = []
            for i in range(num_obstacles):
                while True:
                    x, y = _sample_obstacle_position()
                    geom = utils.Circle(x, y, self.obstacle_radius)
                    # Check: if it collides with any other obstacle, resample
                    resample = False
                    for other_geom in obstacle_geoms:
                        if geom.intersects(other_geom):
                            resample = True
                            break
                    if not resample:
                        break
                obstacle_geoms.append(geom)
                # Add obstacle to state
                state_dict[Object(f"obstacle{i}", self._obstacle_type)] = {
                    "x": x,
                    "y": y,
                    "carried": 0
                }

            assert len(state_dict) == num_obstacles + 3
            state = utils.create_state_from_dict(state_dict)
            assert not goal_atom.holds(
                state
            ), "Error: goal is already satisfied in this state initialization"
            tasks.append(EnvironmentTask(state, goal))

        return tasks

    def _CarHasExited_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        car, = objects
        car_geom = self._object_to_geom(car, state)
        return car_geom.intersects(self._exit_geom)

    def _ObstacleCleared_holds(self, state: State,
                               objects: Sequence[Object]) -> bool:
        obstacle, = objects
        # Check obstacle is within storage area and is not being carried
        in_storage = state.get(obstacle,
                               "y") > self.y_ub - self.storage_area_height
        not_carried = state.get(obstacle, "carried") == 0
        return not_carried and in_storage

    def _ObstacleNotCleared_holds(self, state: State,
                                  objects: Sequence[Object]) -> bool:
        obstacle, = objects
        # Check obstacle is out of storage area and is not being carried
        in_storage = state.get(obstacle,
                               "y") > self.y_ub - self.storage_area_height
        not_carried = state.get(obstacle, "carried") == 0
        return not_carried and not in_storage

    @classmethod
    def coords_out_of_bounds(cls, new_x: float, new_y: float) -> bool:
        """Checks if coordinates are out of the bounds of the environment.

        This is made public because it is used both in simulate and in
        the externally-defined ground-truth options.
        """
        if (cls.x_lb <= new_x <= cls.x_ub) and (cls.y_lb <= new_y <= cls.y_ub):
            return False
        return True

    @classmethod
    def get_car_collision_object(cls, state: State) -> Optional[Object]:
        """Returns the object that the car has collided with, or None.

        This is made public because it is used both in simulate and in
        the externally-defined ground-truth options.
        """
        car, = state.get_objects(cls._car_type)
        car_geom = cls._object_to_geom(car, state)
        # Check for collisions with storage area.
        storage, = state.get_objects(cls._storage_type)
        storage_geom = cls._object_to_geom(storage, state)
        if car_geom.intersects(storage_geom):
            return storage
        # Check for collisions with obstacles
        for obstacle in state.get_objects(cls._obstacle_type):
            # Ignore this obstacle if it is being carried by the robot
            if state.get(obstacle, "carried") == 1:
                continue
            obstacle_geom = cls._object_to_geom(obstacle, state)
            if car_geom.intersects(obstacle_geom):
                return obstacle
        return None

    @classmethod
    def _robot_carrying_obstacle(cls, state: State) -> Optional[Object]:
        """If the robot is currently carrying an obstacle, return it; else
        return None if the robot isn't carrying anything."""
        robot, = state.get_objects(cls._robot_type)
        for obstacle in state.get_objects(cls._obstacle_type):
            if state.get(obstacle, "carried") == 1:
                return obstacle
        assert state.get(robot, "carrying") == 0
        return None  # not carrying anything

    @classmethod
    def _robot_picked_obstacle(cls, state: State) -> Optional[Object]:
        """Used when the robot is trying to pick up an obstacle at its current
        location; if one or more obstacles are not in storage and are touching
        the robot, return the closest one, else return None.

        Also return None if the robot is already carrying something or
        is currently in the storage area (and shouldn't be picking
        anything).
        """
        robot, = state.get_objects(cls._robot_type)
        rx = state.get(robot, "x")
        ry = state.get(robot, "y")
        if ry > cls.y_ub - cls.storage_area_height:
            return None  # robot in storage area
        object_to_pick: Optional[Object] = None
        closest_distance = (cls.robot_radius + cls.obstacle_radius)**2
        for obstacle in state.get_objects(cls._obstacle_type):
            ox = state.get(obstacle, "x")
            oy = state.get(obstacle, "y")
            squared_distance = (rx - ox)**2 + (ry - oy)**2
            # Set current object_to_pick if within range of robot and is
            # closest object so far
            if squared_distance < closest_distance:
                object_to_pick = obstacle
                closest_distance = squared_distance
        return object_to_pick

    @classmethod
    def _object_to_geom(cls, obj: Object, state: State) -> _Geom2D:
        """Converts objects to _Geom2D for collision checking, rendering."""
        # Storage area has a static position
        if obj.is_instance(cls._storage_type):
            return utils.Rectangle(x=cls.x_lb,
                                   y=cls.y_ub - cls.storage_area_height,
                                   width=cls.x_ub - cls.x_lb,
                                   height=cls.storage_area_height,
                                   theta=0)
        # Everything else has x and y properties
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        # Car
        if obj.is_instance(cls._car_type):
            theta = state.get(obj, "theta")
            # Need to rotate about the center of the car not the top corner
            # so create the translated Rectangle without rotation, then rotate
            geom = utils.Rectangle(x - cls.car_length / 2.0,
                                   y - cls.car_width / 2.0,
                                   cls.car_length,
                                   cls.car_width,
                                   theta=0)
            return geom.rotate_about_point(x, y, theta)
        # Robot
        if obj.is_instance(cls._robot_type):
            return utils.Circle(x, y, cls.robot_radius)
        # Obstacles
        assert obj.is_instance(cls._obstacle_type)
        return utils.Circle(x, y, cls.obstacle_radius)

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:

        logging.info(
            "Controls: click to move robot, use arrow keys for car, "
            "and press (g) to toggle the robot gripper. Press (q) to quit.")

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:
            if event.key == "q":
                raise utils.HumanDemonstrationFailure("Human quit.")

            if event.key == "g":
                return Action(
                    np.array([0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

            if event.key == "left":
                v = -self.car_max_absolute_vel
                return Action(
                    np.array([v, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

            if event.key == "right":
                v = self.car_max_absolute_vel
                return Action(
                    np.array([v, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

            if event.key == "up":
                omega = self.car_steering_omega_limit / 10.0
                return Action(
                    np.array([0.0, omega, 0.0, 0.0, 1.0], dtype=np.float32))

            if event.key == "down":
                omega = -self.car_steering_omega_limit / 10.0
                return Action(
                    np.array([0.0, omega, 0.0, 0.0, 1.0], dtype=np.float32))

            # Only remaining option is clicked.
            tx = event.xdata
            ty = event.ydata
            if (tx is None or ty is None):
                raise NotImplementedError("No valid action found.")

            robot_x = state.get(self._robot, "x")
            robot_y = state.get(self._robot, "y")
            dx = tx - robot_x
            dy = ty - robot_y
            mag = np.linalg.norm([dx, dy])
            if mag > self.car_max_absolute_vel:
                scale = self.car_max_absolute_vel / mag
                dx *= scale
                dy *= scale
            return Action(np.array([0.0, 0.0, dx, dy, 0.0], dtype=np.float32))

        return _event_to_action
