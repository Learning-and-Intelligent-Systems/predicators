from copy import deepcopy
from dataclasses import dataclass
import logging
import math
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple, cast
import numpy as np
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, State, Type
from experiments.envs.utils import BoxWH
import gym

from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("tkagg")

from predicators.utils import abstract

__all__ = ['Statue']

@dataclass
class SimulatorState:
    world_width: int
    world_height: int
    vertical_doors: List[List[Object]]
    horizontal_doors: List[List[Object]]
    door_height_class: Dict[Object, bool] # True if Large (used for drawing)
    rooms_x_offset: int
    rooms_y_offset: int

class Statue(BaseEnv):
    """Statue environment"""

    # Settings
    ## Task generation settings
    range_train_world_size: ClassVar[Tuple[int, int]] = (2, 4)

    ## World shape settings
    range_small_door_width = (0.4, 0.6)
    range_small_door_height = (0.4, 0.6)
    range_large_door_width = (1.4, 1.8)
    range_large_door_height = (1.4, 1.8)

    range_statue_width = (0.1, 0.3)
    range_statue_height = (1.0, 1.3)
    range_statue_depth = (0.1, 0.3)

    room_size = 2.0
    world_generation_margin = 0.01

    robot_radius = 0.2 # Only for drawing
    statue_radius = 0.12 # Only for drawing

    ## Predicate thresholds
    door_vertical_thresh: ClassVar[float] = 0.5
    statue_vertical_thresh: ClassVar[float] = 0.5
    fingers_open_thresh: ClassVar[float] = 0.5
    held_thresh: ClassVar[float] = 0.5
    equality_margin = 1e-2

    # Types
    _room_type: ClassVar[Type] = Type("room", ["x", "y"])
    _door_type: ClassVar[Type] = Type("door", ["x", "y", "direction", "width", "height"])

    _movable_type: ClassVar[Type] = Type("movable", ["x", "y"])
    _statue_type: ClassVar[Type] = Type("statue", ["x", "y", "held", "grasp", "width", "height", "depth"], _movable_type)
    _robot_type: ClassVar[Type] = Type("robot", ["x", "y", "fingers"], _movable_type)

    # Predicates
    ## InRoom Predicate
    def _InRoom_holds(state: State, objects: Sequence[Object]) -> bool:
        movable, room = objects
        movable_x, movable_y = state.get(movable, "x"), state.get(movable, "y")
        room_x, room_y = state.get(room, "x"), state.get(room, "y")
        return room_x <= movable_x < room_x + Statue.room_size and \
            room_y <= movable_y < room_y + Statue.room_size
    _InRoom: ClassVar[Predicate] = Predicate("InRoom", [_movable_type, _room_type], _InRoom_holds)

    ## DoorwayFor Predicate
    def _DoorwayFor_holds(state: State, objects: Sequence[Object]) -> bool:
        door, room = objects
        door_x, door_y = state.get(door, "x"), state.get(door, "y")
        room_x, room_y = state.get(room, "x"), state.get(room, "y")
        door_vertical = state.get(door, "direction") >= Statue.door_vertical_thresh
        if door_vertical:
            return 0 <= room_x - door_x <= Statue.room_size + Statue.equality_margin and \
                np.allclose(door_y, room_y, atol=Statue.equality_margin)
        else:
            return np.allclose(door_x, room_x, atol=Statue.equality_margin) and 0 <= room_y - door_y <= Statue.room_size + Statue.equality_margin
    _DoorwayFor: ClassVar[Predicate] = Predicate("DoorwayFor", [_door_type, _room_type], _DoorwayFor_holds)

    ## Held Predicate
    def _Held_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, statue = objects
        robot_x, robot_y = state.get(robot, "x"), state.get(robot, "y")
        statue_x, statue_y = state.get(statue, "x"), state.get(statue, "y")

        fingers = state.get(robot, "fingers") >= Statue.fingers_open_thresh
        held = state.get(statue, "held") >= Statue.held_thresh
        if held:
            assert not fingers and np.allclose(robot_x, statue_x, atol=Statue.equality_margin) and \
                np.allclose(robot_y, statue_y, atol=Statue.equality_margin)
        else:
            assert fingers

        return held
    _Held: ClassVar[Predicate] = Predicate("Held", [_robot_type, _statue_type], _Held_holds)

    # Common objects
    _robot = Object("robot", _robot_type)
    _statue = Object("statue", _statue_type)

    @classmethod
    def get_name(cls) -> str:
        return "statue"

    @classmethod
    def simulate(cls, state: State, action: Action) -> State:
        move, grab_place = action.arr[3:5]
        affinities = [
            (move, cls._transition_move),
            (1 - grab_place, cls._transition_grab),
            (grab_place + 1, cls._transition_place),
        ]
        _, transition_fn = min(affinities, key = lambda t: t[0])
        return transition_fn(state, action)

    @classmethod
    def _transition_move(cls, state: State, action: Action) -> State:
        logging.info("TRANSITION MOVE")
        new_x, new_y = action.arr[0:2]
        next_state = state.copy()
        simulator_state = cls._get_simulator_state(state)
        mb_statue = cls._get_held_statue(state)

        # Check if the robot is not moving out of bounds
        robot_x, robot_y = state.get(cls._robot, "x"), state.get(cls._robot, "y")

        room_x = math.floor(robot_x / cls.room_size)
        room_y = math.floor(robot_y / cls.room_size)
        new_room_x = math.floor(new_x / cls.room_size)
        new_room_y = math.floor(new_y / cls.room_size)

        if new_room_x < simulator_state.rooms_x_offset or \
            new_room_y < simulator_state.rooms_y_offset or \
            new_room_x >= simulator_state.world_width + simulator_state.rooms_x_offset or \
            new_room_y >= simulator_state.world_height + simulator_state.rooms_y_offset:
            logging.info("ROBOT OOB")
            return next_state

        # Check if the robot is moving at most one space and the door through which it goes
        door = None
        if (new_room_x, new_room_y) == (room_x, room_y):
            horizontal = np.abs(robot_x - new_x) <= np.abs(robot_y - new_y)
        elif (new_room_x, new_room_y) in {(room_x - 1, room_y), (room_x + 1, room_y)}:
            horizontal = True
            if mb_statue is not None:
                door = simulator_state.vertical_doors[room_y - simulator_state.rooms_y_offset][
                    min(room_x, new_room_x) - simulator_state.rooms_x_offset
                ]
        elif (new_room_x, new_room_y) in {(room_x, room_y - 1), (room_x, room_y + 1)}:
            horizontal = False
            if mb_statue is not None:
                door = simulator_state.horizontal_doors[
                    min(room_y, new_room_y) - simulator_state.rooms_y_offset
                ][room_x - simulator_state.rooms_x_offset]
        else:
            logging.info("ROBOT NOT MOVING TO AN ADJACENT ROOM")
            return next_state

        # Check if the statue fits
        if door is not None and mb_statue is not None:
            door_width = state.get(door, "width")
            door_height = state.get(door, "height")
            statue_x_size, _, statue_z_size = \
                cls._get_statue_shape(state, mb_statue)
            if door_width < statue_x_size or door_height < statue_z_size:
                logging.info("STATUE DOES NOT FIT THROUGH THE DOORWAY")
                return next_state

        # Check if the statue does not go out of the room's bounds
        if mb_statue is not None:
            statue_x_size, statue_y_size, _ = cls._get_statue_shape(state, mb_statue, horizontal)
            room_box = BoxWH(new_room_x * cls.room_size, new_room_y * cls.room_size, cls.room_size, cls.room_size)
            statue_box = BoxWH(new_x - statue_x_size / 2, new_y - statue_y_size / 2, statue_x_size, statue_y_size)
            if not room_box.contains(statue_box):
                logging.info("STATUE OOB")
                return next_state

        # Move the robot
        next_state.set(cls._robot, "x", new_x)
        next_state.set(cls._robot, "y", new_y)
        if mb_statue is not None:
            next_state.set(mb_statue, "x", new_x)
            next_state.set(mb_statue, "y", new_y)

        # Move all the objects so that the robot is in room (0,0)
        cls._normalize_object_positions(next_state)

        return next_state

    @classmethod
    def _transition_grab(cls, state: State, action: Action) -> State:
        logging.info("TRANSITION GRAB")
        next_state = state.copy()
        grasp = action.arr[2]

        # Make sure the robot isn't holding anything
        if cls._get_held_statue(state) is not None:
            logging.info("STATUE ALREADY HELD")
            return next_state

        # Check if the robot and statue are in the same room
        robot_x, robot_y = state.get(cls._robot, "x"), state.get(cls._robot, "y")
        robot_room_x = math.floor(robot_x / cls.room_size)
        robot_room_y = math.floor(robot_y / cls.room_size)

        statue_x, statue_y = state.get(cls._statue, "x"), state.get(cls._statue, "y")
        statue_room_x = math.floor(statue_x / cls.room_size)
        statue_room_y = math.floor(statue_y / cls.room_size)

        if (robot_room_x, robot_room_y) != (statue_room_x, statue_room_y):
            logging.info("STATUE IN A DIFFERENT ROOM")
            return next_state

        # Check if the new placement of the statue won't interfere with the room
        next_state.set(cls._statue, "x", robot_x)
        next_state.set(cls._statue, "y", robot_y)
        next_state.set(cls._statue, "grasp", grasp)
        next_state.set(cls._statue, "held", 1.0)
        next_state.set(cls._robot, "fingers", 0.0)

        statue_x_size, statue_y_size, _ = cls._get_statue_shape(state, cls._statue)

        room_box = BoxWH(robot_room_x * cls.room_size, robot_room_y * cls.room_size, cls.room_size, cls.room_size)
        statue_box = BoxWH(robot_x - statue_x_size/2, robot_y - statue_y_size/2, statue_x_size, statue_y_size)

        if not room_box.contains(statue_box):
            logging.info("STATUE OOB")
            return state.copy()

        return next_state

    @classmethod
    def _transition_place(cls, state: State, action: Action) -> State:
        logging.info("TRANSITION PLACE")
        next_state = state.copy()

        # Make sure that a statue is held
        mb_statue = cls._get_held_statue(state)
        if mb_statue is None:
            logging.info("STATUE NOT HELD")
            return next_state

        # Place the statue
        next_state.set(mb_statue, "held", 0.0)
        next_state.set(cls._robot, "fingers", 1.0)

        return next_state

    @staticmethod
    def _get_simulator_state(state: State) -> SimulatorState:
        return cast(SimulatorState, state.simulator_state)

    @classmethod
    def _get_held_statue(cls, state: State) -> Optional[Object]:
        if state.get(cls._robot, "fingers") >= cls.fingers_open_thresh:
            return None

        statue, = [
            statue for statue in state.get_objects(cls._statue_type)
            if state.get(statue, "held") >= cls.held_thresh
        ]
        return statue

    @classmethod
    def _get_statue_shape(cls, state: State, statue: Object, horizontal: bool = False) -> Tuple[float, float, float]:
        """Outputs the projected size of the statue in the x, y and z axis"""
        assert statue.is_instance(cls._statue_type)
        statue_vertical = state.get(statue, "grasp") > cls.statue_vertical_thresh or \
            state.get(statue, "held") < cls.held_thresh
        statue_width = state.get(statue, "width")
        statue_depth = state.get(statue, "depth")
        statue_height = state.get(statue, "height")
        if not statue_vertical:
            statue_width, statue_height = statue_height, statue_width
        if horizontal:
            statue_width, statue_depth = statue_depth, statue_width
        return statue_width, statue_depth, statue_height

    @classmethod
    def _normalize_object_positions(cls, state: State) -> None:
        """Moves all the objects so that the robot is in the center room"""
        return
        simulator_state = cls._get_simulator_state(state)

        rooms_x_offset = -math.floor(state.get(cls._robot, "x") / cls.room_size)
        rooms_y_offset = -math.floor(state.get(cls._robot, "y") / cls.room_size)
        offset_x, offset_y = rooms_x_offset * cls.room_size, rooms_y_offset * cls.room_size

        simulator_state.rooms_x_offset += rooms_x_offset
        simulator_state.rooms_y_offset += rooms_y_offset
        for obj in state:
            assert obj.type.feature_names[:2] == ["x", "y"]
            state[obj][:2] += (offset_x, offset_y)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        if not self._train_tasks:
            self._train_tasks = self._generate_tasks(
                rng = self._train_rng,
                num_tasks = CFG.num_train_tasks,
                range_world_size = self.range_train_world_size,
            )
        return self._train_tasks

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if not self._test_tasks:
            self._test_tasks = self._generate_tasks(
                rng = self._test_rng,
                num_tasks = CFG.num_test_tasks,
                range_world_size = (CFG.statue_test_world_size, CFG.statue_test_world_size),
            )
        return self._test_tasks

    def _generate_tasks(
        self,
        rng: np.random.Generator,
        num_tasks: int,
        range_world_size: Tuple[int, int],
    ) -> List[EnvironmentTask]:
        return [
            self._generate_task(
                rng,
                range_world_size,
            ) for _ in range(num_tasks)
        ]

    def _generate_task(
        self,
        rng: np.random.Generator,
        range_world_size: Tuple[int, int],
    ) -> EnvironmentTask:
        world_width = rng.integers(*range_world_size, endpoint=True)
        world_height = rng.integers(*range_world_size, endpoint=True)
        rooms_x_offset = rng.integers(0, max(self.range_train_world_size[1], CFG.statue_test_world_size) - world_width, endpoint=True)
        rooms_y_offset = rng.integers(0, max(self.range_train_world_size[1], CFG.statue_test_world_size) - world_height, endpoint=True)

        # Generating objects
        rooms = [[Object(f"room_{x}_{y}", self._room_type) for x in range(world_width)] for y in range(world_height)]

        vert_ids = rng.permutation((world_width - 1) * world_height).reshape((world_width - 1, world_height))
        vertical_doors = [[Object(f"doorvert_{vert_ids[x, y]}_{x}_{y}", self._door_type) for x in range(world_width - 1)] for y in range(world_height)]

        horiz_ids = rng.permutation(world_width * (world_height - 1)).reshape((world_width, world_height - 1))
        horizontal_doors = [[Object(f"doorhoriz_{horiz_ids[x, y]}_{x}_{y}", self._door_type) for x in range(world_width)] for y in range(world_height - 1)]

        # Generating goal
        goal = {self._InRoom([self._statue, rooms[-1][-1]])}

        # Constructing placeholder state
        door_height_class = {}
        state: State = State({
            obj: np.zeros((obj.type.dim,), dtype=np.float32)
            for obj in sum(rooms, []) + sum(vertical_doors, []) + sum(horizontal_doors, []) + [self._robot, self._statue]
        }, SimulatorState(world_width, world_height, vertical_doors, horizontal_doors, door_height_class, rooms_x_offset, rooms_y_offset))

        # Setting statue params
        statue_width = rng.uniform(*self.range_statue_width)
        statue_height = rng.uniform(*self.range_statue_height)
        statue_depth = rng.uniform(*self.range_statue_depth)
        state.set(self._statue, "x", rng.random())
        state.set(self._statue, "y", rng.random())
        state.set(self._statue, "width", statue_width)
        state.set(self._statue, "height", statue_height)
        state.set(self._statue, "depth", statue_depth)
        state.set(self._statue, "held", 0.0)
        state.set(self._statue, "grasp", 1.0)

        # Setting robot position
        while True:
            robot_x = rng.uniform(0, world_width * self.room_size)
            robot_y = rng.uniform(0, world_height * self.room_size)
            if robot_x % self.room_size > self.room_size - max(statue_width, statue_depth) - self.world_generation_margin or\
                robot_x % self.room_size < max(statue_width, statue_depth) + self.world_generation_margin:
                continue
            if robot_y % self.room_size > self.room_size - statue_depth - self.world_generation_margin or\
                robot_y % self.room_size < statue_depth + self.world_generation_margin:
                continue
            break

        state.set(self._robot, "x", robot_x)
        state.set(self._robot, "y", robot_y)
        state.set(self._robot, "fingers", 1.0)

        # Setting positions of rooms
        for y, rooms_row in enumerate(rooms):
            for x, room in enumerate(rooms_row):
                state.set(room, "x", x * self.room_size)
                state.set(room, "y", y * self.room_size)

        # Setting default positions and sizes of doors
        for direction, doors in enumerate([horizontal_doors, vertical_doors]):
            for y, doors_row in enumerate(doors):
                for x, door in enumerate(doors_row):
                    state.set(door, "x", x * self.room_size)
                    state.set(door, "y", y * self.room_size)
                    state.set(door, "direction", direction)
                    state.set(door, "width", rng.uniform(*self.range_large_door_width))
                    state.set(door, "height", rng.uniform(*self.range_large_door_height))
                    door_height_class[door] = True

        # Figuring out the small doors
        small_doors = []
        if rng.choice([True, False]): # Should the first door be vertical
            x = rng.integers(1, world_width)
            y = 0
            small_doors.append(vertical_doors[y][x-1])
            y += 1
        else:
            x = world_width
            y = rng.integers(1, world_height)
            small_doors.append(horizontal_doors[y-1][x-1])
            x -= 1

        while x != 0 and y != world_height:
            if rng.choice([True, False]):
                small_doors.append(vertical_doors[y][x-1])
                y += 1
            else:
                small_doors.append(horizontal_doors[y-1][x-1])
                x -= 1

        # Setting the small doors
        small_doors_orientation = rng.choice([True, False], len(small_doors))
        while np.all(small_doors_orientation) or np.all(np.logical_not(small_doors_orientation)):
            small_doors_orientation = rng.choice([True, False], len(small_doors))

        for small_door, orientation in zip(small_doors, small_doors_orientation):#, strict=True):
            if orientation:
                state.set(small_door, "height", rng.uniform(*self.range_small_door_height))
                door_height_class[small_door] = False
            else:
                state.set(small_door, "width", rng.uniform(*self.range_small_door_width))

        # Adjusting the position of everything to accommodate the offset
        for obj in state:
            assert obj.type.feature_names[:2] == ["x", "y"]
            state[obj][:2] += (rooms_x_offset * self.room_size, rooms_y_offset * self.room_size)

        # Normalizing the positions of everything with respect to the robot
        self._normalize_object_positions(state)

        return EnvironmentTask(state, goal)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InRoom, self._DoorwayFor, self._Held}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InRoom}

    @property
    def types(self) -> Set[Type]:
        return {self._room_type, self._door_type, self._movable_type, self._statue_type, self._robot_type}

    @property
    def action_space(self) -> gym.spaces.Box:
        """(x, y, grasp, move, grab_place)"""
        max_x = max(self.range_train_world_size[1], CFG.statue_test_world_size) * self.room_size - self.equality_margin
        max_y = max(self.range_train_world_size[1], CFG.statue_test_world_size) * self.room_size - self.equality_margin
        lower_bound = np.array([-max_x, -max_y, 0.0, 0.0, -1.0], dtype=np.float32)
        upper_bound = np.array([max_x, max_y, 1.0, 1.0, 1.0], dtype=np.float32)
        return gym.spaces.Box(lower_bound, upper_bound)

    @classmethod
    def render_state_plt(
        cls,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None
    ) -> matplotlib.figure.Figure:
        fig = plt.figure()
        ax = fig.add_subplot()
        simulator_state = cls._get_simulator_state(state)
        fig.suptitle(caption)

        # Draw the horizontal doors
        for y, doors_row in enumerate(simulator_state.horizontal_doors):
            for x, door in enumerate(doors_row):
                w = state.get(door, "width")
                ax.add_line(Line2D(
                    [x * cls.room_size + (cls.room_size - w) / 2, x * cls.room_size + (cls.room_size + w) / 2],
                    [(y + 1) * cls.room_size] * 2,
                    color='green' if simulator_state.door_height_class[door] else 'red'
                )) # TODO: improve door visuals

        # Draw the vertical doors
        for y, doors_row in enumerate(simulator_state.vertical_doors):
            for x, door in enumerate(doors_row):
                w = state.get(door, "width")
                ax.add_line(Line2D(
                    [(x + 1) * cls.room_size] * 2,
                    [y * cls.room_size + (cls.room_size - w) / 2, y * cls.room_size + (cls.room_size + w) / 2],
                    color='green' if simulator_state.door_height_class[door] else 'red'
                )) # TODO: improve door visuals

        # Draw the robot
        robot_x, robot_y = state.get(cls._robot, "x"), state.get(cls._robot, "y")
        robot_x -= simulator_state.rooms_x_offset * cls.room_size
        robot_y -= simulator_state.rooms_y_offset * cls.room_size
        ax.add_patch(patches.Circle((robot_x, robot_y), cls.robot_radius, color='yellow'))

        # Draw the statue
        statue_x, statue_y = state.get(cls._statue, "x"), state.get(cls._statue, "y")
        statue_x -= simulator_state.rooms_x_offset * cls.room_size
        statue_y -= simulator_state.rooms_y_offset * cls.room_size
        ax.add_patch(patches.Circle((statue_x, statue_y), cls.statue_radius, color='black'))

        ax.autoscale_view()

        return fig