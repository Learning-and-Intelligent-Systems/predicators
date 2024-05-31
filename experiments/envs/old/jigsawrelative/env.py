import dataclasses
import itertools
import logging
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple, cast
from copy import deepcopy

import numpy as np
from experiments.envs.utils import BoxWH, construct_subcell_box, plot_geometry
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, State, Type
import gym
from shapely.geometry import Polygon, Point
from shapely.affinity import translate, rotate

from matplotlib import patches, pyplot as plt
import matplotlib
matplotlib.use("tkagg")


@dataclasses.dataclass(eq=False)
class SimulatorState():
    polys: Dict[Object, Polygon] = dataclasses.field(default_factory=dict)
    desired_poses: Dict[Object, Tuple[float, float, float]] = dataclasses.field(default_factory=dict)
    held_block: Optional[Object] = None
    dx: float = 0.0
    dy: float = 0.0

    def copy(self) -> 'SimulatorState':
        return dataclasses.replace(self)


class JigsawRelative(BaseEnv):
    """JigsawRelative environment"""

    # Settings
    ## Task generation settings
    num_tries: ClassVar[int] = 100000

    range_train_blocks: ClassVar[Tuple[int, int]] = (10, 10)
    range_test_blocks: ClassVar[Tuple[int, int]] = (10, 10)
    range_t_blocks: ClassVar[Tuple[int, int]] = (2, 4)

    ## World shape settings
    world_range_x: ClassVar[Tuple[float, float]] = (-1, 30)
    world_range_y: ClassVar[Tuple[float, float]] = (-1, 25)
    robot_point = Point(0, 0)
    block_placement_margin = 0.1

    container_x_pos: ClassVar[float] = 0.0
    container_y_pos: ClassVar[float] = 0.0

    blocks_start_x: ClassVar[float] = 3.0
    blocks_start_y: ClassVar[float] = 0.0
    sub_cell_size: ClassVar[float] = 1.0
    sub_cell_margin: ClassVar[float] = 0.2

    ## Predicate thresholds
    sub_block_present_thresh: ClassVar[float] = 0.5
    next_to_thresh: ClassVar[float] = 1.0
    holding_thresh: ClassVar[float] = 0.5

    # Types
    _robot_type = Type("robot", ["x", "y", "holding"])
    _object_type = Type("object", ["x", "y"])
    _container_type = Type("container", ["x", "y", "width", "height"], _object_type)
    _block_type = Type("block", ["x", "y", "orientation", "held"] + [f"sub({x},{y})" for x in range(2) for y in range(3)], _object_type)

    # Predicates
    ## Inside predicate
    @staticmethod
    def _Inside_holds(state: State, objects: Sequence[Object]) -> bool:
        container, block = objects
        return JigsawRelative._get_held_block(state) != block and \
            JigsawRelative._get_shape(state, container).contains(JigsawRelative._get_shape(state, block))

    _Inside: ClassVar[Predicate] = Predicate("Inside", [_container_type, _block_type], _Inside_holds)

    ## Outside predicate
    @staticmethod
    def _Outside_holds(state: State, objects: Sequence[Object]) -> bool:
        return not JigsawRelative._Inside_holds(state, objects)
    _Outside: ClassVar[Predicate] = Predicate("Outside", [_container_type, _block_type], _Outside_holds)

    # NextTo predicate
    @staticmethod
    def _Nextto_holds(state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        return JigsawRelative._get_shape(state, obj1).distance(JigsawRelative._get_shape(state, obj2)) <= JigsawRelative.next_to_thresh

    _NextToRobot: ClassVar[Predicate] = Predicate("NextToRobot", [_robot_type, _object_type], _Nextto_holds)

    # Held predicate
    @staticmethod
    def _Held_holds(state: State, objects: Sequence[Object]) -> bool:
        _, block, = objects
        return JigsawRelative._get_held_block(state) == block

    _Held: ClassVar[Predicate] = Predicate("Held", [_robot_type, _block_type], _Held_holds)

    # NotHeld predicate
    @staticmethod
    def _NotHeld_holds(state: State, objects: Sequence[Object]) -> bool:
        return JigsawRelative._get_held_block(state) is None

    _NotHeld: ClassVar[Predicate] = Predicate("NotHeld", [_robot_type], _NotHeld_holds)

    # Common Objects
    _robot = Object("robot", _robot_type)
    _container = Object("container", _container_type)

    # Common geometries
    @staticmethod
    def _construct_block_polygon(
        cells: List[bool],
        sub_cell_size: float,
        sub_cell_margin: float,
        sub_block_present_thresh: float
    ) -> Polygon:
        return construct_subcell_box(np.array(cells).reshape(3, 2).T, sub_cell_size).buffer(-sub_cell_margin, join_style='mitre')

    ## T-block
    t_block_cells = [True, True, True, False, True, False]
    t_block_poly = _construct_block_polygon(t_block_cells, sub_cell_size, sub_cell_margin, sub_block_present_thresh)

    ## Z-block
    z_block_cells = [True, True, False, False, True, True]
    z_block_poly = _construct_block_polygon(z_block_cells, sub_cell_size, sub_cell_margin, sub_block_present_thresh)

    ## S-block
    s_block_cells = [False, True, True, True, True, False]
    s_block_poly = _construct_block_polygon(s_block_cells, sub_cell_size, sub_cell_margin, sub_block_present_thresh)

    @classmethod
    def get_name(cls) -> str:
        return "jigsawrelative"

    def simulate(self, state: State, action: Action) -> State:
        global calls_to_simulate
        grab, move, place, _, _, _ = action.arr

        affinities = [
            (grab, self._transition_grab),
            (move, self._transition_move),
            (place, self._transition_place),
        ]
        _, transition_fn = max(affinities, key = lambda t: t[0])
        next_state = transition_fn(state, action)
        return next_state

    def _transition_grab(self, state: State, action: Action) -> State:
        logging.info("GRAB TRANSITION")
        _, _, _, x, y, _ = action.arr
        finger = Point(x, y)
        next_state = state.copy()

        # Check if the robot is holding anything
        if self._get_held_block(state) is not None:
            logging.info("ALREADY HOLDING A BLOCK")
            return next_state

        # Check which block was selected
        selected_blocks = [block for block in state.get_objects(self._block_type) if self._get_shape(state, block).contains_properly(finger)]
        if not selected_blocks:
            logging.info("NO BLOCK SELECTED")
            return next_state
        selected_block, = selected_blocks

        # Check if the block is not too far away
        if not JigsawRelative._Nextto_holds(state, [self._robot, selected_block]):
            logging.info("ROBOT NOT NEXT TO THE BLOCK")
            return next_state

        # Changing the state
        next_state.set(selected_block, "x", -self.sub_cell_size)
        next_state.set(selected_block, "y", -1.5 * self.sub_cell_size)
        self._set_held_block(next_state, selected_block)
        return next_state

    def _transition_move(self, state: State, action: Action) -> State:
        logging.info("MOVE TRANSITION")
        _, _, _, x, y, _ = action.arr
        next_state = state.copy()

        # Move the robot
        next_state.set(self._robot, "x", x)
        next_state.set(self._robot, "y", y)

        # Check if the robot is holding anything
        held_block = self._get_held_block(next_state)
        if held_block is not None:
            next_state.set(held_block, "x", x - self.sub_cell_size)
            next_state.set(held_block, "y", y - 1.5 * self.sub_cell_size)

        self._renormalize_state(next_state)
        return next_state

    def _transition_place(self, state: State, action: Action) -> State:
        logging.info("PLACE TRANSITION")
        _, _, _, x, y, orientation = action.arr
        next_state = state.copy()

        # Check if a block is held
        held_block = self._get_held_block(state)
        if held_block is None:
            logging.info("NO BLOCK HELD")
            return next_state

        # Check new placement
        new_block_shape = self._get_shape(
            state, held_block, x, y, orientation
        )
        container_shape = self._get_shape(state, self._container)
        if container_shape.boundary.intersects(new_block_shape):
            logging.info("BLOCK INTERSECTS WITH BOUNDARY")
            return next_state

        # Check collisions
        if any(self._get_shape(state, block).intersects(new_block_shape)
               for block in state.get_objects(self._block_type) if block != held_block):
            logging.info("BLOCK COLLIDES WITH OTHER BLOCKS")
            return next_state

        # Set the new state
        next_state.set(held_block, "x", x)
        next_state.set(held_block, "y", y)
        next_state.set(held_block, "orientation", orientation)
        self._set_held_block(next_state, None)

        # Check if the new placement isn't too far away
        if not JigsawRelative._Nextto_holds(next_state, [self._robot, held_block]):
            logging.info("ROBOT NOT NEXT TO THE BLOCK")
            return state.copy()

        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        if not self._train_tasks:
            self._train_tasks = self._generate_tasks(
                rng = self._train_rng,
                num_tasks = CFG.num_train_tasks,
                range_blocks = self.range_train_blocks,
            )
        return self._train_tasks

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if not self._test_tasks:
            self._test_tasks = self._generate_tasks(
                rng = self._test_rng,
                num_tasks = CFG.num_test_tasks,
                range_blocks = self.range_train_blocks,
            )
        return self._test_tasks

    def _generate_tasks(
        self,
        rng: np.random.Generator,
        num_tasks: int,
        range_blocks: Tuple[int, int]
    ) -> List[EnvironmentTask]:
        return [self._generate_task(rng, range_blocks) for _ in range(num_tasks)]

    def _generate_task(
        self,
        rng: np.random.Generator,
        range_blocks: Tuple[int, int],
    ) -> EnvironmentTask:
        num_blocks = rng.integers(*range_blocks, endpoint=True)
        num_t_blocks = rng.integers(*self.range_t_blocks, endpoint=True)

        # Generating the block order
        t_block_positions = rng.choice(num_blocks, num_t_blocks, replace=False)
        t_block_positions = sorted(t_block_positions[t_block_positions < num_blocks])
        z_starting_block = rng.choice([True, False])

        # Generating blocks data
        block_range_endpoints = [-1] + t_block_positions + [num_blocks]
        block_ids = rng.permutation(num_blocks)
        blocks_data = [(
                Object(f"z_block_{block_ids[block_pos]}", self._block_type), self.z_block_cells,
                self.z_block_poly, self.container_y_pos + (block_pos * 2), 0.0
            ) if z_block else (
                Object(f"s_block_{block_ids[block_pos]}", self._block_type), self.s_block_cells,
                self.s_block_poly, self.container_y_pos + (block_pos * 2), 0.0
            )
            for block_range_start, block_range_end, z_block in zip(
                block_range_endpoints, block_range_endpoints[1:], itertools.cycle([z_starting_block, not z_starting_block])
            )
            for block_pos in range(block_range_start + 1, block_range_end)
        ] + [(
            Object(f"t_block_{block_ids[block_pos]}", self._block_type), self.t_block_cells,
            self.t_block_poly, self.container_y_pos + (block_pos * 2), desired_orientation
        ) for block_pos, desired_orientation in zip(
            t_block_positions, itertools.cycle([0.0, 2.0][::(1 if z_starting_block else -1)]
        ))]
        blocks, _, _, _, _ = zip(*blocks_data)

        # Generating goal
        goal = {self._Inside([self._container, block]) for block in blocks}

        # Constructing placeholder state
        simulator_state = SimulatorState()
        state = State({obj: np.zeros((obj.type.dim,), dtype=np.float32) for obj in blocks + (self._robot, self._container)}, simulator_state)

        # Setting robot params
        state.set(self._robot, "x", 0)
        state.set(self._robot, "y", 0)
        state.set(self._robot, "holding", 0)
        simulator_state.polys[self._robot] = self.robot_point

        # Setting container params
        container_width = 2 * self.sub_cell_size
        container_height = (num_blocks * 2 + 1) * self.sub_cell_size
        state.set(self._container, "x", self.container_x_pos)
        state.set(self._container, "y", self.container_y_pos)
        state.set(self._container, "width", container_width)
        state.set(self._container, "height", container_height)
        simulator_state.polys[self._container] = BoxWH(
            self.container_x_pos, self.container_y_pos,
            container_width, container_height,
        )

        # Setting blocks params
        block_y_pos = self.container_x_pos
        for block_x_pos, (block, cells, poly, desired_y, desired_orientation) in zip(
            rng.permutation(np.arange(num_blocks) * (
                self.sub_cell_size * 2 + self.block_placement_margin
            ) + self.container_x_pos + self.sub_cell_size * 2), blocks_data
        ):
            state.data[block] = np.array([block_x_pos, block_y_pos, 0, 0] + cells, dtype=np.float32)
            simulator_state.polys[block] = poly
            simulator_state.desired_poses[block] = (self.container_x_pos, desired_y, desired_orientation)

        # Renromalizing the state to relative coordinates
        self._renormalize_state(state)

        return EnvironmentTask(state, goal)

    @classmethod
    def _get_shape(
        cls, state: State, obj: Object, x: Optional[float] = None,
        y: Optional[float] = None, orientation: Optional[float] = None
    ) -> Polygon:
        if x is None:
            x = state.get(obj, "x")
        if y is None:
            y = state.get(obj, "y")

        poly = cast(SimulatorState, state.simulator_state).polys[obj]
        if obj.is_instance(cls._block_type):
            if orientation is None:
                orientation = state.get(obj, "orientation")
            poly = rotate(poly, round(orientation) * 90)

        return translate(poly, x, y)

    @classmethod
    def _get_desired_pos(cls, state: State, obj: Object) -> Tuple[float, float, float]:
        simulator_state = cast(SimulatorState, state.simulator_state)
        dx, dy = simulator_state.dx, simulator_state.dy
        desired_x, desired_y, desired_orientation = simulator_state.desired_poses[obj]
        return desired_x + dx, desired_y + dy, desired_orientation

    @classmethod
    def _renormalize_state(cls, state: State) -> None:
        dx, dy = -state.get(cls._robot, "x"), -state.get(cls._robot, "y")
        for obj in state:
            state.set(obj, "x", state.get(obj, "x") + dx)
            state.set(obj, "y", state.get(obj, "y") + dy)
        simulator_state = cast(SimulatorState, state.simulator_state)
        simulator_state.dx += dx
        simulator_state.dy += dy

    @classmethod
    def _set_held_block(cls, state: State, held_block: Optional[Object]) -> None:
        simulator_state = cast(SimulatorState, state.simulator_state)
        old_held_block = simulator_state.held_block
        simulator_state.held_block = held_block
        if old_held_block is not None:
            state.set(old_held_block, "held", 0.0)
        if held_block is not None:
            state.set(held_block, "held", 1.0)
            state.set(cls._robot, "holding", 1.0)
        else:
            state.set(cls._robot, "holding", 0.0)

    @classmethod
    def _get_held_block(cls, state: State) -> Optional[Object]:
        held_block = cast(SimulatorState, state.simulator_state).held_block
        assert (state.get(cls._robot, "holding") >= cls.holding_thresh) ^ (held_block is None)
        assert held_block is None or state.get(held_block, "held") >= cls.holding_thresh
        return held_block

    @classmethod
    def _get_unnormalized_coordinates(cls, state: State, obj: Object) -> Tuple[float, float]:
        x, y = state.get(obj, "x"), state.get(obj, "y")
        simulator_state = cast(SimulatorState, state.simulator_state)
        dx, dy = simulator_state.dx, simulator_state.dy
        return x - dx, y - dy

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Inside, self._Outside, self._NextToRobot, self._Held, self._NotHeld}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Inside}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._object_type, self._container_type, self._block_type}

    @property
    def action_space(self) -> gym.spaces.Box:
        """(grab_action, move_action, place_action, x, y, new_block_orientation)"""
        move_range_x = self.world_range_x[1] - self.world_range_x[0]
        move_range_y = self.world_range_y[1] - self.world_range_y[0]
        lower_bound = np.array([0.0, 0.0, 0.0, -move_range_x, -move_range_y, 0.0], dtype=np.float32)
        upper_bound = np.array([1.0, 1.0, 1.0, move_range_x, move_range_y, 4.0-1e-6], dtype=np.float32)
        return gym.spaces.Box(lower_bound, upper_bound)

    @classmethod
    def _get_obj_patch(cls, state: State, obj: Object, **kwargs) -> patches.Patch:
        x, y = cls._get_unnormalized_coordinates(state, obj)
        return plot_geometry(cls._get_shape(state, obj, x, y), **kwargs)

    @classmethod
    def render_state_plt(
        cls,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None
    ) -> matplotlib.figure.Figure: # type: ignore
        fig = plt.figure()
        ax = fig.add_subplot()
        if caption:
            fig.suptitle(caption)

        # Drawing the container
        ax.add_patch(cls._get_obj_patch(state, cls._container, color='pink', linestyle='--', fill=False))

        # Drawing the blocks
        held_block = cls._get_held_block(state)
        for block in state.get_objects(cls._block_type):
            if block == held_block:
                continue
            x, y = cls._get_unnormalized_coordinates(state, block)
            ax.add_patch(cls._get_obj_patch(state, block, facecolor='green', edgecolor='darkgreen'))

        # Drawing the held block
        if held_block is not None:
            ax.add_patch(cls._get_obj_patch(state, held_block, edgecolor='darkgreen', linestyle=':', fill=False))

        # Drawing the robot
        ax.add_patch(cls._get_obj_patch(state, cls._robot, color='red'))

        ax.set_xlim(*cls.world_range_x)
        ax.set_ylim(*cls.world_range_y)
        return fig