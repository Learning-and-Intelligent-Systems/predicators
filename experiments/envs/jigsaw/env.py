from dataclasses import dataclass
import itertools
import logging
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from experiments.envs.utils import BoxWH, plot_geometry
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, State, Task, Type
import gym
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import translate, rotate

from matplotlib import patches, pyplot as plt
import matplotlib
from copy import deepcopy

class SimulatorState():
    def __init__(self):
        self.polys = {}
        self.desired_poses = {}


class Jigsaw(BaseEnv):
    """Jigsaw environment"""

    # Settings
    ## Task generation settings
    num_tries: ClassVar[int] = 100000

    range_train_blocks: ClassVar[Tuple[int, int]] = (8, 8)
    range_test_blocks: ClassVar[Tuple[int, int]] = (8, 8)
    range_t_blocks: ClassVar[Tuple[int, int]] = (1, 3)

    ## World shape settings
    world_range_x: ClassVar[Tuple[float, float]] = (-1, 30)
    world_range_y: ClassVar[Tuple[float, float]] = (-1, 25)
    block_placement_margin = 0.1

    container_x_pos: ClassVar[float] = 0.0
    container_y_pos: ClassVar[float] = 0.0

    blocks_start_x: ClassVar[float] = 3.0
    blocks_start_y: ClassVar[float] = 0.0
    sub_cell_size: ClassVar[float] = 1.0
    sub_cell_margin: ClassVar[float] = 0.1

    ## Predicate thresholds
    sub_block_present_thresh: ClassVar[float] = 0.5

    # Types
    _container_type = Type("container", ["x", "y", "width", "height"])
    _block_type = Type("block", ["x", "y", "orientation"] + [f"sub({x},{y})" for x in range(2) for y in range(3)])

    # Predicates
    ## Inside predicate
    @staticmethod
    def _Inside_holds(state: State, objects: Sequence[Object]) -> bool:
        container, block = objects
        return Jigsaw._get_shape(state, container).contains(Jigsaw._get_shape(state, block))

    _Inside: ClassVar[Predicate] = Predicate("Inside", [_container_type, _block_type], _Inside_holds)

    ## Outside predicate
    @staticmethod
    def _Outside_holds(state: State, objects: Sequence[Object]) -> bool:
        return not Jigsaw._Inside_holds(state, objects)
    _Outside: ClassVar[Predicate] = Predicate("Outside", [_container_type, _block_type], _Outside_holds)


    # Common Objects
    _container = Object("container", _container_type)

    # Common geometries
    @staticmethod
    def _construct_block_polygon(
        cells: List[float],
        sub_cell_size: float,
        sub_cell_margin: float,
        sub_block_present_thresh: float
    ) -> Polygon:
        main_polygon = BoxWH(0, 0, 2 * sub_cell_size, 3 * sub_cell_size)
        for (sx, sy), cell in zip(itertools.product(range(2), range(3)), cells):
            if cell >= sub_block_present_thresh:
                continue
            dx, dy, dw, dh = (sx * sub_cell_size, sy * sub_cell_size, sub_cell_size * 2, sub_cell_size)
            if sx == 0:
                dx -= sub_cell_size
            if sy in {0, 2}:
                dh += sub_cell_size
            if sy == 0:
                dy -= sub_cell_size
            main_polygon = main_polygon.difference(BoxWH(dx, dy, dw, dh))
        return main_polygon.buffer(-sub_cell_margin, join_style='mitre')

    ## L-block
    l_block_cells = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    l_block_poly = _construct_block_polygon(l_block_cells, sub_cell_size, sub_cell_margin, sub_block_present_thresh)

    ## T-block
    t_block_cells = [1.0, 1.0, 1.0, 0, 1.0, 0]
    t_block_poly = _construct_block_polygon(t_block_cells, sub_cell_size, sub_cell_margin, sub_block_present_thresh)

    ## Z-block
    z_block_cells = [1.0, 1.0, 0, 0, 1.0, 1.0]
    z_block_poly = _construct_block_polygon(z_block_cells, sub_cell_size, sub_cell_margin, sub_block_present_thresh)

    ## S-block
    s_block_cells = [0, 1.0, 1.0, 1.0, 1.0, 0]
    s_block_poly = _construct_block_polygon(s_block_cells, sub_cell_size, sub_cell_margin, sub_block_present_thresh)

    @classmethod
    def get_name(cls) -> str:
        return "jigsaw"

    def simulate(self, state: State, action: Action) -> State:
        logging.info("TRANSITION")
        old_x, old_y, new_x, new_y, orientation = action.arr
        next_state = state.copy()

        # Check which block was selected
        finger = Point(old_x, old_y)
        selected_blocks = [block for block in state.get_objects(self._block_type) if self._get_shape(state, block).contains_properly(finger)]
        if not selected_blocks:
            logging.info("NO BLOCK SELECTED")
            return next_state
        selected_block, = selected_blocks

        # Check new placement
        new_block_shape = self._get_shape(
            state, selected_block, new_x, new_y, orientation
        )
        container_shape = self._get_shape(state, self._container)
        if container_shape.boundary.intersects(new_block_shape):
            logging.info("BLOCK INTERSECTS WITH BOUNDARY")
            return next_state

        # Check collisions
        if any(self._get_shape(state, block).intersects(new_block_shape)
               for block in state.get_objects(self._block_type) if block != selected_block):
            logging.info("BLOCK COLLIDES WITH OTHER BLOCKS")
            return next_state

        # Move the block
        next_state.set(selected_block, "x", new_x)
        next_state.set(selected_block, "y", new_y)
        next_state.set(selected_block, "orientation", orientation)

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
        # task = self._generate_task(rng, range_blocks)
        # return [self._modify_task(task, rng) for _ in range(num_tasks)]
        return [self._generate_task(rng, range_blocks) for _ in range(num_tasks)]

    def _modify_task(self, task: EnvironmentTask, rng: np.random.Generator) -> EnvironmentTask:
        # task = deepcopy(task)
        # blocks = task.init.get_objects(self._block_type)
        # block_ids = rng.permutation(len(blocks))
        # for block_id, block in zip(block_ids, blocks):
        #     block.name = f"{block_id}_" + block.name
        return task

    def _generate_task(
        self,
        rng: np.random.Generator,
        range_blocks: Tuple[int, int],
    ) -> EnvironmentTask:
        num_blocks = rng.integers(*range_blocks, endpoint=True)
        assert num_blocks >= 3
        num_t_blocks = rng.integers(*self.range_t_blocks, endpoint=True)
        num_big_blocks = num_blocks - 2

        # Generating the block order
        t_block_positions = rng.choice(num_blocks - 2, num_t_blocks, replace=False)
        t_block_positions = sorted(t_block_positions[t_block_positions < num_big_blocks])
        z_starting_block = rng.choice([True, False])

        # Generating blocks data
        t_block_cells = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0]
        z_block_cells = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        s_block_cells = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
        l_block_cells = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]

        block_range_endpoints = [-1] + t_block_positions + [num_blocks - 2]
        block_ids = rng.permutation(num_blocks)
        blocks_data = [(
            Object(f"{block_ids[0]}_l_block_0", self._block_type), self.l_block_cells, self.l_block_poly,
            self.container_y_pos, 1.0 if z_starting_block else 0.0
        )] + sum(([(
                Object(
                    f"{block_ids[block_id + 1]}_z_block_{block_id + 1}"
                    if z_blocks else f"{block_ids[block_id + 1]}_s_block_{block_id + 1}", self._block_type
                ), self.z_block_cells if z_blocks else self.s_block_cells,
                self.z_block_poly if z_blocks else self.s_block_poly,
                self.container_y_pos + (block_id * 2 + 1) * self.sub_cell_size, 0.0
            ) for block_id in range(block_range_start + 1, block_range_end)] + ([(
                Object(f"{block_ids[block_range_start + 1]}_t_block_{block_range_start + 1}", self._block_type), self.t_block_cells,
                self.t_block_poly, self.container_y_pos + (block_range_start * 2 + 1) * self.sub_cell_size, 2.0 if z_blocks else 0.0
            )] if block_range_start != -1 else [])
            for block_range_start, block_range_end, z_blocks in zip(
                block_range_endpoints, block_range_endpoints[1:], itertools.cycle([z_starting_block, not z_starting_block])
            )), []
        ) + [(
            Object(f"{block_ids[-1]}_l_block_{num_blocks - 1}", self._block_type), self.l_block_cells,
            self.l_block_poly, self.container_y_pos + (num_blocks * 2 - 3) * self.sub_cell_size,
            2.0 if z_starting_block ^ (len(t_block_positions) % 2 == 0) else 3.0
        )]
        blocks, _, _, _, _ = zip(*blocks_data)

        # Generating goal
        goal = {self._Inside([self._container, block]) for block in blocks}

        # Constructing placeholder state
        simulator_state = SimulatorState()
        state = State({obj: np.zeros((obj.type.dim,), dtype=np.float32) for obj in blocks + (self._container,)}, simulator_state)

        # Setting container params
        container_width = 2 * self.sub_cell_size
        container_height = (num_blocks * 2 - 1) * self.sub_cell_size
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
            state.data[block] = np.array([block_x_pos, block_y_pos, 0.0] + cells, dtype=np.float32)
            simulator_state.polys[block] = poly
            simulator_state.desired_poses[block] = (self.container_x_pos, desired_y, desired_orientation)

        return EnvironmentTask(state, goal)

    @classmethod
    def _get_shape(
        cls, state: State, obj: Object, x: Optional[float] = None,
        y: Optional[float] = None, orientation: Optional[float] = None
    ) -> Polygon:
        poly = cast(SimulatorState, state.simulator_state).polys[obj]
        if not obj.is_instance(cls._block_type):
            return poly

        if x is None:
            x = state.get(obj, "x")
        if y is None:
            y = state.get(obj, "y")
        if orientation is None:
            orientation = state.get(obj, "orientation")

        return translate(rotate(poly, round(orientation) * 90), x, y)

    @classmethod
    def _get_desired_pos(cls, state: State, obj: Object) -> Tuple[float, float, float]:
        return cast(SimulatorState, state.simulator_state).desired_poses[obj]

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Inside, self._Outside}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Inside}

    @property
    def types(self) -> Set[Type]:
        return {self._container_type, self._block_type}

    @property
    def action_space(self) -> gym.spaces.Box:
        """(current_block_x, current_block_y, new_block_x, new_block_y, new_block_orientation)"""
        lower_bound = np.array([self.world_range_x[0], self.world_range_y[0]] * 2 + [0], dtype=np.float32)
        upper_bound = np.array([self.world_range_x[1], self.world_range_y[1]] * 2 + [4-1e-6], dtype=np.float32)
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
        fig.suptitle(caption)

        # Drawing the container
        ax.add_patch(plot_geometry(cls._get_shape(state, cls._container), color='pink', linestyle='--', fill=False))

        # Drawing the blocks
        for block in state.get_objects(cls._block_type):
            ax.add_patch(plot_geometry(cls._get_shape(state, block), facecolor='green', edgecolor='darkgreen'))

        ax.set_xlim(*cls.world_range_x)
        ax.set_ylim(*cls.world_range_y)
        return fig